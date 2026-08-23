from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def _cpu_placement(cpu_ids):
    cpus = list(cpu_ids)
    threads = len(cpus)
    return {
        "schema": "summit.openmp_placement_attestation.v1",
        "schema_version": 1,
        "verified": True,
        "immutable": True,
        "requested_threads": threads,
        "expected_cpu_ids": cpus,
        "omp_dynamic": False,
        "omp_thread_limit": threads,
        "omp_max_active_levels": 1,
        "omp_proc_bind": "spread",
        "omp_binding_active": True,
        "omp_num_places": threads,
        "effective_openmp_capacity": threads,
        "place_cpu_ids": [[cpu] for cpu in cpus],
        "team_size": threads,
        "exact_singleton_places": True,
        "exact_team_coverage": True,
        "workers": [
            {
                "thread_num": index,
                "place_num": index,
                "place_cpu_ids": [cpu],
                "sched_affinity_cpu_ids": [cpu],
                "current_cpu": cpu,
                "verified": True,
            }
            for index, cpu in enumerate(cpus)
        ],
        "vendor_calls": 0,
    }


def _placement_build_info(placement):
    return {
        "api_version": 9,
        "backend_version": "1.9",
        "openmp_effective_capacity_policy": (
            "bound_places_else_sched_affinity_v1"
        ),
        "openmp_placement_contract_supported": True,
        "openmp_placement_contract_schema": (
            "summit.openmp_placement_attestation.v1"
        ),
        "openmp_placement_contract_configured": True,
        "openmp_placement_contract_immutable": True,
        "openmp_placement_probe_vendor_calls": 0,
        "openmp_placement_contract_evidence": placement,
        "blas_runtime_threading_layer": "openmp",
    }


def test_thread_preparse_uses_build_handoff_default_and_respects_overrides(
    monkeypatch,
):
    from summit import cli

    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    monkeypatch.delenv("GOMP_SPINCOUNT", raising=False)
    monkeypatch.delenv("OPENBLAS_THREAD_TIMEOUT", raising=False)
    cli._set_thread_env_vars(3)
    assert os.environ["OMP_WAIT_POLICY"] == cli.RECOMMENDED_OMP_WAIT_POLICY
    if cli.RECOMMENDED_OMP_WAIT_POLICY == "PASSIVE":
        assert os.environ["GOMP_SPINCOUNT"] == "0"
    assert os.environ["OPENBLAS_THREAD_TIMEOUT"] == "1"

    monkeypatch.setenv("OMP_WAIT_POLICY", "ACTIVE")
    monkeypatch.setenv("OPENBLAS_THREAD_TIMEOUT", "12")
    cli._set_thread_env_vars(2)
    assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE"
    assert os.environ["OPENBLAS_THREAD_TIMEOUT"] == "12"


def test_thread_preparse_uses_generated_build_wait_policy(monkeypatch):
    from summit import cli

    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    monkeypatch.setattr(cli, "RECOMMENDED_OMP_WAIT_POLICY", "ACTIVE")

    cli._set_thread_env_vars(2)

    assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE"


def test_explicit_openmp_placement_parser_and_scope_validation():
    from summit import cli

    parser = cli.build_parser()
    ordinary = parser.parse_args([])
    assert ordinary.gxe_explicit_openmp_placement is False
    assert ordinary.gxe_explicit_openmp_memory_scope == "selected-cpus"
    socket_scope_without_placement = parser.parse_args(
        ["--gxe-explicit-openmp-memory-scope", "selected-socket"]
    )
    with pytest.raises(ValueError, match="requires --gxe-explicit-openmp-placement"):
        cli._validate_explicit_openmp_placement_request(
            socket_scope_without_placement
        )
    explicit = parser.parse_args(
        [
            "--geno", "genotype",
            "--env", "environment.tsv",
            "--gxe-env-cols", "age,bmi",
            "--gxe-native-backend", "direct",
            "--gxe-parallel-environment-groups", "1",
            "--num-threads", "32",
            "--gxe-explicit-openmp-placement",
            "--gxe-explicit-openmp-memory-scope", "selected-socket",
        ]
    )
    cli._validate_explicit_openmp_placement_request(explicit)
    explicit.gxe_parallel_environment_groups = "2"
    with pytest.raises(ValueError, match="--gxe-parallel-environment-groups=1"):
        cli._validate_explicit_openmp_placement_request(explicit)
    explicit.gxe_parallel_environment_groups = "1"
    explicit.gxe_native_backend = "python"
    with pytest.raises(ValueError, match="valid only for direct"):
        cli._validate_explicit_openmp_placement_request(explicit)
    explicit.gxe_native_backend = "direct"
    explicit.num_threads = None
    with pytest.raises(ValueError, match="explicit positive --num-threads"):
        cli._validate_explicit_openmp_placement_request(explicit)


def test_explicit_single_group_layout_uses_first_verified_physical_cpus(monkeypatch):
    from summit import cli

    monkeypatch.setattr(
        cli,
        "_validated_explicit_outer_cpu_affinity",
        lambda: tuple(range(8)),
    )
    monkeypatch.setattr(
        cli,
        "_socket_local_core_groups",
        lambda *, allowed_cpu_ids=None: [
            {
                "socket": 0,
                "cpus": (0, 2, 4, 6),
                "nodes": (0, 1),
                "node_by_cpu": {0: 0, 2: 0, 4: 1, 6: 1},
            },
            {
                "socket": 1,
                "cpus": (1, 3, 5, 7),
                "nodes": (2, 3),
                "node_by_cpu": {1: 2, 3: 2, 5: 3, 7: 3},
            },
        ],
    )
    args = SimpleNamespace(
        gxe_explicit_openmp_placement=True,
        gxe_parallel_environment_groups="1",
        _gxe_environment_group_worker=False,
        gxe_native_backend="direct",
        num_threads=3,
    )

    layout = cli._parallel_environment_layout(args, ("age", "bmi"))

    assert layout == (
        {
            "columns": ("age", "bmi"),
            "cpus": (0, 2, 4),
            "nodes": (0, 1),
            "threads": 3,
        },
    )
    args.num_threads = 1
    assert cli._parallel_environment_layout(args, ("age", "bmi")) == (
        {
            "columns": ("age", "bmi"),
            "cpus": (0,),
            "nodes": (0,),
            "threads": 1,
        },
    )
    args.gxe_explicit_openmp_memory_scope = "selected-socket"
    monkeypatch.setattr(
        cli, "_verified_full_socket_numa_nodes", lambda socket_id: (0, 1)
    )
    assert cli._parallel_environment_layout(args, ("age", "bmi")) == (
        {
            "columns": ("age", "bmi"),
            "cpus": (0,),
            "nodes": (0, 1),
            "threads": 1,
        },
    )
    monkeypatch.setattr(
        cli, "_verified_full_socket_numa_nodes", lambda socket_id: (1, 2, 3)
    )
    with pytest.raises(RuntimeError, match="complete, unambiguous online socket"):
        cli._parallel_environment_layout(args, ("age", "bmi"))
    args.gxe_explicit_openmp_memory_scope = "selected-cpus"
    args.gxe_explicit_openmp_placement = False
    assert cli._parallel_environment_layout(args, ("age", "bmi")) is None
    args.gxe_explicit_openmp_placement = True
    args._gxe_environment_group_worker = True
    assert cli._parallel_environment_layout(args, ("age", "bmi")) is None
    args._gxe_environment_group_worker = False
    args.num_threads = 9
    with pytest.raises(RuntimeError, match="cannot satisfy the requested thread"):
        cli._parallel_environment_layout(args, ("age", "bmi"))
    args.num_threads = 2
    args.gxe_explicit_openmp_memory_scope = "selected-socket"
    monkeypatch.setattr(
        cli,
        "_socket_local_core_groups",
        lambda *, allowed_cpu_ids=None: [
            {
                "socket": 0,
                "cpus": (0, 2),
                "nodes": (0, 1),
                "node_by_cpu": {0: 0, 2: 0},
            },
        ],
    )
    with pytest.raises(RuntimeError, match="complete allowed physical"):
        cli._parallel_environment_layout(args, ("age", "bmi"))
    args.gxe_explicit_openmp_memory_scope = "forged-scope"
    with pytest.raises(RuntimeError, match="must be selected-cpus or selected-socket"):
        cli._parallel_environment_layout(args, ("age", "bmi"))


def test_selected_socket_scope_expands_beyond_outer_cpu_affinity(monkeypatch):
    from summit import cli

    monkeypatch.setattr(
        cli, "_validated_explicit_outer_cpu_affinity", lambda: (0,)
    )
    monkeypatch.setattr(
        cli,
        "_socket_local_core_groups",
        lambda *, allowed_cpu_ids=None: [
            {
                "socket": 0,
                "cpus": (0,),
                "nodes": (0,),
                "node_by_cpu": {0: 0},
            },
        ],
    )
    observed = []
    monkeypatch.setattr(
        cli,
        "_verified_full_socket_numa_nodes",
        lambda socket_id: observed.append(socket_id) or (0, 1, 2, 3),
    )
    args = SimpleNamespace(
        gxe_explicit_openmp_placement=True,
        gxe_explicit_openmp_memory_scope="selected-socket",
        gxe_parallel_environment_groups="1",
        _gxe_environment_group_worker=False,
        gxe_native_backend="direct",
        num_threads=1,
    )

    layout = cli._parallel_environment_layout(args, ("age", "bmi"))

    assert observed == [0]
    assert layout == (
        {
            "columns": ("age", "bmi"),
            "cpus": (0,),
            "nodes": (0, 1, 2, 3),
            "threads": 1,
        },
    )


def test_explicit_outer_layout_recovers_pre_numerical_mask_but_default_uses_live(
    monkeypatch,
):
    from summit import cli

    captured = tuple(range(32))
    live = {"cpus": {0}}
    monkeypatch.setattr(cli, "_PRE_NUMERICAL_CPU_AFFINITY", captured)
    monkeypatch.setattr(
        cli.os, "sched_getaffinity", lambda _pid: set(live["cpus"])
    )
    for name, value in {
        "OMP_NUM_THREADS": "32",
        "OMP_THREAD_LIMIT": "32",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "SPREAD",
        "OMP_PLACES": cli._canonical_omp_places(captured),
        "OMP_MAX_ACTIVE_LEVELS": "1",
    }.items():
        monkeypatch.setenv(name, value)

    observed = []

    def topology(*, allowed_cpu_ids=None):
        selected = (
            tuple(sorted(cli.os.sched_getaffinity(0)))
            if allowed_cpu_ids is None
            else tuple(allowed_cpu_ids)
        )
        observed.append(selected)
        return [
            {
                "socket": 0,
                "cpus": selected,
                "nodes": (0,),
                "node_by_cpu": {cpu: 0 for cpu in selected},
            }
        ]

    monkeypatch.setattr(cli, "_socket_local_core_groups", topology)
    explicit = SimpleNamespace(
        gxe_explicit_openmp_placement=True,
        gxe_parallel_environment_groups="1",
        _gxe_environment_group_worker=False,
        gxe_native_backend="direct",
        num_threads=32,
    )

    layout = cli._parallel_environment_layout(explicit, ("age", "bmi"))

    assert layout[0]["cpus"] == captured
    assert observed == [captured]

    live["cpus"] = {7}
    ordinary = SimpleNamespace(
        gxe_explicit_openmp_placement=False,
        gxe_parallel_environment_groups="auto",
        _gxe_environment_group_worker=False,
        gxe_native_backend="direct",
        num_threads=1,
    )
    assert (
        cli._parallel_environment_layout(ordinary, ("a", "b", "c", "d"))
        is None
    )
    assert observed == [captured, (7,)]


def test_explicit_outer_affinity_capture_and_subset_evidence_fail_closed(
    monkeypatch,
):
    from summit import cli

    monkeypatch.setattr(cli, "_PRE_NUMERICAL_CPU_AFFINITY", [0, 1])
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _pid: {0})
    with pytest.raises(RuntimeError, match="valid immutable"):
        cli._validated_explicit_outer_cpu_affinity()

    monkeypatch.setattr(cli, "_PRE_NUMERICAL_CPU_AFFINITY", (0, 1))
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _pid: {2})
    with pytest.raises(RuntimeError, match="escapes"):
        cli._validated_explicit_outer_cpu_affinity()

    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _pid: set())
    with pytest.raises(RuntimeError, match="nonempty current"):
        cli._validated_explicit_outer_cpu_affinity()

    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _pid: {0})
    for name, value in {
        "OMP_NUM_THREADS": "2",
        "OMP_THREAD_LIMIT": "2",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "SPREAD",
        "OMP_PLACES": "{0},{1}",
        "OMP_MAX_ACTIVE_LEVELS": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OMP_MAX_ACTIVE_LEVELS", "2")
    with pytest.raises(RuntimeError, match="canonical explicit OpenMP launcher"):
        cli._validated_explicit_outer_cpu_affinity()


def test_full_socket_numa_topology_fails_closed_when_ambiguous_or_unallowed(
    tmp_path,
):
    from summit import cli

    node_root = tmp_path / "node"
    cpu_root = tmp_path / "cpu"
    process_status = tmp_path / "status"
    for node, cpus in ((0, "0"), (1, "1"), (2, "2")):
        path = node_root / f"node{node}"
        path.mkdir(parents=True)
        (path / "cpulist").write_text(cpus + "\n", encoding="utf-8")
    (node_root / "online").write_text("0-2\n", encoding="utf-8")
    for cpu, package in ((0, 0), (1, 0), (2, 1)):
        path = cpu_root / f"cpu{cpu}" / "topology"
        path.mkdir(parents=True)
        (path / "physical_package_id").write_text(
            f"{package}\n", encoding="utf-8"
        )
    process_status.write_text("Mems_allowed_list:\t0-2\n", encoding="utf-8")
    kwargs = {
        "node_root": node_root,
        "cpu_root": cpu_root,
        "process_status": process_status,
    }

    assert cli._verified_full_socket_numa_nodes(0, **kwargs) == (0, 1)

    process_status.write_text("Mems_allowed_list:\t0,2\n", encoding="utf-8")
    assert cli._verified_full_socket_numa_nodes(0, **kwargs) is None

    process_status.write_text("Mems_allowed_list:\t0-2\n", encoding="utf-8")
    (node_root / "node1" / "cpulist").write_text("1-2\n", encoding="utf-8")
    assert cli._verified_full_socket_numa_nodes(0, **kwargs) is None


def test_explicit_single_group_routes_through_fresh_worker_dispatch(monkeypatch):
    from summit import cli

    expected_layout = (
        {
            "columns": ("age", "bmi"),
            "cpus": (0, 1),
            "nodes": (0,),
            "threads": 2,
        },
    )
    monkeypatch.setattr(
        cli, "_parallel_environment_layout", lambda args, columns: expected_layout
    )
    observed = {}
    monkeypatch.setattr(
        cli,
        "_dispatch_parallel_gxe_environment_groups",
        lambda args, columns, layout, log: observed.update(
            columns=columns, layout=layout
        ),
    )
    args = SimpleNamespace(gxe_env_cols="age,bmi")

    cli._dispatch_gxe_multi_reference(args, None, False, {})

    assert observed == {
        "columns": ("age", "bmi"),
        "layout": expected_layout,
    }


def test_multi_environment_parallel_layout_uses_distinct_physical_sockets(monkeypatch):
    from summit import cli

    monkeypatch.setattr(
        cli,
        "_socket_local_core_groups",
        lambda: [
            {
                "socket": 0,
                "cpus": tuple(range(32)),
                "nodes": (0, 1, 2, 3),
                "node_by_cpu": {cpu: cpu // 8 for cpu in range(32)},
            },
            {
                "socket": 1,
                "cpus": tuple(range(32, 64)),
                "nodes": (4, 5, 6, 7),
                "node_by_cpu": {cpu: cpu // 8 for cpu in range(32, 64)},
            },
        ],
    )
    args = SimpleNamespace(
        gxe_parallel_environment_groups="auto",
        _gxe_environment_group_worker=False,
        gxe_native_backend="direct",
        num_threads=64,
    )
    layout = cli._parallel_environment_layout(
        args, ("age", "sex", "bmi", "alcohol", "smoking")
    )
    assert [record["columns"] for record in layout] == [
        ("age", "sex", "bmi"),
        ("alcohol", "smoking"),
    ]
    assert set(layout[0]["cpus"]).isdisjoint(layout[1]["cpus"])
    assert [record["threads"] for record in layout] == [32, 32]


def test_parallel_layout_binds_only_nodes_covered_by_selected_cpus(monkeypatch):
    from summit import cli

    monkeypatch.setattr(
        cli,
        "_socket_local_core_groups",
        lambda: [
            {
                "socket": 0,
                "cpus": tuple(range(32)),
                "nodes": (0, 1),
                "node_by_cpu": {
                    cpu: 0 if cpu < 16 else 1 for cpu in range(32)
                },
            },
            {
                "socket": 1,
                "cpus": tuple(range(32, 64)),
                "nodes": (2, 3),
                "node_by_cpu": {
                    cpu: 2 if cpu < 48 else 3 for cpu in range(32, 64)
                },
            },
        ],
    )
    args = SimpleNamespace(
        gxe_parallel_environment_groups="2",
        _gxe_environment_group_worker=False,
        gxe_native_backend="direct",
        num_threads=32,
    )

    layout = cli._parallel_environment_layout(args, ("a", "b", "c", "d"))

    assert [record["cpus"] for record in layout] == [
        tuple(range(16)),
        tuple(range(32, 48)),
    ]
    assert [record["nodes"] for record in layout] == [(0,), (2,)]


def test_explicit_parallel_layout_fails_closed_without_physical_topology(monkeypatch):
    from summit import cli

    monkeypatch.setattr(cli, "_socket_local_core_groups", lambda: [])
    args = SimpleNamespace(
        gxe_parallel_environment_groups="2",
        _gxe_environment_group_worker=False,
        gxe_native_backend="direct",
        num_threads=32,
    )
    with pytest.raises(RuntimeError, match="Two GxE environment groups require"):
        cli._parallel_environment_layout(args, ("a", "b", "c", "d"))

    args.gxe_parallel_environment_groups = "auto"
    assert cli._parallel_environment_layout(args, ("a", "b", "c", "d")) is None

    monkeypatch.setattr(
        cli,
        "_socket_local_core_groups",
        lambda: [
            {
                "socket": 0,
                "cpus": tuple(range(16)),
                "nodes": (0,),
                "node_by_cpu": {cpu: 0 for cpu in range(16)},
            },
            {
                "socket": 1,
                "cpus": tuple(range(16, 32)),
                "nodes": (1,),
                "node_by_cpu": {cpu: 1 for cpu in range(16, 32)},
            },
        ],
    )
    args.gxe_parallel_environment_groups = "2"
    args.num_threads = 64
    with pytest.raises(RuntimeError, match="cannot satisfy the requested thread"):
        cli._parallel_environment_layout(args, ("a", "b", "c", "d"))

    monkeypatch.setattr(
        cli,
        "_socket_local_core_groups",
        lambda: [
            {
                "socket": 0,
                "cpus": tuple(range(32)),
                "nodes": (0,),
                "node_by_cpu": {cpu: 0 for cpu in range(32)},
            },
            {
                "socket": 1,
                "cpus": tuple(range(32, 40)),
                "nodes": (1,),
                "node_by_cpu": {cpu: 1 for cpu in range(32, 40)},
            },
        ],
    )
    args.num_threads = 32
    asymmetric = cli._parallel_environment_layout(args, ("a", "b", "c", "d"))
    assert [record["threads"] for record in asymmetric] == [24, 8]


def test_group_worker_authentication_and_cpu_contract(monkeypatch):
    from summit import cli

    token = "unit-test-secret"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    args = cli.build_parser().parse_args(
        [
            "--_gxe-environment-group-worker",
            "--_gxe-worker-cpus", "2-3",
            "--_gxe-worker-auth-sha256", digest,
            "--gxe-native-backend", "direct",
            "--gxe-parallel-environment-groups", "1",
            "--gxe-env-cols", "a,b",
            "--_gxe-multi-batch-manifest", "group.json",
            "--num-threads", "2",
        ]
    )
    monkeypatch.setenv(cli._GXE_GROUP_WORKER_TOKEN_ENV, token)
    contract = cli._authenticate_gxe_group_worker(args)
    assert contract.cpu_ids == (2, 3)
    assert contract.threads == 2
    assert args._gxe_worker_auth_sha256 is None
    assert cli._GXE_GROUP_WORKER_TOKEN_ENV not in os.environ

    wrong = cli.build_parser().parse_args(
        [
            "--_gxe-environment-group-worker",
            "--_gxe-worker-cpus", "2-3",
            "--_gxe-worker-auth-sha256", "0" * 64,
            "--gxe-native-backend", "direct",
            "--gxe-parallel-environment-groups", "1",
            "--gxe-env-cols", "a,b",
            "--_gxe-multi-batch-manifest", "group.json",
            "--num-threads", "2",
        ]
    )
    monkeypatch.setenv(cli._GXE_GROUP_WORKER_TOKEN_ENV, token)
    with pytest.raises(RuntimeError, match="authentication failed"):
        cli._authenticate_gxe_group_worker(wrong)


def test_group_worker_auth_hash_is_never_logged(monkeypatch):
    from summit import cli

    digest = "a" * 64
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--_gxe-worker-auth-sha256", digest,
            "--_gxe-worker-cpus", "2-3",
        ]
    )
    observed = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit",
            "--_gxe-worker-auth-sha", digest,
            "--_gxe-worker-cpus", "2-3",
        ],
    )
    cli._log_cli_args(parser, args, SimpleNamespace(_log=observed.append))
    assert all(digest not in line for line in observed)


def test_group_worker_placement_probe_requires_exact_native_contract(monkeypatch):
    from summit import cli

    placement = _cpu_placement([2, 3])
    native = SimpleNamespace(
        configure_openmp_placement=lambda cpus, threads: placement,
        build_info=lambda: _placement_build_info(placement),
    )
    contract = cli._GxeGroupWorkerCpuContract((2, 3), 2)
    canonical_environment = {
        "OMP_NUM_THREADS": "2",
        "OMP_THREAD_LIMIT": "2",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "SPREAD",
        "OMP_PLACES": "{2},{3}",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "BLIS_NUM_THREADS": "2",
    }
    for key, value in canonical_environment.items():
        monkeypatch.setenv(key, value)
    for key in cli._GXE_OMP_AFFINITY_CONFLICTS:
        monkeypatch.delenv(key, raising=False)
    for key in cli._GXE_BLIS_AUTOMATIC_CONFLICTS:
        monkeypatch.delenv(key, raising=False)

    assert cli._configure_gxe_group_worker_placement(contract, native) == placement

    monkeypatch.setenv("BLIS_NT", "8")
    with pytest.raises(RuntimeError, match="noncanonical threading environment"):
        cli._configure_gxe_group_worker_placement(contract, native)
    monkeypatch.delenv("BLIS_NT")

    malformed = _placement_build_info(placement)
    malformed["api_version"] = True
    native.build_info = lambda: malformed
    with pytest.raises(RuntimeError, match="build contract"):
        cli._configure_gxe_group_worker_placement(contract, native)

    pthread_build = _placement_build_info(placement)
    pthread_build["blas_runtime_threading_layer"] = "pthreads"
    native.build_info = lambda: pthread_build
    with pytest.raises(RuntimeError, match="build contract"):
        cli._configure_gxe_group_worker_placement(contract, native)

    malformed_placement = _cpu_placement([2, 3])
    malformed_placement["omp_max_active_levels"] = True
    native.configure_openmp_placement = lambda cpus, threads: malformed_placement
    with pytest.raises(RuntimeError, match="incomplete or noncanonical"):
        cli._configure_gxe_group_worker_placement(contract, native)


def test_authenticated_capacity_is_the_only_caller_affinity_override(monkeypatch):
    from summit.ldscore import gw_ldscore

    placement = _cpu_placement([2, 3])
    authenticated = {
        "num_threads": 2,
        "_gxe_group_worker_authenticated": True,
        "_gxe_worker_cpu_ids": (2, 3),
        "_gxe_cpu_placement": placement,
        "_gxe_cpu_placement_complete": True,
    }
    assert gw_ldscore._validated_thread_capacity(authenticated, 1) == 2
    assert gw_ldscore._validated_thread_capacity({}, 1) == 1
    monkeypatch.setattr(gw_ldscore.os, "sched_getaffinity", lambda pid: {2})
    assert gw_ldscore.apply_env(
        dict(
            authenticated,
            _runtime_preconfigured=True,
            _actual_runtime_threads=2,
        )
    ) == 2
    with pytest.raises(RuntimeError, match="disagree with verified OpenMP capacity"):
        gw_ldscore.apply_env(
            dict(
                authenticated,
                _runtime_preconfigured=True,
                _actual_runtime_threads=1,
            )
        )
    forged = dict(authenticated, _gxe_group_worker_authenticated=False)
    with pytest.raises(RuntimeError, match="authenticated complete"):
        gw_ldscore._validated_thread_capacity(forged, 1)


@pytest.mark.parametrize(
    "layout",
    [
        (
            {
                "columns": ("a", "b", "c", "d"),
                "cpus": (2, 3),
                "nodes": (0, 1),
                "threads": 2,
            },
        ),
        (
            {"columns": ("a", "b"), "cpus": (2, 3), "nodes": (0,), "threads": 2},
            {"columns": ("c", "d"), "cpus": (6, 7), "nodes": (1,), "threads": 2},
        ),
    ],
)
def test_parallel_dispatch_seals_child_openmp_environment(
    tmp_path, monkeypatch, layout
):
    from summit import cli

    launched = []
    expected_python_prefix = [sys.executable, "-S", "-B"]

    class CompletedProcess:
        def __init__(self, command, env):
            launched.append((command, env))

        def poll(self):
            return 0

    monkeypatch.setattr(cli.subprocess, "Popen", CompletedProcess)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        cli, "_python_isolation_prefix", lambda: list(expected_python_prefix)
    )
    combined = {}

    def fake_combine(paths, **kwargs):
        combined["paths"] = tuple(paths)
        combined.update(kwargs)
        return kwargs["batch_manifest"]

    monkeypatch.setattr(
        cli, "combine_multi_environment_reference_batches", fake_combine
    )
    for name in cli._GXE_OMP_AFFINITY_CONFLICTS:
        monkeypatch.setenv(name, "inherited-conflict")
    for name in cli._GXE_BLIS_AUTOMATIC_CONFLICTS:
        monkeypatch.setenv(name, "inherited-conflict")
    argv = [
        "summit", "--gxe-env-cols", "a,b,c,d", "--out", str(tmp_path / "run")
    ]
    if len(layout) == 1:
        argv.extend(
            ["--gxe-explicit-openmp-memory-scope", "selected-socket"]
        )
    monkeypatch.setattr(sys, "argv", argv)
    args = SimpleNamespace(
        out=str(tmp_path / "run"),
        _gxe_multi_batch_manifest=None,
    )
    cli._dispatch_parallel_gxe_environment_groups(
        args, ("a", "b", "c", "d"), layout, SimpleNamespace(_log=lambda *_: None)
    )

    assert len(launched) == len(layout)
    assert combined["require_cpu_placement"] is True
    assert combined["expected_cpu_groups"] == [
        record["cpus"] for record in layout
    ]
    assert combined["expected_numa_groups"] == [
        record["nodes"] for record in layout
    ]
    for (command, environment), record in zip(launched, layout, strict=True):
        assert command[:3] == ["/usr/bin/taskset", "-c", cli._format_integer_ranges(record["cpus"])]
        assert command[3:6] == expected_python_prefix
        assert command[6:8] == ["-m", "summit.cli"]
        digest_index = command.index("--_gxe-worker-auth-sha256") + 1
        cpu_index = command.index("--_gxe-worker-cpus") + 1
        raw_token = environment[cli._GXE_GROUP_WORKER_TOKEN_ENV]
        assert command[digest_index] == hashlib.sha256(
            raw_token.encode("utf-8")
        ).hexdigest()
        assert raw_token not in command
        assert command[cpu_index] == cli._format_integer_ranges(record["cpus"])
        numa_mode_index = command.index("--numa-mode") + 1
        numa_nodes_index = command.index("--numa-nodes") + 1
        assert command[numa_mode_index] == "membind"
        assert command[numa_nodes_index] == cli._format_integer_ranges(
            record["nodes"]
        )
        if len(layout) == 1:
            scope_index = command.index(
                "--gxe-explicit-openmp-memory-scope"
            ) + 1
            assert command[scope_index] == "selected-socket"
        assert environment["OMP_PLACES"] == cli._canonical_omp_places(record["cpus"])
        assert environment["OMP_PROC_BIND"] == "SPREAD"
        assert environment["OMP_NUM_THREADS"] == "2"
        assert environment["OMP_THREAD_LIMIT"] == "2"
        assert environment["OMP_MAX_ACTIVE_LEVELS"] == "1"
        assert all(name not in environment for name in cli._GXE_OMP_AFFINITY_CONFLICTS)
        assert environment["BLIS_NUM_THREADS"] == "2"
        assert all(
            name not in environment
            for name in cli._GXE_BLIS_AUTOMATIC_CONFLICTS
        )


def test_outer_numactl_sentinel_prevents_nested_cli_reexec(monkeypatch):
    from summit.ldscore import gw_ldscore

    monkeypatch.setenv("SUMMIT_NUMACTL_WRAPPED", "1")
    monkeypatch.setattr(gw_ldscore.shutil, "which", lambda name: "/usr/bin/numactl")

    def forbidden_exec(*_):
        raise AssertionError("sealed outer NUMA launch attempted a nested re-exec")

    monkeypatch.setattr(gw_ldscore.os, "execv", forbidden_exec)
    threads = gw_ldscore.apply_env(
        {
            "numa_mode": "interleave",
            "numa_nodes": "all",
            "force_affinity_all": False,
            "num_threads": 1,
            "decode_threads_cap": 1,
        }
    )
    assert isinstance(threads, int) and threads >= 1
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["SUMMIT_NUMACTL_WRAPPED"] == "1"


def test_numactl_reexec_preserves_python_isolation_and_module_invocation(monkeypatch):
    from summit.ldscore import gw_ldscore

    monkeypatch.delenv("SUMMIT_NUMACTL_WRAPPED", raising=False)
    monkeypatch.setattr(gw_ldscore.shutil, "which", lambda name: "/usr/bin/numactl")
    monkeypatch.setattr(
        gw_ldscore, "attest_numa_policy_request", lambda *_args: False
    )
    original = [
        "python",
        "-S",
        "-B",
        "-m",
        "summit.cli",
        "--gxe-parallel-environment-groups",
        "2",
    ]
    monkeypatch.setattr(sys, "orig_argv", original, raising=False)
    observed = {}

    class ReexecCaptured(RuntimeError):
        pass

    def capture_exec(executable, arguments):
        observed["executable"] = executable
        observed["arguments"] = list(arguments)
        raise ReexecCaptured

    monkeypatch.setattr(gw_ldscore.os, "execv", capture_exec)
    with pytest.raises(ReexecCaptured):
        gw_ldscore.apply_env(
            {
                "numa_mode": "interleave",
                "numa_nodes": "0-3",
                "force_affinity_all": False,
                "num_threads": 2,
                "decode_threads_cap": 2,
            }
        )

    assert observed == {
        "executable": "/usr/bin/numactl",
        "arguments": [
            "/usr/bin/numactl",
            "--interleave=0-3",
            sys.executable,
            *original[1:],
        ],
    }
    assert os.environ["SUMMIT_NUMACTL_WRAPPED"] == "1"


def test_internal_worker_python_prefix_preserves_isolation_flags(monkeypatch):
    from summit import cli

    monkeypatch.setattr(
        cli.sys,
        "flags",
        SimpleNamespace(
            isolated=0,
            ignore_environment=1,
            no_user_site=1,
            safe_path=1,
            no_site=1,
            dont_write_bytecode=1,
            optimize=2,
        ),
    )
    assert cli._python_isolation_prefix() == [
        sys.executable,
        "-E",
        "-s",
        "-P",
        "-S",
        "-B",
        "-OO",
    ]


def test_apply_env_uses_early_numa_attestation_before_legacy_paths(monkeypatch):
    from summit.ldscore import gw_ldscore

    requests = []
    monkeypatch.setattr(
        gw_ldscore,
        "attest_numa_policy_request",
        lambda mode, nodes: requests.append((mode, nodes)) or True,
    )
    monkeypatch.setattr(
        gw_ldscore.shutil,
        "which",
        lambda _: pytest.fail("attested policy must bypass numactl re-exec"),
    )
    threads = gw_ldscore.apply_env(
        {
            "numa_mode": "membind",
            "numa_nodes": "0-1",
            "force_affinity_all": False,
            "num_threads": 1,
            "decode_threads_cap": 1,
            "_runtime_threadpool_capped": True,
        }
    )

    assert requests == [("membind", "0-1")]
    assert isinstance(threads, int) and threads >= 1


def test_apply_env_rejects_late_or_environment_only_membind(monkeypatch):
    from summit.ldscore import gw_ldscore

    monkeypatch.setenv("SUMMIT_NUMA_POLICY_APPLIED", "libnuma:membind:0,1")
    monkeypatch.setenv("SUMMIT_NUMA_POLICY_PROVENANCE", "pre_numeric_import")
    monkeypatch.setattr(
        gw_ldscore, "attest_numa_policy_request", lambda mode, nodes: False
    )

    with pytest.raises(RuntimeError, match="without a verified pre-numeric-import"):
        gw_ldscore.apply_env(
            {
                "numa_mode": "membind",
                "numa_nodes": "0-1",
                "force_affinity_all": False,
                "num_threads": 1,
                "decode_threads_cap": 1,
            }
        )


def test_preconfigured_apply_env_still_requires_membind_attestation(monkeypatch):
    from summit.ldscore import gw_ldscore

    requests = []
    monkeypatch.setattr(
        gw_ldscore,
        "attest_numa_policy_request",
        lambda mode, nodes: requests.append((mode, nodes)) or False,
    )

    with pytest.raises(RuntimeError, match="Preconfigured NUMA membind lacks"):
        gw_ldscore.apply_env(
            {
                "_runtime_preconfigured": True,
                "_actual_runtime_threads": 4,
                "numa_mode": "membind",
                "numa_nodes": "0-1",
            }
        )
    assert requests == [("membind", "0-1")]


def test_gxe_cli_creates_new_log_then_refuses_existing_prefix(tmp_path, monkeypatch):
    from summit import cli

    calls = []
    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    prefix = tmp_path / "run"

    def dispatch(*_):
        calls.append("dispatch")
        (tmp_path / "run.gxe.ref.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(cli, "_dispatch_ldscore", dispatch)
    argv = [
        "summit",
        "--geno", str(tmp_path / "geno"),
        "--env", str(tmp_path / "env.tsv"),
        "--out", str(prefix),
        "--suppress",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls == ["dispatch"]
    log_path = tmp_path / "run.gxe.log"
    assert log_path.is_file()
    assert (log_path.stat().st_mode & 0o777) == 0o600
    original = log_path.read_bytes()

    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        cli.main()
    assert calls == ["dispatch"]
    assert log_path.read_bytes() == original

    monkeypatch.setattr(sys, "argv", [*argv, "--gxe-overwrite"])
    cli.main()
    assert calls == ["dispatch", "dispatch"]


def test_failed_gxe_cli_attempt_can_retry_with_existing_log(tmp_path, monkeypatch):
    from summit import cli

    calls = 0
    monkeypatch.setattr(cli, "apply_env", lambda _: None)

    def dispatch(*_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected failure")

    monkeypatch.setattr(cli, "_dispatch_ldscore", dispatch)
    prefix = tmp_path / "retry"
    argv = [
        "summit",
        "--geno", str(tmp_path / "geno"),
        "--env", str(tmp_path / "env.tsv"),
        "--out", str(prefix),
        "--suppress",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="injected failure"):
        cli.main()
    assert Path(str(prefix) + ".gxe.log").is_file()

    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls == 2


def test_common_cohort_multi_environment_cli_dispatches_once(tmp_path, monkeypatch):
    from summit import cli

    observed = []
    runtime = []
    monkeypatch.setattr(
        cli, "apply_env", lambda low_level: runtime.append(low_level)
    )
    monkeypatch.setattr(
        cli,
        "_dispatch_gxe_multi_reference",
        lambda args, *_: observed.append(args.gxe_env_cols),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "wide.tsv",
            "--gxe-env-cols", "age,bmi", "--out", str(tmp_path / "multi"),
            "--suppress",
        ],
    )
    cli.main()
    assert observed == ["age,bmi"]
    assert len(runtime) == 1


def test_explicit_outer_dispatch_skips_all_runtime_configuration(
    tmp_path, monkeypatch
):
    from summit import cli

    def forbidden(*_args, **_kwargs):
        raise AssertionError("outer dispatcher configured a numerical runtime")

    monkeypatch.setattr(cli, "_make_low_level_env", forbidden)
    monkeypatch.setattr(cli, "_apply_runtime_thread_cap", forbidden)
    monkeypatch.setattr(cli, "apply_env", forbidden)
    observed = []
    monkeypatch.setattr(
        cli,
        "_dispatch_gxe_multi_reference",
        lambda args, _log, _verbose, low_level: observed.append(
            (args.gxe_env_cols, low_level)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "wide.tsv",
            "--gxe-env-cols", "age,bmi",
            "--gxe-native-backend", "direct",
            "--gxe-parallel-environment-groups", "1",
            "--gxe-explicit-openmp-placement", "--num-threads", "32",
            "--out", str(tmp_path / "explicit-outer"), "--suppress",
        ],
    )

    cli.main()

    assert observed == [("age,bmi", None)]


def test_explicit_outer_main_preserves_canonical_threads_through_real_layout(
    tmp_path, monkeypatch
):
    from summit import cli

    captured = tuple(range(32))
    monkeypatch.setattr(cli, "_PRE_NUMERICAL_CPU_AFFINITY", captured)
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _pid: {0})
    for name, value in {
        "OMP_NUM_THREADS": "32",
        "OMP_THREAD_LIMIT": "32",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "SPREAD",
        "OMP_PLACES": cli._canonical_omp_places(captured),
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "BLIS_NUM_THREADS": "32",
    }.items():
        monkeypatch.setenv(name, value)

    observed_topology_masks = []

    def topology(*, allowed_cpu_ids=None):
        observed_topology_masks.append(allowed_cpu_ids)
        return [
            {
                "socket": 0,
                "cpus": captured,
                "nodes": (0,),
                "node_by_cpu": {cpu: 0 for cpu in captured},
            }
        ]

    monkeypatch.setattr(cli, "_socket_local_core_groups", topology)
    runtime_calls = []

    def clamp_if_called(*_args, **_kwargs):
        runtime_calls.append(True)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["BLIS_NUM_THREADS"] = "1"
        return 1

    monkeypatch.setattr(cli, "_apply_runtime_thread_cap", clamp_if_called)
    monkeypatch.setattr(cli, "apply_env", clamp_if_called)
    dispatched = []
    monkeypatch.setattr(
        cli,
        "_dispatch_parallel_gxe_environment_groups",
        lambda _args, columns, layout, _log: dispatched.append(
            {
                "columns": columns,
                "layout": layout,
                "omp_threads": os.environ["OMP_NUM_THREADS"],
                "blis_threads": os.environ["BLIS_NUM_THREADS"],
            }
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "wide.tsv",
            "--gxe-env-cols", "age,bmi",
            "--gxe-native-backend", "direct",
            "--gxe-parallel-environment-groups", "1",
            "--gxe-explicit-openmp-placement", "--num-threads", "32",
            "--out", str(tmp_path / "explicit-real-layout"), "--suppress",
        ],
    )

    cli.main()

    assert runtime_calls == []
    assert observed_topology_masks == [captured]
    assert dispatched == [
        {
            "columns": ("age", "bmi"),
            "layout": (
                {
                    "columns": ("age", "bmi"),
                    "cpus": captured,
                    "nodes": (0,),
                    "threads": 32,
                },
            ),
            "omp_threads": "32",
            "blis_threads": "32",
        }
    ]


@pytest.mark.parametrize(
    ("invalid", "error", "message"),
    (
        (("--ld-wind-kb", "100"), SystemExit, None),
        (("--write-ld-mc-var",), ValueError, "do not yet support"),
    ),
)
def test_explicit_outer_dispatch_preserves_ldscore_validation(
    tmp_path, monkeypatch, invalid, error, message
):
    from summit import cli

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid outer request reached runtime or dispatch")

    monkeypatch.setattr(cli, "_make_low_level_env", forbidden)
    monkeypatch.setattr(cli, "_apply_runtime_thread_cap", forbidden)
    monkeypatch.setattr(cli, "apply_env", forbidden)
    monkeypatch.setattr(cli, "_dispatch_gxe_multi_reference", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "wide.tsv",
            "--gxe-env-cols", "age,bmi",
            "--gxe-native-backend", "direct",
            "--gxe-parallel-environment-groups", "1",
            "--gxe-explicit-openmp-placement", "--num-threads", "32",
            *invalid,
            "--out", str(tmp_path / f"invalid-{'-'.join(invalid)}"),
            "--suppress",
        ],
    )

    with pytest.raises(error, match=message):
        cli.main()


def test_authenticated_explicit_worker_keeps_runtime_configuration(
    tmp_path, monkeypatch
):
    from summit import cli

    contract = SimpleNamespace(cpu_ids=(0,), threads=1)
    monkeypatch.setattr(
        cli, "_authenticate_gxe_group_worker", lambda _args: contract
    )
    monkeypatch.setattr(
        cli, "_configure_gxe_group_worker_placement", lambda _contract: {}
    )
    runtime = []
    monkeypatch.setattr(
        cli,
        "_apply_runtime_thread_cap",
        lambda threads, *, log: runtime.append(("cap", threads)) or True,
    )
    monkeypatch.setattr(
        cli,
        "apply_env",
        lambda low_level: runtime.append(("apply", low_level)) or 1,
    )
    dispatched = []
    monkeypatch.setattr(
        cli,
        "_dispatch_gxe_multi_reference",
        lambda _args, _log, _verbose, low_level: dispatched.append(low_level),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "wide.tsv",
            "--gxe-env-cols", "age,bmi",
            "--gxe-native-backend", "direct",
            "--gxe-parallel-environment-groups", "1",
            "--gxe-explicit-openmp-placement", "--num-threads", "1",
            "--_gxe-environment-group-worker",
            "--out", str(tmp_path / "explicit-worker"), "--suppress",
        ],
    )

    cli.main()

    assert runtime[0] == ("cap", 1)
    assert runtime[1][0] == "apply"
    assert len(dispatched) == 1
    assert dispatched[0]["_runtime_preconfigured"] is True


@pytest.mark.parametrize("layout", (None, ()))
def test_explicit_outer_layout_cannot_fall_through_to_estimator(
    monkeypatch, layout
):
    from summit import cli

    monkeypatch.setattr(
        cli, "_parallel_environment_layout", lambda _args, _columns: layout
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("explicit outer path constructed an estimator")

    monkeypatch.setattr(cli, "_make_gxe_generator", forbidden)
    args = SimpleNamespace(
        gxe_env_cols="age,bmi",
        gxe_explicit_openmp_placement=True,
        _gxe_environment_group_worker=True,
        _gxe_group_worker_authenticated=False,
    )

    with pytest.raises(RuntimeError, match="mandatory fresh-worker layout"):
        cli._dispatch_gxe_multi_reference(args, None, False, None)


@pytest.mark.parametrize(
    "layout", ("source-tt-target-current", "source-tt-target-row")
)
def test_noncurrent_fp64_layout_is_rejected_before_output(
    tmp_path, monkeypatch, capsys, layout
):
    from summit import cli

    prefix = tmp_path / "single-environment"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "environment.tsv",
            "--gxe-fp64-layout", layout,
            "--out", str(prefix), "--suppress",
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert not Path(f"{prefix}.gxe.log").exists()


def test_explicit_current_layout_reaches_fused_multi_environment_dispatch(
    tmp_path, monkeypatch
):
    from summit import cli

    observed = []
    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    monkeypatch.setattr(
        cli,
        "_dispatch_gxe_multi_reference",
        lambda args, *_: observed.append(
            (args.gxe_env_cols, args.gxe_fp64_layout)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "wide.tsv",
            "--gxe-env-cols", "age,bmi",
            "--gxe-fp64-layout", "current",
            "--out", str(tmp_path / "multi-current"), "--suppress",
        ],
    )
    cli.main()
    assert observed == [("age,bmi", "current")]


def test_single_environment_direct_reference_uses_unified_descriptor_pipeline(
    monkeypatch,
):
    from summit import cli

    calls = []

    class Estimator:
        def close(self):
            calls.append(("closed",))

    estimator = Estimator()

    def make_generator(*_args, **kwargs):
        calls.append(("make", kwargs.get("native_backend")))
        return estimator

    def generate(estimators, **kwargs):
        calls.append(("generate", tuple(estimators), kwargs))
        return "unified.gxe.multi.json"

    monkeypatch.setattr(cli, "_make_gxe_generator", make_generator)
    monkeypatch.setattr(cli, "generate_multi_environment_references", generate)
    args = SimpleNamespace(
        env="environment.tsv",
        write_ld_mc_var=False,
        skip_ld_mc=False,
        ld_wind_kb=None,
        gxe_env_cols=None,
        gxe_native_backend="direct",
        gxe_pheno=None,
        _gxe_feature_cache=None,
        _gxe_reference_shard=False,
        _gxe_multi_batch_manifest=None,
        gxe_fp64_layout="current",
        out="reference",
    )
    log = SimpleNamespace(_log=lambda _message: None)

    cli._dispatch_ldscore(args, log, False, None)

    assert calls[0] == ("make", "python")
    assert calls[1][0] == "generate"
    assert calls[1][1] == (estimator,)
    assert calls[1][2] == {
        "batch_manifest": "reference.gxe.multi.json",
        "requested_backend": "direct",
        "full_precision_layout": "current",
    }
    assert calls[2] == ("closed",)


def test_population_reference_cli_dispatches_one_trait_to_scalar_scorer(
    tmp_path, monkeypatch
):
    from summit import cli
    from summit.logger import Logger

    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            gwas=Path("height.gxe.gwas.tsv.gz"),
            gwis=Path("height.gxe.gwis.tsv.gz"),
            moments=Path("height.gxe.moments.json"),
        )

    monkeypatch.setattr(cli, "score_phenotype_from_reference", capture)

    def forbidden_wide(**_):
        raise AssertionError("population-reference mode dispatched the wide scorer")

    monkeypatch.setattr(cli, "score_phenotypes_from_reference", forbidden_wide)
    args = cli.build_parser().parse_args(
        [
            "--geno", "study",
            "--env", "age.tsv",
            "--covar", "covariates.tsv",
            "--gxe-pheno", "phenotype.tsv",
            "--gxe-pheno-col", "height",
            "--gxe-score-reference", "reference.gxe.ref.json",
            "--gxe-population-reference",
            "--out", str(tmp_path / "height"),
        ]
    )
    cli._dispatch_gxe_score(args, Logger(suppress=True))

    assert observed["pheno_col"] == "height"
    assert observed["population_transfer"] is True
    assert observed["reference_manifest"] == "reference.gxe.ref.json"


@pytest.mark.parametrize(
    ("extra", "dispatch_name"),
    [
        (
            [
                "--geno", "geno", "--env", "env", "--gxe-pheno", "traits.tsv",
                "--gxe-score-reference", "reference.json",
            ],
            "_dispatch_gxe_score",
        ),
        (["--gxe-fit-batch", "batch.json"], "_dispatch_gxe_fit_batch"),
    ],
)
def test_gxe_reusable_workflow_modes_dispatch_once(
    tmp_path, monkeypatch, extra, dispatch_name
):
    from summit import cli

    calls = []
    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    for name in (
        "_dispatch_gxe_cache",
        "_dispatch_gxe_score",
        "_dispatch_gxe_merge",
        "_dispatch_gxe_fit_batch",
    ):
        monkeypatch.setattr(
            cli,
            name,
            lambda *args, _name=name: calls.append(_name),
        )
    monkeypatch.setattr(
        sys,
        "argv",
        ["summit", *extra, "--out", str(tmp_path / dispatch_name), "--suppress"],
    )
    cli.main()
    assert calls == [dispatch_name]


@pytest.mark.parametrize(
    "retired",
    [
        ["--geno", "geno", "--env", "env", "--_gxe-build-cache"],
        [
            "--geno", "geno", "--env", "env", "--_gxe-reference-shard",
            "--_gxe-feature-cache", "features.npz",
        ],
        [
            "--_gxe-merge-shards", "shard-0.json", "shard-1.json",
            "--_gxe-feature-cache", "features.npz",
        ],
    ],
)
def test_disk_backed_gxe_construction_flags_are_not_parseable(
    tmp_path, monkeypatch, retired
):
    from summit import cli

    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["summit", *retired, "--out", str(tmp_path / "retired"), "--suppress"],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_private_gxe_reference_shard_reserves_identity_sidecar(tmp_path, monkeypatch):
    from summit import cli

    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    called = False

    def capture(*_):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "_dispatch_ldscore", capture)
    prefix = tmp_path / "reserved"
    Path(str(prefix) + ".gxe.shard.identity.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summit", "--geno", "geno", "--env", "env",
            "--_gxe-reference-shard", "--_gxe-feature-cache", "features.npz",
            "--nvecs", "10", "--out", str(prefix), "--suppress",
        ],
    )
    with pytest.raises(SystemExit):
        cli.main()
    assert not called


def test_gxe_reusable_mode_validation_rejects_ambiguous_or_unsafe_calls(tmp_path, monkeypatch):
    from summit import cli

    monkeypatch.setattr(cli, "apply_env", lambda _: None)
    cases = [
        [
            "--geno", "geno", "--env", "env", "--_gxe-build-cache",
            "--gxe-score-reference", "reference.json", "--gxe-pheno", "traits.tsv",
        ],
        ["--geno", "geno", "--env", "env", "--_gxe-reference-shard"],
        ["--_gxe-merge-shards", "shard.json"],
        [
            "--geno", "geno", "--env", "env", "--gxe-pheno", "traits.tsv",
            "--gxe-score-reference", "reference.json", "--gxe-overwrite",
        ],
        ["--gxe-fit", "reference.json", "--gxe-fit-batch", "batch.json"],
        ["--gxe-fit-batch", "batch.json", "--gxe-moments", "moments.json"],
        ["--gxe-fit-batch", "batch.json", "--_gxe-feature-cache", "cache.npz"],
        ["--gxe-fit-batch", "batch.json", "--covar", "covariates.tsv"],
        ["--gxe-fit-batch", "batch.json", "--annot", "annotations.tsv"],
        ["--gxe-fit-batch", "batch.json", "--gxe-kernel-mode", "genie"],
        ["--gxe-fit-batch=batch.json", "--gxe-kernel-mode=standardized"],
        ["--gxe-fit-batch", "batch.json", "--_gxe-feature-c", "cache.npz"],
        ["--gxe-fit-batch", "batch.json", "--gxe-kernel-m", "genie"],
        ["--gxe-fit-batch", "batch.json", "--ann", "annotations.tsv"],
        ["--gxe-fit-batch", "batch.json", "--gxe-overwrite"],
    ]
    for index, extra in enumerate(cases):
        monkeypatch.setattr(
            sys,
            "argv",
            ["summit", *extra, "--out", str(tmp_path / f"bad-{index}"), "--suppress"],
        )
        with pytest.raises(SystemExit):
            cli.main()


@pytest.mark.parametrize(
    "removed_option",
    (
        "--gxe-build-cache",
        "--gxe-feature-cache",
        "--gxe-jackknife-scratch-gib",
        "--gxe-reference-shard",
        "--gxe-probe-offset",
        "--gxe-merge-shards",
    ),
)
def test_gxe_cache_and_shard_controls_are_not_public_cli_options(removed_option):
    from summit import cli

    parser = cli.build_parser()
    help_text = parser.format_help()
    assert removed_option not in help_text
    assert "--_gxe-" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args([removed_option])
