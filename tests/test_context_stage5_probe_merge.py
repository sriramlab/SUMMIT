from __future__ import annotations

import multiprocessing
import os
import socket
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

import summit.context.probe_merge_v1 as probe_merge_v1
from summit.context.fit_v1 import (
    CONTEXTUAL_FIT_V1_SUFFIX,
    load_contextual_fit_v1,
    write_contextual_fit_v1,
)
from summit.context.probe_merge_v1 import (
    CONTEXTUAL_VARIANT_PROBE_SCIENCE_IDENTITY_KEYS_V1,
    NUMPY_ORACLE_TRANSIENT_FINALIZER_V1,
    ContextualVariantProbeMergeIdentityV1,
    ContextualVariantProbePartialV1,
    adapt_native_contextual_variant_probe_partial_v1,
    contextual_variant_probe_plan_sha256_v1,
    merge_contextual_variant_probe_partials_v1,
)
from summit.context.reference_v1 import (
    CONTEXTUAL_REFERENCE_V1_SUFFIX,
    load_contextual_reference_v1,
    write_contextual_reference_v1,
)
from summit.context.spec import (
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_sha256,
)
from summit.context.trait_v1 import (
    CONTEXTUAL_TRAIT_V1_SUFFIX,
    load_contextual_trait_v1,
    write_contextual_trait_v1,
)
from test_context_stage3_reference_v1 import _native_build_provenance


ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _readonly(value: Any, dtype: str) -> np.ndarray:
    source = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(source.tobytes(order="C"), dtype=dtype).reshape(source.shape)
    result.setflags(write=False)
    return result


def _socket_id(cpu: int) -> int | None:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id")
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _worker_cpu_targets(count: int) -> tuple[int | None, ...]:
    if not hasattr(os, "sched_getaffinity"):
        return (None,) * count
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        return (None,) * count
    by_socket: dict[int, int] = {}
    for cpu in allowed:
        socket_id = _socket_id(cpu)
        if socket_id is not None:
            by_socket.setdefault(socket_id, cpu)
    preferred = [by_socket[key] for key in sorted(by_socket)]
    for cpu in allowed:
        if cpu not in preferred:
            preferred.append(cpu)
    while len(preferred) < count:
        preferred.append(preferred[-1])
    return tuple(preferred[:count])


def _partial_worker(
    connection: Any,
    g_values: np.ndarray,
    begin: int,
    end: int,
    process_slot: int,
    target_cpu: int | None,
) -> None:
    try:
        if target_cpu is not None:
            os.sched_setaffinity(0, {target_cpu})
        affinity = (
            tuple(sorted(os.sched_getaffinity(0)))
            if hasattr(os, "sched_getaffinity")
            else ()
        )
        if not affinity:
            raise RuntimeError("spawned merge worker has no CPU-affinity evidence")
        local = np.asarray(g_values[begin:end], dtype=np.float64)
        probe_sums = np.sum(local, axis=0, dtype=np.float64)
        within = np.einsum("bci,bdi->cd", local, local, optimize=False)
        socket_ids = sorted(
            {value for cpu in affinity if (value := _socket_id(cpu)) is not None}
        )
        connection.send(
            {
                "status": "ok",
                "probe_sums": probe_sums,
                "within_probe_cross": within,
                "process_id": os.getpid(),
                "process_slot": process_slot,
                "cpu_affinity": affinity,
                "socket_ids": tuple(socket_ids),
                "host_name": socket.gethostname(),
            }
        )
    except BaseException:
        connection.send({"status": "error", "traceback": traceback.format_exc()})
    finally:
        connection.close()


def _identity(
    ordered_probe_sha256s: Sequence[str],
    *,
    policy: str,
    science: str = "common",
) -> ContextualVariantProbeMergeIdentityV1:
    return ContextualVariantProbeMergeIdentityV1(
        global_probe_plan_sha256=contextual_variant_probe_plan_sha256_v1(
            ordered_probe_sha256s
        ),
        science_identity={
            name: _sha(f"science-{science}-{name}")
            for name in CONTEXTUAL_VARIANT_PROBE_SCIENCE_IDENTITY_KEYS_V1
        },
        source_tree_sha256=_sha("source-tree"),
        native_build_provenance_sha256=_sha("native-build-provenance"),
        build_id="stage5-test-build",
        native_backend="contextual_reference_streamed_v1",
        native_execution_backend="descriptor_streamed_protected_v1",
        variant_probe_policy=policy,
        native_api_version=1,
        native_backend_version=1,
    )


def _native_snapshot_fixture(
    g_values: np.ndarray,
) -> tuple[dict[str, Any], ContextualVariantProbeMergeIdentityV1, tuple[str, ...],]:
    local = np.asarray(g_values, dtype=np.float64)
    probe_sums = _readonly(np.sum(local, axis=0, dtype=np.float64).T, "<f8")
    within = _readonly(np.einsum("bci,bdi->cd", local, local, optimize=False), "<f8")
    leaves = tuple(_sha(f"native-probe-{index}") for index in range(len(local)))
    annotation_names = ("annotation-0", "annotation-1")
    group_names = ("group-0", "group-1")
    pairs = ContextPairIndex(1)
    components = ContextComponentIndex(annotation_names, pairs)
    pair_q = _readonly([entry.q for entry in pairs.entries], "<i8")
    pair_r = _readonly([entry.r for entry in pairs.entries], "<i8")
    pair_eta = _readonly([entry.kernel_factor for entry in pairs.entries], "<i8")
    component_annotation = _readonly(
        [entry.annotation_index for entry in components.entries], "<i8"
    )
    component_pair = _readonly(
        [entry.pair_index for entry in components.entries], "<i8"
    )
    direct = {
        name: _sha(f"native-{name}")
        for name in (
            "annotation_map_sha256",
            "evaluated_phi_sha256",
            "fixed_basis_sha256",
            "group_map_sha256",
            "missingness_sha256",
            "retained_sample_map_sha256",
            "retained_variant_order_sha256",
            "scale_plan_sha256",
            "variant_order_allele_sha256",
        )
    }
    science_identity = {
        **direct,
        "annotation_names_sha256": canonical_sha256(
            {"axis": "annotation", "names": list(annotation_names)}
        ),
        "group_names_sha256": canonical_sha256(
            {"axis": "deletion_group", "names": list(group_names)}
        ),
        "pair_map_sha256": pairs.digest,
        "component_map_sha256": components.digest,
    }
    source_tree = _sha("native-source-tree")
    build_id = "5" * 40
    build_provenance = _native_build_provenance(build_id, source_tree)
    plan = contextual_variant_probe_plan_sha256_v1(leaves)
    identity = ContextualVariantProbeMergeIdentityV1(
        global_probe_plan_sha256=plan,
        science_identity=science_identity,
        source_tree_sha256=source_tree,
        native_build_provenance_sha256=canonical_sha256(build_provenance),
        build_id=build_id,
        native_backend="plink_bed_descriptor_stream_stage2_v1",
        native_execution_backend=("deterministic_tiled_fp64_with_scalar_witness_v1"),
        variant_probe_policy="explicit_variant_rademacher_v1",
        native_api_version=1,
        native_backend_version=2,
    )
    native_variant_probe_identity = _sha("native-variant-probe-matrix")
    native_identity = {
        "schema": "contextual_native_variant_probe_full_range_identity_v1",
        "source_commit": build_id,
        "source_tree_sha256": source_tree,
        "build_id": build_id,
        "build_provenance": build_provenance,
        "native_api_version": 1,
        "native_backend_version": 2,
        "native_backend": "plink_bed_descriptor_stream_stage2_v1",
        "native_execution_backend": ("deterministic_tiled_fp64_with_scalar_witness_v1"),
        "numeric_policy": "fp64_v1",
        "feature_mode": "P_diag_phi_G_v1",
        "execution_plan_sha256": _sha("native-execution-plan"),
        "sealed_plan_sha256": _sha("native-sealed-plan"),
        "variant_probe_policy": "explicit_variant_rademacher_v1",
        "native_variant_probe_identity_sha256": native_variant_probe_identity,
        "global_probe_begin": 0,
        "global_probe_end": len(leaves),
        "global_probe_count": len(leaves),
        "ordered_probe_sha256s": list(leaves),
        "global_probe_plan_sha256": plan,
        "sample_count": probe_sums.shape[0],
        "component_count": probe_sums.shape[1],
        "probe_sums_layout": "sample_component_c_v1",
        "probe_cross_layout": "component_component_c_v1",
        "full_executor_range_only": True,
        "partitioned_native_finalizer_claimed": False,
        "science_identity_inputs": {
            **direct,
            "annotation_names": list(annotation_names),
            "group_names": list(group_names),
            "pair_q": pair_q,
            "pair_r": pair_r,
            "pair_eta": pair_eta,
            "component_annotation": component_annotation,
            "component_pair": component_pair,
        },
    }
    snapshot = {
        "same_probe_sums": probe_sums,
        "same_probe_cross": within,
        "same_probe_sums_layout": "sample_component_c_v1",
        "same_probe_cross_layout": "component_component_c_v1",
        "same_probe_sums_sha256": array_sha256(probe_sums),
        "same_probe_cross_sha256": array_sha256(within),
        "same_probe_sample_count": probe_sums.shape[0],
        "same_probe_component_count": probe_sums.shape[1],
        "same_probe_variant_probe_count": len(leaves),
        "same_probe_variant_probe_identity_sha256": (native_variant_probe_identity),
        "same_probe_retained_sample_map_sha256": direct["retained_sample_map_sha256"],
        "same_probe_component_annotation_sha256": array_sha256(component_annotation),
        "same_probe_component_pair_sha256": array_sha256(component_pair),
        "same_probe_native_identity": native_identity,
    }
    return snapshot, identity, leaves


def _partial(
    g_values: np.ndarray,
    begin: int,
    end: int,
    ordered_probe_sha256s: Sequence[str],
    identity: ContextualVariantProbeMergeIdentityV1,
    *,
    slot: int,
    lifecycle: str = "partial_complete",
) -> ContextualVariantProbePartialV1:
    local = np.asarray(g_values[begin:end], dtype=np.float64)
    return ContextualVariantProbePartialV1(
        probe_sums=np.sum(local, axis=0, dtype=np.float64),
        within_probe_cross=np.einsum("bci,bdi->cd", local, local, optimize=False),
        global_probe_begin=begin,
        global_probe_end=end,
        global_probe_count=g_values.shape[0],
        ordered_probe_sha256s=tuple(ordered_probe_sha256s[begin:end]),
        identity=identity,
        process_id=os.getpid(),
        process_slot=slot,
        cpu_affinity=tuple(sorted(os.sched_getaffinity(0))),
        socket_ids=tuple(
            sorted(
                {
                    value
                    for cpu in os.sched_getaffinity(0)
                    if (value := _socket_id(cpu)) is not None
                }
            )
        ),
        host_name=socket.gethostname(),
        process_start_method="in_process_test",
        lifecycle=lifecycle,
    )


def _spawn_partials(
    g_values: np.ndarray,
    ranges: Sequence[tuple[int, int]],
    ordered_probe_sha256s: Sequence[str],
    identity: ContextualVariantProbeMergeIdentityV1,
) -> tuple[ContextualVariantProbePartialV1, ...]:
    context = multiprocessing.get_context("spawn")
    targets = _worker_cpu_targets(len(ranges))
    processes: list[multiprocessing.Process] = []
    receivers: list[Any] = []
    for slot, ((begin, end), target_cpu) in enumerate(zip(ranges, targets)):
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_partial_worker,
            args=(sender, g_values, begin, end, slot, target_cpu),
        )
        process.start()
        sender.close()
        receivers.append(receiver)
        processes.append(process)

    payloads: list[dict[str, Any]] = []
    try:
        for receiver in receivers:
            if not receiver.poll(30.0):
                raise AssertionError("spawned merge worker timed out")
            payload = receiver.recv()
            if payload.get("status") != "ok":
                raise AssertionError(payload.get("traceback", str(payload)))
            payloads.append(payload)
    finally:
        for receiver in receivers:
            receiver.close()
        for process in processes:
            process.join(timeout=30.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
    assert all(process.exitcode == 0 for process in processes)

    partials = []
    for (begin, end), payload in zip(ranges, payloads):
        partials.append(
            ContextualVariantProbePartialV1(
                probe_sums=payload["probe_sums"],
                within_probe_cross=payload["within_probe_cross"],
                global_probe_begin=begin,
                global_probe_end=end,
                global_probe_count=g_values.shape[0],
                ordered_probe_sha256s=tuple(ordered_probe_sha256s[begin:end]),
                identity=identity,
                process_id=payload["process_id"],
                process_slot=payload["process_slot"],
                cpu_affinity=payload["cpu_affinity"],
                socket_ids=payload["socket_ids"],
                host_name=payload["host_name"],
                process_start_method="spawn",
            )
        )
    return tuple(partials)


def _oracle(g_values: np.ndarray) -> np.ndarray:
    probe_sums = np.sum(g_values, axis=0, dtype=np.float64)
    within = np.einsum("bci,bdi->cd", g_values, g_values, optimize=False)
    count = g_values.shape[0]
    result = (probe_sums @ probe_sums.T - within) / (count * (count - 1))
    return 0.5 * (result + result.T)


def test_spawn_bd2_one_probe_per_process_recovers_global_signed_ustat() -> None:
    first = np.asarray(
        [[1.0, -2.0, 0.5, 3.0], [-1.5, 0.25, 2.0, -0.75]],
        dtype=np.float64,
    )
    g_values = np.stack((first, -first), axis=0)
    explicit_probes = np.asarray(
        [[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=np.float64
    )
    leaves = tuple(array_sha256(explicit_probes[:, probe]) for probe in range(2))
    identity = _identity(leaves, policy="explicit_rademacher_v1")
    partials = _spawn_partials(g_values, ((0, 1), (1, 2)), leaves, identity)

    result = merge_contextual_variant_probe_partials_v1(partials)
    np.testing.assert_allclose(
        result.same_person, _oracle(g_values), atol=0.0, rtol=0.0
    )
    assert np.all(np.diag(result.same_person) < 0.0)
    assert result.finalizer == NUMPY_ORACLE_TRANSIENT_FINALIZER_V1
    assert result.production_protected_finalizer is False
    assert result.contains_sample_axis is False
    assert result.durable_artifact is False
    assert not result.same_person.flags.writeable
    assert [item["global_probe_begin"] for item in result.provenance["ranges"]] == [
        0,
        1,
    ]
    assert len({partial.process_id for partial in partials}) == 2
    targets = _worker_cpu_targets(2)
    if targets[0] is not None and targets[0] != targets[1]:
        assert partials[0].cpu_affinity == (targets[0],)
        assert partials[1].cpu_affinity == (targets[1],)

    with pytest.raises(ValueError):
        partials[0].probe_sums.setflags(write=True)
    with pytest.raises(TypeError, match="immutable"):
        result.provenance["ranges"][0]["process_slot"] = 99


def test_spawn_bd5_reverse_arrival_philox_commitments_is_deterministic() -> None:
    generator = np.random.default_rng(50127)
    g_values = generator.normal(size=(5, 3, 7))
    leaves = tuple(
        canonical_sha256(
            {
                "kind": "numpy_philox_per_probe_key_v1",
                "global_probe_index": probe,
                "key": [81273 + probe, 99181 - probe],
            }
        )
        for probe in range(5)
    )
    identity = _identity(leaves, policy="numpy_philox_per_probe_key_v1")
    partials = _spawn_partials(g_values, ((0, 2), (2, 5)), leaves, identity)

    forward = merge_contextual_variant_probe_partials_v1(partials)
    reverse = merge_contextual_variant_probe_partials_v1(tuple(reversed(partials)))
    np.testing.assert_array_equal(forward.same_person, reverse.same_person)
    np.testing.assert_allclose(
        forward.same_person, _oracle(g_values), atol=2e-15, rtol=2e-15
    )
    assert forward.provenance == reverse.provenance
    assert [item["global_probe_begin"] for item in forward.provenance["ranges"]] == [
        0,
        2,
    ]


def test_native_snapshot_adapter_binds_full_range_science_and_build_identity() -> None:
    generator = np.random.default_rng(761)
    g_values = generator.normal(size=(3, 2, 5))
    snapshot, identity, leaves = _native_snapshot_fixture(g_values)
    expected = _partial(g_values, 0, 3, leaves, identity, slot=0)
    observed = adapt_native_contextual_variant_probe_partial_v1(
        snapshot,
        identity=identity,
        global_probe_begin=0,
        global_probe_end=3,
        global_probe_count=3,
        ordered_probe_sha256s=leaves,
        process_slot=0,
        process_start_method="in_process_test",
    )
    np.testing.assert_array_equal(observed.probe_sums, expected.probe_sums)
    np.testing.assert_array_equal(
        observed.within_probe_cross, expected.within_probe_cross
    )
    assert observed.partial_commitment_sha256

    with pytest.raises(ValueError, match="full-range commitment"):
        adapt_native_contextual_variant_probe_partial_v1(
            snapshot,
            identity=identity,
            global_probe_begin=0,
            global_probe_end=1,
            global_probe_count=3,
            ordered_probe_sha256s=leaves[:1],
            process_slot=0,
        )

    graft_identity = ContextualVariantProbeMergeIdentityV1(
        global_probe_plan_sha256=identity.global_probe_plan_sha256,
        science_identity={
            **dict(identity.science_identity),
            "scale_plan_sha256": _sha("grafted-scale"),
        },
        source_tree_sha256=identity.source_tree_sha256,
        native_build_provenance_sha256=(identity.native_build_provenance_sha256),
        build_id=identity.build_id,
        native_backend=identity.native_backend,
        native_execution_backend=identity.native_execution_backend,
        variant_probe_policy=identity.variant_probe_policy,
        native_api_version=identity.native_api_version,
        native_backend_version=identity.native_backend_version,
    )
    with pytest.raises(ValueError, match="science identity graft"):
        adapt_native_contextual_variant_probe_partial_v1(
            snapshot,
            identity=graft_identity,
            global_probe_begin=0,
            global_probe_end=3,
            global_probe_count=3,
            ordered_probe_sha256s=leaves,
            process_slot=0,
        )

    bad_build = dict(snapshot)
    bad_build_identity = dict(snapshot["same_probe_native_identity"])
    bad_build_provenance = dict(bad_build_identity["build_provenance"])
    bad_build_provenance["unexpected"] = "hidden"
    bad_build_identity["build_provenance"] = bad_build_provenance
    bad_build["same_probe_native_identity"] = bad_build_identity
    with pytest.raises(ValueError, match="build provenance schema"):
        adapt_native_contextual_variant_probe_partial_v1(
            bad_build,
            identity=identity,
            global_probe_begin=0,
            global_probe_end=3,
            global_probe_count=3,
            ordered_probe_sha256s=leaves,
            process_slot=0,
        )

    bad_digest = dict(snapshot)
    bad_digest["same_probe_sums_sha256"] = _sha("wrong-digest")
    with pytest.raises(ValueError, match="raw evidence"):
        adapt_native_contextual_variant_probe_partial_v1(
            bad_digest,
            identity=identity,
            global_probe_begin=0,
            global_probe_end=3,
            global_probe_count=3,
            ordered_probe_sha256s=leaves,
            process_slot=0,
        )

    with pytest.raises(ValueError, match="layout identity"):
        adapt_native_contextual_variant_probe_partial_v1(
            snapshot,
            identity=identity,
            global_probe_begin=0,
            global_probe_end=3,
            global_probe_count=3,
            ordered_probe_sha256s=leaves,
            process_slot=0,
            probe_sums_layout="sample_component_f_v1",
        )

    # The only admitted native range is the one authenticated by the executor;
    # two relabeled copies cannot masquerade as disjoint process partials.
    with pytest.raises(ValueError, match="full-range commitment"):
        adapt_native_contextual_variant_probe_partial_v1(
            snapshot,
            identity=identity,
            global_probe_begin=1,
            global_probe_end=3,
            global_probe_count=3,
            ordered_probe_sha256s=leaves[1:],
            process_slot=1,
        )

    observed_again = adapt_native_contextual_variant_probe_partial_v1(
        snapshot,
        identity=identity,
        global_probe_begin=0,
        global_probe_end=3,
        global_probe_count=3,
        ordered_probe_sha256s=leaves,
        process_slot=1,
        process_start_method="in_process_test",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        merge_contextual_variant_probe_partials_v1((observed, observed_again))


def test_merge_rejects_interval_identity_digest_and_lifecycle_tampering() -> None:
    generator = np.random.default_rng(9182)
    g_values = generator.normal(size=(5, 2, 4))
    leaves = tuple(_sha(f"range-{probe}") for probe in range(5))
    identity = _identity(leaves, policy="explicit_rademacher_v1")

    gap = (
        _partial(g_values, 0, 1, leaves, identity, slot=0),
        _partial(g_values, 2, 5, leaves, identity, slot=1),
    )
    with pytest.raises(ValueError, match="Gap"):
        merge_contextual_variant_probe_partials_v1(gap)

    overlap = (
        _partial(g_values, 0, 3, leaves, identity, slot=0),
        _partial(g_values, 2, 5, leaves, identity, slot=1),
    )
    with pytest.raises(ValueError, match="Overlapping"):
        merge_contextual_variant_probe_partials_v1(overlap)

    duplicate = _partial(g_values, 0, 5, leaves, identity, slot=0)
    with pytest.raises(ValueError, match="Duplicate"):
        merge_contextual_variant_probe_partials_v1((duplicate, duplicate))

    alternate = _identity(leaves, policy="explicit_rademacher_v1", science="alternate")
    mixed = (
        _partial(g_values, 0, 2, leaves, identity, slot=0),
        _partial(g_values, 2, 5, leaves, alternate, slot=1),
    )
    with pytest.raises(ValueError, match="Mixed identity"):
        merge_contextual_variant_probe_partials_v1(mixed)

    tampered = _partial(g_values, 0, 5, leaves, identity, slot=0)
    object.__setattr__(tampered, "range_commitment_sha256", _sha("tampered"))
    with pytest.raises(ValueError, match="range commitment"):
        merge_contextual_variant_probe_partials_v1((tampered,))

    digest_tampered = _partial(g_values, 0, 5, leaves, identity, slot=0)
    object.__setattr__(digest_tampered, "probe_sums_sha256", _sha("tampered-array"))
    with pytest.raises(ValueError, match="probe_sums_sha256"):
        merge_contextual_variant_probe_partials_v1((digest_tampered,))

    asymmetric = _partial(g_values, 0, 5, leaves, identity, slot=0)
    asymmetric_cross = np.array(asymmetric.within_probe_cross, copy=True)
    asymmetric_cross[0, 1] += 1.0
    immutable_asymmetric_cross = np.frombuffer(
        np.ascontiguousarray(asymmetric_cross, dtype="<f8").tobytes(order="C"),
        dtype="<f8",
    ).reshape(asymmetric_cross.shape)
    immutable_asymmetric_cross.setflags(write=False)
    object.__setattr__(
        asymmetric,
        "within_probe_cross",
        immutable_asymmetric_cross,
    )
    object.__setattr__(
        asymmetric,
        "within_probe_cross_sha256",
        array_sha256(immutable_asymmetric_cross),
    )
    with pytest.raises(ValueError, match="exactly symmetric"):
        merge_contextual_variant_probe_partials_v1((asymmetric,))

    process_tampered = _partial(g_values, 0, 5, leaves, identity, slot=0)
    object.__setattr__(process_tampered, "process_slot", 1)
    with pytest.raises(ValueError, match="transport commitment"):
        merge_contextual_variant_probe_partials_v1((process_tampered,))

    finalized = _partial(
        g_values,
        0,
        5,
        leaves,
        identity,
        slot=0,
        lifecycle="locally_finalized",
    )
    with pytest.raises(ValueError, match="Already-finalized"):
        merge_contextual_variant_probe_partials_v1((finalized,))


def test_partial_rejects_nonfinite_and_global_probe_plan_mismatch() -> None:
    g_values = np.ones((2, 1, 3), dtype=np.float64)
    leaves = (_sha("first"), _sha("second"))
    identity = _identity(leaves, policy="explicit_rademacher_v1")
    assert tuple(identity.science_identity) == (
        CONTEXTUAL_VARIANT_PROBE_SCIENCE_IDENTITY_KEYS_V1
    )
    assert identity.science_identity_sha256 == canonical_sha256(
        dict(identity.science_identity)
    )
    with pytest.raises(ValueError, match="finite"):
        ContextualVariantProbePartialV1(
            probe_sums=np.asarray([[np.nan, 0.0, 1.0]]),
            within_probe_cross=np.ones((1, 1)),
            global_probe_begin=0,
            global_probe_end=1,
            global_probe_count=2,
            ordered_probe_sha256s=(leaves[0],),
            identity=identity,
            process_id=os.getpid(),
            process_slot=0,
            cpu_affinity=tuple(sorted(os.sched_getaffinity(0))),
            host_name=socket.gethostname(),
        )

    wrong_identity = ContextualVariantProbeMergeIdentityV1(
        global_probe_plan_sha256=_sha("wrong-global-plan"),
        science_identity=identity.science_identity,
        source_tree_sha256=identity.source_tree_sha256,
        native_build_provenance_sha256=(identity.native_build_provenance_sha256),
        build_id=identity.build_id,
        native_backend=identity.native_backend,
        native_execution_backend=identity.native_execution_backend,
        variant_probe_policy=identity.variant_probe_policy,
        native_api_version=identity.native_api_version,
        native_backend_version=identity.native_backend_version,
    )
    partial = _partial(g_values, 0, 2, leaves, wrong_identity, slot=0)
    with pytest.raises(ValueError, match="Global ordered"):
        merge_contextual_variant_probe_partials_v1((partial,))

    with pytest.raises(ValueError, match="SHA-256"):
        ContextualVariantProbeMergeIdentityV1(
            global_probe_plan_sha256="A" * 64,
            science_identity=identity.science_identity,
            source_tree_sha256=identity.source_tree_sha256,
            native_build_provenance_sha256=(identity.native_build_provenance_sha256),
            build_id=identity.build_id,
            native_backend=identity.native_backend,
            native_execution_backend=identity.native_execution_backend,
            variant_probe_policy=identity.variant_probe_policy,
            native_api_version=identity.native_api_version,
            native_backend_version=identity.native_backend_version,
        )


@pytest.mark.parametrize(
    ("writer", "loader", "suffix"),
    [
        (
            write_contextual_reference_v1,
            load_contextual_reference_v1,
            CONTEXTUAL_REFERENCE_V1_SUFFIX,
        ),
        (
            write_contextual_trait_v1,
            load_contextual_trait_v1,
            CONTEXTUAL_TRAIT_V1_SUFFIX,
        ),
        (write_contextual_fit_v1, load_contextual_fit_v1, CONTEXTUAL_FIT_V1_SUFFIX),
    ],
)
def test_transient_partial_has_no_stable_writer_or_cross_family_loader(
    tmp_path: Path, writer: Any, loader: Any, suffix: str
) -> None:
    g_values = np.ones((2, 1, 3), dtype=np.float64)
    leaves = (_sha("one"), _sha("two"))
    partial = _partial(
        g_values,
        0,
        2,
        leaves,
        _identity(leaves, policy="explicit_rademacher_v1"),
        slot=0,
    )
    with pytest.raises(ValueError, match="Only Contextual"):
        writer(partial, tmp_path / "forbidden")

    path = tmp_path / f"forbidden{suffix}"
    np.savez(
        path,
        same_probe_sums=np.asarray(partial.probe_sums),
        same_probe_cross=np.asarray(partial.within_probe_cross),
    )
    with pytest.raises(
        ValueError,
        match=(
            "key mismatch|lacks its manifest|required member .* is missing|"
            "ZIP member .* is not canonical"
        ),
    ):
        loader(path)
    assert partial.contains_sample_axis is True
    assert partial.durable_artifact is False
    assert not any(name.startswith("write_") for name in probe_merge_v1.__all__)


def test_cmake_sanitizer_contract_is_explicit_and_preserves_normal_release() -> None:
    source = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "option(GWLDCORE_ENABLE_ASAN_UBSAN" in source
    assert "option(GWLDCORE_ENABLE_UBSAN_ONLY" in source
    assert "GWLDCORE_ENABLE_ASAN_UBSAN AND GWLDCORE_ENABLE_UBSAN_ONLY" in source
    assert "-fsanitize=address,undefined" in source
    assert 'set(GWLDCORE_SANITIZER_FLAGS "-fsanitize=undefined")' in source
    assert "-fno-omit-frame-pointer" in source
    assert "-fno-sanitize-recover=all" in source
    assert "function(gwldcore_apply_sanitizer target_name)" in source
    assert "gwldcore_apply_sanitizer(gwldcore_core)" in source
    assert "gwldcore_apply_sanitizer(${target_name})" in source
    assert "set(GWLDCORE_ENABLE_NATIVE_OPT OFF CACHE BOOL" in source
    assert "target_compile_options(gwldcore_core PRIVATE -O3)" in source
    assert "target_compile_options(${target_name} PRIVATE -O3)" in source
