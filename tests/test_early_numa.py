from __future__ import annotations

import json

import pytest

from summit import _early_numa as early_numa


class _FakeLibnuma:
    def __init__(self, observed=(0, 1), maximum=3):
        self.observed = tuple(observed)
        self.maximum = int(maximum)
        self.set_calls = []
        self.bind_calls = []
        self.policy_query_calls = []
        self.page_query_calls = []
        self.freed = []
        self.observed_mask = object()
        self.policy_mode = early_numa._MPOL_BIND | early_numa._MPOL_F_STATIC_NODES
        self.policy_nodes = self.observed
        self.range_policy_mode = None
        self.range_policy_nodes = None
        self.page_histogram = None
        self.status_sha256 = "ab" * 32

    def maximum_node(self):
        return self.maximum

    def allocate_mask(self, nodes):
        return ("target", tuple(nodes))

    def free_mask(self, mask):
        self.freed.append(mask)

    def set_membind(self, mask):
        self.set_calls.append(mask)

    def set_static_membind(self, nodes):
        self.set_calls.append(tuple(nodes))
        self.policy_nodes = tuple(nodes)

    def query_policy(self, *, address=None):
        self.policy_query_calls.append(address)
        if address is None:
            return self.policy_mode, self.policy_nodes
        return (
            self.policy_mode
            if self.range_policy_mode is None
            else self.range_policy_mode,
            self.policy_nodes
            if self.range_policy_nodes is None
            else self.range_policy_nodes,
        )

    def bind_memory_range(self, address, length, nodes, *, strict):
        self.bind_calls.append(
            {
                "address": address,
                "length": length,
                "nodes": tuple(nodes),
                "strict": strict,
            }
        )

    def query_page_nodes(
        self, address, page_count, page_size, *, chunk_pages
    ):
        self.page_query_calls.append(
            {
                "address": address,
                "page_count": page_count,
                "page_size": page_size,
                "chunk_pages": chunk_pages,
            }
        )
        histogram = (
            {self.policy_nodes[0]: page_count}
            if self.page_histogram is None
            else dict(self.page_histogram)
        )
        chunks = (page_count + chunk_pages - 1) // chunk_pages
        return histogram, self.status_sha256, chunks

    def get_membind(self):
        return self.observed_mask

    def mask_nodes(self, mask, maximum_node):
        assert mask is self.observed_mask
        assert maximum_node == self.maximum
        return self.observed


@pytest.fixture(autouse=True)
def _clear_attestation(monkeypatch):
    monkeypatch.setattr(early_numa, "_ATTESTATION", None)
    monkeypatch.setattr(early_numa, "_loaded_numeric_modules", lambda: ())
    monkeypatch.delenv("SUMMIT_NUMA_POLICY_APPLIED", raising=False)
    monkeypatch.delenv("SUMMIT_NUMA_POLICY_PROVENANCE", raising=False)
    yield
    # ``_publish_attestation`` writes through ``os.environ`` directly, after
    # the fixture's initial ``delenv`` call.  Remove those process-global
    # markers explicitly so this module cannot opt later native tests into a
    # strict NUMA contract backed by the fake libnuma policy above.
    early_numa._ATTESTATION = None
    early_numa.os.environ.pop("SUMMIT_NUMA_POLICY_APPLIED", None)
    early_numa.os.environ.pop("SUMMIT_NUMA_POLICY_PROVENANCE", None)


def _install_fake_runtime(monkeypatch, *, observed=(0, 1), task_count=1):
    library = _FakeLibnuma(observed=observed)
    monkeypatch.setattr(early_numa, "_read_mems_allowed", lambda: (0, 1, 2, 3))
    monkeypatch.setattr(early_numa, "_read_task_count", lambda: task_count)
    monkeypatch.setattr(early_numa, "_load_libnuma", lambda: library)
    return library


def test_early_membind_is_verified_attested_and_idempotent(monkeypatch):
    library = _install_fake_runtime(monkeypatch)

    attestation = early_numa.apply_early_numa_membind("0-1")

    assert attestation == {
        "schema": "summit.numa_policy_attestation.v1",
        "mode": "membind",
        "requested_nodes": "0-1",
        "effective_nodes": [0, 1],
        "task_count_at_application": 1,
        "applied_before_numeric_import": True,
        "verified": True,
        "source": "libnuma",
        "applied_policy": "libnuma:membind:0,1",
        "static_nodes": True,
        "pid": attestation["pid"],
    }
    json.dumps(attestation)
    assert early_numa.os.environ["SUMMIT_NUMA_POLICY_APPLIED"] == (
        "libnuma:membind:0,1"
    )
    assert early_numa.os.environ["SUMMIT_NUMA_POLICY_PROVENANCE"] == (
        "pre_numeric_import"
    )
    assert len(library.set_calls) == 1

    # A returned JSON copy cannot mutate the private frozen attestation.
    attestation["effective_nodes"].append(3)
    assert early_numa.current_numa_policy_attestation()["effective_nodes"] == [0, 1]

    repeated = early_numa.apply_early_numa_membind("0-1")
    assert repeated["effective_nodes"] == [0, 1]
    assert len(library.set_calls) == 1


def test_early_membind_rejects_conflicting_reentry(monkeypatch):
    _install_fake_runtime(monkeypatch)
    early_numa.apply_early_numa_membind("0-1")

    with pytest.raises(RuntimeError, match="Conflicting NUMA policy request"):
        early_numa.apply_early_numa_membind("2-3")


def test_early_membind_requires_single_task(monkeypatch):
    library = _install_fake_runtime(monkeypatch, task_count=2)

    with pytest.raises(RuntimeError, match="observed 2 process tasks"):
        early_numa.apply_early_numa_membind("0-1")

    assert library.set_calls == []
    assert early_numa.current_numa_policy_attestation() is None


def test_early_membind_rejects_already_loaded_numeric_runtime(monkeypatch):
    monkeypatch.setattr(early_numa, "_loaded_numeric_modules", lambda: ("numpy",))
    monkeypatch.setattr(
        early_numa,
        "_read_mems_allowed",
        lambda: pytest.fail("policy checks must stop before reading NUMA state"),
    )

    with pytest.raises(RuntimeError, match="must precede numerical imports"):
        early_numa.apply_early_numa_membind("0-1")


def test_early_membind_rejects_nodes_outside_mems_allowed(monkeypatch):
    monkeypatch.setattr(early_numa, "_read_mems_allowed", lambda: (0, 1))
    monkeypatch.setattr(
        early_numa,
        "_load_libnuma",
        lambda: pytest.fail("libnuma must not be called for a disallowed mask"),
    )

    with pytest.raises(ValueError, match="subset of Mems_allowed_list"):
        early_numa.apply_early_numa_membind("1-2")


def test_early_membind_rejects_mask_verification_mismatch(monkeypatch):
    library = _install_fake_runtime(monkeypatch)

    def mismatched_policy(*, address=None):
        assert address is None
        return (
            early_numa._MPOL_BIND | early_numa._MPOL_F_STATIC_NODES,
            (0,),
        )

    monkeypatch.setattr(library, "query_policy", mismatched_policy)

    with pytest.raises(RuntimeError, match="static membind verification failed"):
        early_numa.apply_early_numa_membind("0-1")

    assert early_numa.current_numa_policy_attestation() is None


def test_early_membind_rejects_nonstatic_live_policy(monkeypatch):
    library = _install_fake_runtime(monkeypatch)
    library.policy_mode = early_numa._MPOL_BIND

    with pytest.raises(RuntimeError, match="static membind verification failed"):
        early_numa.apply_early_numa_membind("0-1")

    assert early_numa.current_numa_policy_attestation() is None


def test_early_membind_fails_closed_without_libnuma(monkeypatch):
    monkeypatch.setattr(early_numa, "_read_mems_allowed", lambda: (0, 1))
    monkeypatch.setattr(
        early_numa,
        "_load_libnuma",
        lambda: (_ for _ in ()).throw(RuntimeError("libnuma unavailable")),
    )

    with pytest.raises(RuntimeError, match="libnuma unavailable"):
        early_numa.apply_early_numa_membind("0-1")

    assert early_numa.current_numa_policy_attestation() is None


def test_early_membind_fails_closed_on_policy_application_error(monkeypatch):
    library = _install_fake_runtime(monkeypatch)

    def fail_set(_):
        raise OSError(1, "not permitted")

    monkeypatch.setattr(library, "set_static_membind", fail_set)
    with pytest.raises(OSError, match="not permitted"):
        early_numa.apply_early_numa_membind("0-1")

    assert early_numa.current_numa_policy_attestation() is None


def test_numa_argv_preparse_only_acts_on_explicit_membind(monkeypatch):
    calls = []
    monkeypatch.setattr(
        early_numa,
        "apply_early_numa_membind",
        lambda nodes: calls.append(nodes) or {"requested_nodes": nodes},
    )

    assert early_numa.preconfigure_numa_from_argv([]) is None
    assert early_numa.preconfigure_numa_from_argv(
        ["--numa-nodes", "0-1"]
    ) is None
    assert early_numa.preconfigure_numa_from_argv(
        ["--numa-mode", "interleave", "--numa-nodes", "0-1"]
    ) is None
    assert calls == []

    result = early_numa.preconfigure_numa_from_argv(
        ["--numa-mode=membind", "--numa-nodes=2-3"]
    )
    assert result == {"requested_nodes": "2-3"}
    assert calls == ["2-3"]

    result = early_numa.preconfigure_numa_from_argv(
        ["--numa-m", "membind", "--numa-n=0-1"]
    )
    assert result == {"requested_nodes": "0-1"}
    assert calls == ["2-3", "0-1"]

    with pytest.raises(ValueError, match="Ambiguous NUMA option"):
        early_numa.preconfigure_numa_from_argv(["--numa", "membind"])


def test_later_policy_validation_uses_attestation_not_environment(monkeypatch):
    _install_fake_runtime(monkeypatch)
    early_numa.apply_early_numa_membind("0-1")
    monkeypatch.setenv("SUMMIT_NUMA_POLICY_APPLIED", "forged")

    assert early_numa.attest_numa_policy_request("membind", "0-1") is True
    assert early_numa.os.environ["SUMMIT_NUMA_POLICY_APPLIED"] == (
        "libnuma:membind:0,1"
    )
    with pytest.raises(RuntimeError, match="conflicts"):
        early_numa.attest_numa_policy_request("membind", "2-3")


def test_forged_environment_is_not_an_in_process_attestation(monkeypatch):
    monkeypatch.setenv("SUMMIT_NUMA_POLICY_APPLIED", "libnuma:membind:0,1")
    monkeypatch.setenv("SUMMIT_NUMA_POLICY_PROVENANCE", "pre_numeric_import")

    assert early_numa.current_numa_policy_attestation() is None
    assert early_numa.attest_numa_policy_request("membind", "0-1") is False


def test_bound_anonymous_buffer_is_prebound_with_exact_compact_evidence(
    monkeypatch,
):
    library = _install_fake_runtime(monkeypatch)
    early_numa.apply_early_numa_membind("0-1")
    library.bind_calls.clear()
    library.policy_query_calls.clear()
    page_size = early_numa.os.sysconf("SC_PAGE_SIZE")
    byte_count = page_size + 17

    owner, evidence = early_numa.allocate_numa_bound_anonymous_buffer(
        byte_count, (0, 1)
    )
    try:
        assert evidence == {
            "schema": "summit.numa_bound_anonymous_buffer.v1",
            "schema_version": 1,
            "byte_count": byte_count,
            "mapping_bytes": 2 * page_size,
            "page_size": page_size,
            "page_count": 2,
            "selected_nodes": [0, 1],
            "policy_mode": "bind_static_nodes",
            "page_aligned_mapping": True,
            "bound_before_first_touch": True,
            "live_owner_policy_verified": True,
            "range_policy_verified": True,
            "post_decode_complete_page_query": False,
            "page_migration_requested": False,
            "placement_repair_performed": False,
        }
        assert library.policy_query_calls[0] is None
        assert library.policy_query_calls[1] == library.bind_calls[0]["address"]
        assert library.bind_calls == [
            {
                "address": library.bind_calls[0]["address"],
                "length": 2 * page_size,
                "nodes": (0, 1),
                "strict": False,
            }
        ]
        assert not any(
            "address" in key.lower() or "pointer" in key.lower()
            for key in evidence
        )
        json.dumps(evidence)
    finally:
        owner.close()


def test_bound_anonymous_buffer_complete_local_verification(monkeypatch):
    library = _install_fake_runtime(monkeypatch)
    early_numa.apply_early_numa_membind("0-1")
    page_size = early_numa.os.sysconf("SC_PAGE_SIZE")
    byte_count = 2 * page_size + 1
    owner, _ = early_numa.allocate_numa_bound_anonymous_buffer(
        byte_count, (0, 1)
    )
    library.page_histogram = {0: 2, 1: 1}
    try:
        evidence = early_numa.verify_numa_bound_anonymous_buffer(
            owner, byte_count, (0, 1), chunk_pages=2
        )

        assert evidence == {
            "schema": "summit.numa_bound_anonymous_buffer.v1",
            "schema_version": 1,
            "byte_count": byte_count,
            "mapping_bytes": 3 * page_size,
            "page_size": page_size,
            "page_count": 3,
            "selected_nodes": [0, 1],
            "policy_mode": "bind_static_nodes",
            "page_aligned_mapping": True,
            "bound_before_first_touch": True,
            "live_owner_policy_verified": True,
            "range_policy_verified": True,
            "post_decode_complete_page_query": True,
            "post_decode_strict_policy_verified": True,
            "queried_pages": 3,
            "resolved_pages": 3,
            "query_chunks": 2,
            "query_chunk_page_limit": 2,
            "page_migration_requested": False,
            "placement_repair_performed": False,
            "node_histogram": {"0": 2, "1": 1},
            "ordered_status_sha256": "ab" * 32,
            "ordered_status_encoding": (
                f"native_{early_numa.ctypes.sizeof(early_numa.ctypes.c_int) * 8}"
                f"bit_signed_{early_numa.sys.byteorder}"
            ),
            "complete": True,
        }
        assert library.page_query_calls == [
            {
                "address": library.bind_calls[0]["address"],
                "page_count": 3,
                "page_size": page_size,
                "chunk_pages": 2,
            }
        ]
        assert [call["strict"] for call in library.bind_calls] == [False, True]
        assert library.bind_calls[1]["address"] == library.bind_calls[0]["address"]
        assert not any(
            "address" in key.lower() or "pointer" in key.lower()
            for key in evidence
        )
        json.dumps(evidence)
    finally:
        owner.close()


def test_bound_anonymous_buffer_rejects_remote_page_without_repair(monkeypatch):
    library = _install_fake_runtime(monkeypatch)
    early_numa.apply_early_numa_membind("0-1")
    page_size = early_numa.os.sysconf("SC_PAGE_SIZE")
    owner, _ = early_numa.allocate_numa_bound_anonymous_buffer(
        2 * page_size, (0, 1)
    )
    library.page_histogram = {0: 1, 2: 1}
    try:
        with pytest.raises(RuntimeError, match="pages outside its selected nodes"):
            early_numa.verify_numa_bound_anonymous_buffer(
                owner, 2 * page_size, (0, 1), chunk_pages=1
            )

        assert [call["strict"] for call in library.bind_calls] == [False]
        assert library.page_query_calls[0]["page_count"] == 2
    finally:
        owner.close()


@pytest.mark.parametrize(
    ("range_mode", "range_nodes"),
    [
        (early_numa._MPOL_BIND, (0, 1)),
        (
            early_numa._MPOL_BIND | early_numa._MPOL_F_STATIC_NODES,
            (0,),
        ),
    ],
)
def test_bound_anonymous_buffer_rejects_wrong_range_policy(
    monkeypatch, range_mode, range_nodes
):
    library = _install_fake_runtime(monkeypatch)
    early_numa.apply_early_numa_membind("0-1")
    library.range_policy_mode = range_mode
    library.range_policy_nodes = range_nodes

    with pytest.raises(RuntimeError, match="did not retain its exact static bind"):
        early_numa.allocate_numa_bound_anonymous_buffer(1, (0, 1))

    assert len(library.bind_calls) == 1
    assert library.bind_calls[0]["strict"] is False
