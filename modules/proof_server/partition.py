"""Bounded-cost partition of a task graph — Python side of the SP1 statement.

The SP1 program (``sp1/partition-program``) proves: *there exists a partition
of the committed task graph into stages such that every dependency edge flows
forward (the stages are executable in order), each stage's summed FLOPs are
<= C, and each stage's summed NON-whitelisted input tokens are <= S*. The
partition is the private witness; the public outputs are only
(nonce, graph_digest, C, S, n_nodes, n_parts).

This module owns everything the host needs around that statement:

  * ``partition_graph_bytes(graph)``   — the canonical cost-view encoding the
    guest re-builds and hashes. MUST stay byte-identical to the Rust
    ``proof_server_lib::partition_graph_bytes``.
  * ``graph_partition_digest(graph)``  — sha256 over those bytes; what a
    verifier recomputes from the published graphs.json scene and compares
    against the proof's committed ``graph_digest``.
  * ``plan_partition(graph, C, S)``    — greedy planner producing a valid
    witness (or raising if none can exist).
  * ``check_partition(...)``           — pure-Python reference checker with
    the exact semantics of the guest's asserts (fast pre-flight + tests).
  * ``sp1_input_json(...)``            — the stdin document for the
    ``partition-host`` binary (--execute / --prove).

A node's input size is its ``tokens`` (what the pass ingests — the same
number the viz annotates on incoming edges); ``whitelisted`` nodes ingest a
publicly-known constant and count 0 toward S. FLOPs always count toward C:
the whitelist makes *passing* a known input free, never the compute.
"""
from __future__ import annotations

import hashlib
import json
import struct

PARTITION_GRAPH_MAGIC = b"taskgraph-partition-v1\n"

_U32_MAX = 2**32 - 1
_U64_MAX = 2**64 - 1


class PartitionError(ValueError):
    """Raised for malformed graphs, invalid partitions, or infeasible caps."""


def _as_int(value, what: str, bound: int) -> int:
    """Exact non-negative integer in [0, bound] (rejects non-integral floats)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PartitionError(f"{what} must be a number, got {type(value).__name__}")
    if isinstance(value, float):
        if not value.is_integer():
            raise PartitionError(f"{what} must be integral, got {value}")
        value = int(value)
    if not 0 <= value <= bound:
        raise PartitionError(f"{what} out of range [0, {bound}]: {value}")
    return value


def graph_cost_view(graph: dict) -> tuple[list[int], list[int], list[int], list[tuple[int, int]]]:
    """Extract (flops, in_size, whitelisted, edges) in canonical form.

    Nodes are taken in ascending-id order (the builder emits ids in
    topological order); edges become (src_index, dst_index) pairs,
    deduplicated and lexicographically sorted, and must go forward
    (src < dst) — the same invariants the guest asserts.
    """
    nodes = sorted(graph.get("nodes") or [], key=lambda n: n["id"])
    if not nodes:
        raise PartitionError("graph has no nodes")
    index_of = {}
    for i, n in enumerate(nodes):
        if n["id"] in index_of:
            raise PartitionError(f"duplicate node id {n['id']}")
        index_of[n["id"]] = i

    flops = [_as_int(n.get("flops", 0), f"node {n['id']} flops", _U64_MAX) for n in nodes]
    in_size = [_as_int(n.get("tokens", 0), f"node {n['id']} tokens", _U32_MAX) for n in nodes]
    whitelisted = [1 if n.get("whitelisted") else 0 for n in nodes]

    edges = set()
    for e in graph.get("edges") or []:
        try:
            s, d = index_of[e["src"]], index_of[e["dst"]]
        except KeyError as exc:
            raise PartitionError(f"edge references unknown node id {exc}") from exc
        if s >= d:
            raise PartitionError(f"edge {e['src']}->{e['dst']} does not go forward in id order")
        edges.add((s, d))
    return flops, in_size, whitelisted, sorted(edges)


def partition_graph_bytes(graph: dict) -> bytes:
    """Canonical cost-view encoding — byte-identical to the Rust side."""
    flops, in_size, whitelisted, edges = graph_cost_view(graph)
    out = bytearray(PARTITION_GRAPH_MAGIC)
    out += struct.pack("<II", len(flops), len(edges))
    for f, t, w in zip(flops, in_size, whitelisted):
        out += struct.pack("<QIB", f, t, w)
    for s, d in edges:
        out += struct.pack("<II", s, d)
    return bytes(out)


def graph_partition_digest(graph: dict) -> str:
    """``sha256:<hex>`` digest a verifier recomputes from the published graph."""
    return "sha256:" + hashlib.sha256(partition_graph_bytes(graph)).hexdigest()


def _node_input(in_size: int, whitelisted: int) -> int:
    return 0 if whitelisted else in_size


def plan_partition(graph: dict, cap_flops: int, cap_input: int) -> list[int]:
    """Greedy planner: walk nodes in id (= topological) order and pack each
    into the current part until a budget would overflow, then open a new one.

    Contiguous-in-id-order parts automatically satisfy the guest's
    edge-monotonicity check (edges only go forward in id order). Raises
    ``PartitionError`` iff NO partition can exist: some single node exceeds
    a cap on its own (singleton parts are always available otherwise).
    """
    cap_flops = _as_int(cap_flops, "cap_flops", _U64_MAX)
    cap_input = _as_int(cap_input, "cap_input", _U64_MAX)
    flops, in_size, whitelisted, _ = graph_cost_view(graph)

    parts: list[int] = []
    part = 0
    acc_f = acc_i = 0
    for i, (f, t, w) in enumerate(zip(flops, in_size, whitelisted)):
        t_eff = _node_input(t, w)
        if f > cap_flops or t_eff > cap_input:
            raise PartitionError(
                f"infeasible: node index {i} alone exceeds a cap "
                f"(flops={f} vs C={cap_flops}, input={t_eff} vs S={cap_input})")
        if parts and (acc_f + f > cap_flops or acc_i + t_eff > cap_input):
            part += 1
            acc_f = acc_i = 0
        acc_f += f
        acc_i += t_eff
        parts.append(part)
    return parts


def check_partition(graph: dict, parts: list[int], cap_flops: int, cap_input: int) -> dict:
    """Reference checker mirroring the guest's asserts exactly.

    Returns summary stats ``{n_nodes, n_parts, max_part_flops,
    max_part_input}`` on success; raises ``PartitionError`` otherwise.
    """
    cap_flops = _as_int(cap_flops, "cap_flops", _U64_MAX)
    cap_input = _as_int(cap_input, "cap_input", _U64_MAX)
    flops, in_size, whitelisted, edges = graph_cost_view(graph)
    n = len(flops)
    if len(parts) != n:
        raise PartitionError(f"parts length {len(parts)} != node count {n}")
    parts = [_as_int(p, "part id", _U32_MAX) for p in parts]

    n_parts = max(parts) + 1
    if n_parts > n:
        raise PartitionError("more parts than nodes")
    if set(parts) != set(range(n_parts)):
        raise PartitionError("part ids must be contiguous 0..n_parts")
    for s, d in edges:
        if parts[s] > parts[d]:
            raise PartitionError(f"edge {s}->{d} crosses backward between parts")

    flops_sum = [0] * n_parts
    input_sum = [0] * n_parts
    for i in range(n):
        flops_sum[parts[i]] += flops[i]
        input_sum[parts[i]] += _node_input(in_size[i], whitelisted[i])
    for p in range(n_parts):
        if flops_sum[p] > cap_flops:
            raise PartitionError(f"part {p} exceeds FLOP cap: {flops_sum[p]} > {cap_flops}")
        if input_sum[p] > cap_input:
            raise PartitionError(f"part {p} exceeds input cap: {input_sum[p]} > {cap_input}")
    return {
        "n_nodes": n,
        "n_parts": n_parts,
        "max_part_flops": max(flops_sum),
        "max_part_input": max(input_sum),
    }


def sp1_input_json(graph: dict, parts: list[int], cap_flops: int, cap_input: int,
                   auditor_nonce: str = "00" * 32) -> str:
    """The stdin document for the ``partition-host`` binary."""
    if not (isinstance(auditor_nonce, str) and len(auditor_nonce) == 64):
        raise PartitionError("auditor_nonce must be 64 hex chars")
    try:
        bytes.fromhex(auditor_nonce)
    except ValueError as exc:
        raise PartitionError(f"auditor_nonce is not hex: {exc}") from exc
    flops, in_size, whitelisted, edges = graph_cost_view(graph)
    return json.dumps({
        "auditor_nonce": auditor_nonce,
        "cap_flops": _as_int(cap_flops, "cap_flops", _U64_MAX),
        "cap_input": _as_int(cap_input, "cap_input", _U64_MAX),
        "flops": flops,
        "in_size": in_size,
        "whitelisted": whitelisted,
        "edges": [list(e) for e in edges],
        "parts": [_as_int(p, "part id", _U32_MAX) for p in parts],
    })
