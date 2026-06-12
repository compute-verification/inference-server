//! SP1 guest: bounded-cost partition of a task graph.
//!
//! Statement: "I know a partition of the committed task graph into k stages
//! such that
//!   (a) every dependency edge flows from an earlier-or-equal stage to a
//!       later-or-equal stage — i.e. the stages can be executed in order;
//!   (b) each stage's summed FLOPs are <= C;
//!   (c) each stage's summed NON-WHITELISTED input tokens are <= S
//!       (whitelisted inputs are publicly-known constants and free to pass)."
//!
//! The partition assignment is the PRIVATE witness — the proof shows such a
//! partition exists without revealing it. The graph's cost view (per-node
//! flops, input size, whitelist flag, and the edge list) is re-encoded and
//! hashed in-guest, so the committed digest binds the proof to exactly those
//! numbers; the verifier recomputes the digest from the published graph JSON
//! (modules.proof_server.partition.graph_partition_digest) and rejects on
//! mismatch.
//!
//! Public outputs (committed at the end of `main`, in this order):
//!   - 32 bytes: auditor_nonce
//!   - 32 bytes: graph_digest = sha256(taskgraph-partition-v1 encoding)
//!   - 8 bytes:  cap_flops C (le u64)
//!   - 8 bytes:  cap_input S (le u64)
//!   - 4 bytes:  n_nodes (le u32)
//!   - 4 bytes:  n_parts (le u32)

#![no_main]

sp1_zkvm::entrypoint!(main);

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use proof_server_lib::{partition_graph_bytes, PartitionInput, PARTITION_PUBLIC_OUTPUT_LEN};
use sha2::{Digest, Sha256};

pub fn main() {
    let input: PartitionInput = sp1_zkvm::io::read();

    let n = input.flops.len();
    assert!(n > 0, "graph must contain at least one node");
    assert!(n <= u32::MAX as usize, "node count exceeds u32");
    assert_eq!(input.in_size.len(), n, "in_size length must equal node count");
    assert_eq!(input.whitelisted.len(), n, "whitelisted length must equal node count");
    assert_eq!(input.parts.len(), n, "parts length must equal node count");
    for &w in &input.whitelisted {
        assert!(w <= 1, "whitelist flags must be 0 or 1");
    }

    // Edges: strictly lex-increasing (sorted + deduped — keeps the hashed
    // encoding canonical) with src < dst. Node ids ascend in topological
    // order (enforced by the graph builder), so src < dst everywhere makes
    // the graph a DAG by construction.
    let mut prev: Option<(u32, u32)> = None;
    for &(s, d) in &input.edges {
        assert!((d as usize) < n, "edge endpoint out of range");
        assert!(s < d, "edge must go forward in node order");
        if let Some(p) = prev {
            assert!((s, d) > p, "edges must be strictly lex-sorted (sorted + deduped)");
        }
        prev = Some((s, d));
    }

    // Parts: ids must cover exactly 0..n_parts (no gaps — otherwise the
    // committed part count is gameable) and never decrease along an edge.
    // Monotone-along-edges means the quotient graph is acyclic with this
    // numbering as a topological order: the stages really are executable
    // one after another.
    let mut n_parts: u32 = 0;
    for &p in &input.parts {
        if p >= n_parts {
            n_parts = p + 1;
        }
    }
    assert!((n_parts as usize) <= n, "more parts than nodes");
    let mut used = vec![false; n_parts as usize];
    for &p in &input.parts {
        used[p as usize] = true;
    }
    for u in &used {
        assert!(*u, "part ids must be contiguous 0..n_parts");
    }
    for &(s, d) in &input.edges {
        assert!(
            input.parts[s as usize] <= input.parts[d as usize],
            "edge crosses backward between parts",
        );
    }

    // Per-part budgets. u128 accumulators: no overflow for any u64 inputs.
    let mut flops_sum = vec![0u128; n_parts as usize];
    let mut input_sum = vec![0u128; n_parts as usize];
    for i in 0..n {
        let p = input.parts[i] as usize;
        flops_sum[p] += input.flops[i] as u128;
        if input.whitelisted[i] == 0 {
            input_sum[p] += input.in_size[i] as u128;
        }
    }
    for p in 0..n_parts as usize {
        assert!(flops_sum[p] <= input.cap_flops as u128, "part exceeds FLOP cap C");
        assert!(input_sum[p] <= input.cap_input as u128, "part exceeds input cap S");
    }

    // Bind the proof to the graph: hash the canonical cost-view encoding.
    let graph_bytes =
        partition_graph_bytes(&input.flops, &input.in_size, &input.whitelisted, &input.edges);
    let mut hasher = Sha256::new();
    hasher.update(&graph_bytes);
    let graph_digest: [u8; 32] = hasher.finalize().into();

    let mut out: Vec<u8> = Vec::with_capacity(PARTITION_PUBLIC_OUTPUT_LEN);
    out.extend_from_slice(&input.auditor_nonce);
    out.extend_from_slice(&graph_digest);
    out.extend_from_slice(&input.cap_flops.to_le_bytes());
    out.extend_from_slice(&input.cap_input.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&n_parts.to_le_bytes());
    sp1_zkvm::io::commit_slice(&out);
}
