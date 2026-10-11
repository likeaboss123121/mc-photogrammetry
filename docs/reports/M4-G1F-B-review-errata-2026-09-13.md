# M4 G1F-B review errata: diagnostic cost and abort accounting

Date: 2026-09-13

This supplements the original G1F-B report without rewriting its frozen one-shot R4 output. The diagnostic R4 call took 0.4781855000 s; the preceding G1F-A call took 0.0989940001 s, a measured 4.83x runtime on that run (+0.3791915 s). This is an observed instrumentation cost, not a universal ratio or a runtime bound. The diagnostic path is still isolated; its signature stores cap is 4,096 per class, and no search budget was changed.

The zero repeated partial/leaf-signature counts should be read only as a traversal integrity observation. Both signatures include the exact q/b labels, so distinct assignments necessarily remain distinct; this does not show that coarser states lack redundancy. In the same R4 search, 1,509 label attempts hit 56 exact cached objective signatures (1,453 cache hits). That existing cache is the evidence for objective-equivalent work coalescing; the exact-label signature census is not a semantic merge test.

The terminal gate `diagnostic_gate_budget_aborted` is explicitly limited to budget exceptions while collecting compatible records or evaluating the objective for a complete-label attempt. Search-node, partial-axis, or Pareto budget exhaustion outside a complete-label evaluation remains represented by the ordinary result status/reason. After a candidate passes objective gates, certificate or buffered-match-row limits can still abort before append; a separate `diagnostic_post_accept_buffering_aborts` counter now records that class. It does not enter the mutually exclusive terminal-label histogram: the label remains counted as `accepted`, so gate counts still sum exactly to `complete_labels`.

A synthetic regression independently forces both post-accept limits (`candidate_certificates_buffered` and `buffered_match_rows`) and verifies one accepted gate, one post-accept buffering abort, an empty candidate list, and unchanged terminal-gate reconciliation. Focused command:

```text
.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py -k "complete_label_gate or post_accept_buffering" -q
```

Result: 4 passed. No R4 rerun was performed; the prior artifact remains immutable and its one million compatibility-check stop occurred before any buffering gate could bind.