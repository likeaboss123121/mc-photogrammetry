import ast
from pathlib import Path

WORKER=Path(__file__).parents[1]/".tmp-run-bj-selected-pair-worker.py"

def test_probe_supplies_connection_graph_and_keeps_match_selection_unchanged():
    tree=ast.parse(WORKER.read_text(encoding="utf-8"))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Attribute)
           and node.func.attr=="derive_image_derived_graph_attribution"]
    production=[call for call in calls if any(
        row.arg=="matcher_witness_payload_sink" for row in call.keywords)]
    assert len(production)==1
    keywords={row.arg:row.value for row in production[0].keywords}
    assert isinstance(keywords["connection_graph"],ast.Attribute)
    assert (keywords["connection_graph"].value.id,
            keywords["connection_graph"].attr)==("entry","graph")
    selections=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Name) and node.func.id=="min"]
    assert any(isinstance(node.args[0],ast.Name) and node.args[0].id=="matcher"
               for node in selections)

def test_graph_comparison_uses_one_input_set_and_no_witness_sinks():
    tree=ast.parse(WORKER.read_text(encoding="utf-8"))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Attribute)
           and node.func.attr=="derive_image_derived_graph_attribution"]
    graph_only=[call for call in calls if not any(
        row.arg=="matcher_witness_payload_sink" for row in call.keywords)]
    assert len(graph_only)==2
    keyword_sets=[{row.arg for row in call.keywords} for call in graph_only]
    assert sum("connection_graph" in names for names in keyword_sets)==1
    assert all("matcher_witness_sink" not in names and
               "matcher_witness_payload_sink" not in names and
               "witness_budget" not in names for names in keyword_sets)
    for call in graph_only:
        values={row.arg:ast.unparse(row.value) for row in call.keywords}
        assert values["session"]=="entry.session"
        assert values["lattice_result"]=="measured"
        assert values["signatures_result"]=="signatures"

def test_probe_persists_normalized_graph_trace_and_solver_provenance():
    source=WORKER.read_text(encoding="utf-8")
    for field in ("canonical_graph_record","canonical_trace_record",
                  "normalized_solver_input","solver_input_sha256",
                  "expected_graph_sha256","runtime_graph_sha256",
                  "source_label_paths_excluded"):
        assert field in source
    assert "graph_record_mod._payload(graph)" in source
    assert "attribution.to_record()" in source

def test_footprint_attempt_preserves_and_binds_historical_checkpoint():
    source=WORKER.read_text(encoding="utf-8")
    assert ".tmp-bp2-endpoint-footprint-selection.json" in source
    assert ".tmp-bp2-endpoint-footprint-audit.json" in source
    assert "historical=ROOT/'.tmp-bn-highest-overlap-selection.json'" in source
    assert "original_bo_checkpoint_file_sha256" in source
    assert "original_bo_checkpoint_sha256" in source
    harness=(WORKER.parent/".tmp-run-bj-selected-pair-harness.py").read_text(
        encoding="utf-8")
    assert ".tmp-bp2-endpoint-footprint-audit.json" in harness
    assert ".tmp-bo-endpoint-footprint-audit.json" not in harness

def test_footprint_digest_mismatch_is_a_typed_refusal():
    tree=ast.parse(WORKER.read_text(encoding="utf-8"))
    source=ast.unparse(tree)
    assert "'status': 'complete' if parity else 'refused'" in source
    assert "'reason': '' if parity else 'replay_mismatch'" in source
    assert "'diagnostic': canon(diagnostics[0][0]) if parity else None" in source
    assert "'sampling_replay_count': replay_calls[0]" in source
