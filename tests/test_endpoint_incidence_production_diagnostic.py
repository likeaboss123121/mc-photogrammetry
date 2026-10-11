"""Fake-only packet wrapper/callsite controls; no production/image imports."""
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "agent swarm instantiation template/runs/SOL-ENDPOINT-INCIDENCE-PRODUCTION-DIAGNOSTIC-01.driver.py"
ORCHESTRATION = ROOT / "src/mc_photogrammetry/pipeline/image_derived_geometry_orchestration.py"


@pytest.mark.parametrize("count,raises,projection_raises", [(0, False, False), (1, False, False),
    (2, False, False), (1, True, False), (1, False, True)])
def test_exact_source_bound_orchestration_wrapper_forwarding_restoration_primary_only(count, raises, projection_raises):
    # Read/hash/AST only; extraction never executes the packet's top-level code.
    packet_path = ROOT / "agent swarm instantiation template/runs/SOL-ENDPOINT-INCIDENCE-PRODUCTION-DIAGNOSTIC-01.packet-control.json"
    import json
    packet = json.loads(packet_path.read_text())
    pin = packet["orchestration_source"]
    raw = ORCHESTRATION.read_bytes()
    assert len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest().lower() == pin["sha256"].lower()
    source = ast.parse(raw)
    assert any(isinstance(n, ast.ImportFrom) and n.module == "mc_photogrammetry.pipeline"
               and any(a.name == "image_derived_joint_graph" and a.asname == "g_mod" for a in n.names)
               for n in source.body)
    callsites = [n for n in ast.walk(source) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "g_mod" and n.func.attr == "derive_image_derived_graph"]
    assert len(callsites) == 1 and callsites[0].lineno == packet["graph_callsite_line"]
    driver = DRIVER.read_bytes()
    assert hashlib.sha256(driver).hexdigest().lower() == packet["worker_driver_sha256"].lower()
    tree = ast.parse(driver)
    wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "observe_graph")
    namespace = {"contextmanager": contextmanager}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), "pinned-packet-wrapper", "exec"), namespace)
    events, graph_calls, solve_calls = [], [], []
    error = RuntimeError("original-source-error")
    projection_error = RuntimeError("projection-only")
    positional, keyword, budget = object(), object(), object()
    counters = MappingProxyType({"visited_assignments": 4})
    result = SimpleNamespace(counters=counters, result_sha256="f" * 64)
    solve_result = object()
    def solver(*args, **kwargs):
        solve_calls.append((args, kwargs))
        if kwargs.get("endpoint_incidence_sink") is not None:
            kwargs["endpoint_incidence_sink"]("registry", {"only": "scalar-fake"})
        return solve_result
    csp = SimpleNamespace(solve_endpoint_hypotheses=solver)
    def original(*args, **kwargs):
        graph_calls.append((args, kwargs))
        assert args == (positional,) and kwargs["source"] is keyword
        sink = kwargs["endpoint_incidence_sink"]
        assert csp.solve_endpoint_hypotheses(positional, source=keyword, budget=budget,
                    endpoint_incidence_sink=sink) is solve_result
        assert csp.solve_endpoint_hypotheses(positional, source=keyword, budget=budget) is solve_result
        if raises: raise error
        return result
    module = SimpleNamespace(derive_image_derived_graph=original, selection=SimpleNamespace(csp=csp))
    def bind(value):
        assert value is result
        if projection_raises: raise projection_error
    collector = SimpleNamespace(sink=lambda e, p: events.append((e, p)), bind_result=bind)
    try:
        with namespace["observe_graph"](module, csp, collector) as state:
            for _ in range(count):
                actual = module.derive_image_derived_graph(positional, source=keyword)
                assert actual is result and actual.counters is counters and actual.result_sha256 == "f" * 64
    except RuntimeError as caught:
        assert raises and caught is error
    else:
        assert not raises
    assert module.derive_image_derived_graph is original and csp.solve_endpoint_hypotheses is solver
    assert state["graph_restored"] and state["csp_restored"]
    assert state["graph_calls"] == state["primary_solves"] == state["validation_solves"] == count
    assert state["registry_events"] == len(events) == count and state["unexpected_sinks"] == 0
    assert state["projection_failure"] is projection_raises
    assert len(graph_calls) == count and len(solve_calls) == 2 * count
    for args, kwargs in graph_calls:
        assert args[0] is positional and kwargs["source"] is keyword
    for index, (args, kwargs) in enumerate(solve_calls):
        assert args[0] is positional and kwargs["source"] is keyword and kwargs["budget"] is budget
        assert ("endpoint_incidence_sink" in kwargs) is (index % 2 == 0)
