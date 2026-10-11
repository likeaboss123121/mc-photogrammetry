import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export.attribution import box_face_corners
from mc_photogrammetry.reconstruction import raycast
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.face_sampling import (
    CUBE_GEOMETRY,
    EXACT_MODEL,
    MODEL_METADATA_UNAVAILABLE,
    PASS_SURFACE,
    PASS_SURFACE_PROPOSED,
    STATES_INCOMPLETE,
    STATES_UNAVAILABLE,
    FaceSample,
)
from mc_photogrammetry.reconstruction.raycast import reconstruct_scene_surfaces
from mc_photogrammetry.reconstruction.scene import SceneTransform
from mc_photogrammetry.reconstruction.facade import PaletteColor


def test_scene_raycast_recovers_visible_blocks_inside_bounds():
    image = np.full((128, 128, 3), 125, dtype=np.uint8)
    camera = CameraModel((4.0, 4.0, -10.0), 0.0, 0.0, eye_height=0.0)
    scene = SceneTransform((0, 0, 0), (8, 8, 8))
    palette = (PaletteColor(BlockState("minecraft:stone"), (125, 125, 125)),)

    volume = reconstruct_scene_surfaces({"frame.png": image}, {"frame.png": camera}, scene, palette=palette)

    assert volume.blocks
    assert all(0 <= coordinate[index] < scene.size[index] for coordinate in volume.blocks for index in range(3))

def test_a_badly_evidenced_cell_can_adopt_its_neighbours_label():
    """Adjacency alone can only re-rank a cell's own candidates.

    A cell whose evidence never proposed the right answer cannot be rescued by any
    amount of coupling, which is why `fallback_risk` measured inert from 0 to 40.
    `adopt_boost` admits the neighbours' majority as a candidate in its own right.
    Worth about 0.2pp when measured, so it ships off; this pins the mechanism.
    """
    import numpy as np

    from mc_photogrammetry.reconstruction.raycast import _smooth_adjacent_states

    right = BlockState("minecraft:stone")
    wrong = BlockState("minecraft:dirt")
    merged = {}
    for x in range(3):
        for z in range(3):
            centre = (x, 0, z) == (1, 0, 1)
            # The middle cell never considered `right` at all.
            scores = {wrong: 1.0} if centre else {right: 1.0, wrong: 0.1}
            merged[(x, 0, z)] = (wrong if centre else right, 0.0, set(),
                                 [wrong if centre else right], scores)

    without = {k: dict(v[4]) for k, v in merged.items()}
    _smooth_adjacent_states(merged, 1.0, 2, coupling_boost={(1, 0, 1): 5.0},
                            adopt_boost=0.0)
    assert max(merged[(1, 0, 1)][4], key=merged[(1, 0, 1)][4].get) == wrong

    merged = {k: (v[0], v[1], v[2], v[3], dict(without[k])) for k, v in merged.items()}
    _smooth_adjacent_states(merged, 1.0, 2, coupling_boost={(1, 0, 1): 5.0},
                            adopt_boost=2.0)
    assert max(merged[(1, 0, 1)][4], key=merged[(1, 0, 1)][4].get) == right


# ---------------------------------------------------------------------------
# M2-W3: threading an existing-hypothesis snapshot of BlockState into the two
# production sampling calls. Exact baked-state geometry is selected only when
# every target of a call is covered; otherwise that call stays on cube geometry.


def _encoded_cutout_alpha():
    """Opaque only in the rightmost quarter of the face; the rest is transparent."""
    import base64
    from io import BytesIO

    from PIL import Image

    alpha = np.zeros((16, 16), dtype=np.uint8)
    alpha[:, :4] = 255
    buffer = BytesIO()
    Image.fromarray(alpha, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _cutout_palette():
    cutout = BlockState("minecraft:cutout")
    dark = BlockState("minecraft:dark")
    mid = BlockState("minecraft:mid")
    corners = tuple(
        tuple(float(value) for value in point)
        for point in box_face_corners((0, 0, 0), (1, 1, 1), "north")
    )
    palette = (
        PaletteColor(
            cutout, (250, 250, 250),
            model_faces=(("north", corners, (0, 0, -1), None, _encoded_cutout_alpha()),),
        ),
        PaletteColor(dark, (25, 25, 25)),
        PaletteColor(mid, (125, 125, 125)),
    )
    return cutout, dark, mid, palette


def _cutout_camera():
    return CameraModel((0.5, 0.5, -3.0), 0.0, 0.0, fov_degrees=90.0, eye_height=0.0)


def _cutout_frame(background):
    """The cut-out's opaque quarter in its own colour, the rest a contrasting hue."""
    image = np.full((128, 128, 3), background, dtype=np.uint8)
    image[:, 70:] = 200
    return image


def _cutout_reconstruct(image, *, states, palette, scene=None, proposals=None, diagnostics=None):
    return raycast.reconstruct_scene_surfaces(
        {"a.png": image}, {"a.png": _cutout_camera()},
        scene if scene is not None else SceneTransform((0, 0, 0), (1, 1, 1)),
        palette=palette,
        propose_colour_gap=0.0,
        adjacency_strength=0.0,
        simulate_lighting=False,
        correct_light_cast=False,
        reconcile_exposure=False,
        self_calibrate=False,
        surface_only=False,
        fit_partial_shapes=False,
        minimum_visible_fraction=0.1,
        extra_proposals=proposals,
        sampling_states=states,
        diagnostics=diagnostics,
        usable_masks={"a.png": np.ones(image.shape[:2], dtype=bool)},
    )


def _volume_signature(volume):
    return sorted(
        (coordinate, observation.state.name) for coordinate, observation in volume.items()
    )


def test_exact_model_cutout_sampling_end_to_end():
    """The real sampler/renderer must own only the cut-out's opaque pixels.

    With exact geometry the transparent quarter is not evidence, so a background
    behind it cannot reach the classifier; the cube fallback samples straight
    through the whole face and does exactly that.
    """
    cutout, _dark, _mid, palette = _cutout_palette()
    exact_dark_diagnostics: dict[str, object] = {}
    exact_mid_diagnostics: dict[str, object] = {}
    cube_dark_diagnostics: dict[str, object] = {}
    cube_mid_diagnostics: dict[str, object] = {}

    exact_dark = _cutout_reconstruct(
        _cutout_frame(20), states={(0, 0, 0): cutout}, palette=palette,
        diagnostics=exact_dark_diagnostics)
    exact_mid = _cutout_reconstruct(
        _cutout_frame(100), states={(0, 0, 0): cutout}, palette=palette,
        diagnostics=exact_mid_diagnostics)
    cube_dark = _cutout_reconstruct(
        _cutout_frame(20), states=None, palette=palette, diagnostics=cube_dark_diagnostics)
    cube_mid = _cutout_reconstruct(
        _cutout_frame(100), states=None, palette=palette, diagnostics=cube_mid_diagnostics)

    # Real ownership: the exact call resolves the owned face; the cube call's
    # half-transparent patch fails the uniformity gate and leans on its centre.
    assert exact_dark_diagnostics["resolved_face_samples"] == 1
    assert cube_dark_diagnostics["resolved_face_samples"] == 0
    assert cube_dark_diagnostics["fallback_samples"] == 1

    # Transparent-pixel invariance: changing what stands behind the cut-out leaves
    # the exact reconstruction alone, and moves the cube one.
    assert _volume_signature(exact_dark) == [((0, 0, 0), "minecraft:cutout")]
    assert _volume_signature(exact_mid) == _volume_signature(exact_dark)
    assert _volume_signature(cube_dark) == [((0, 0, 0), "minecraft:dark")]
    assert _volume_signature(cube_mid) == [((0, 0, 0), "minecraft:mid")]

    exact_decision = exact_dark_diagnostics["sampling_geometry"][0]
    assert exact_decision["mode"] == EXACT_MODEL
    assert exact_decision["fallback_reason"] is None
    assert (exact_decision["pass_name"], exact_decision["target_count"]) == (PASS_SURFACE, 1)
    cube_decision = cube_dark_diagnostics["sampling_geometry"][0]
    assert cube_decision["mode"] == CUBE_GEOMETRY
    assert cube_decision["fallback_reason"] == STATES_UNAVAILABLE
    assert cube_decision["missing_state_count"] == 1


def test_exact_sampling_records_both_passes_with_complete_states():
    cutout, _dark, _mid, palette = _cutout_palette()
    diagnostics: dict[str, object] = {}

    volume = _cutout_reconstruct(
        _cutout_frame(20),
        states={(0, 0, 0): cutout, (0, 0, 1): cutout},
        palette=palette,
        scene=SceneTransform((0, 0, 0), (1, 1, 2)),
        proposals={(0, 0, 1)},
        diagnostics=diagnostics,
    )

    decisions = diagnostics["sampling_geometry"]
    assert [decision["pass_name"] for decision in decisions] == [PASS_SURFACE, PASS_SURFACE_PROPOSED]
    assert [decision["mode"] for decision in decisions] == [EXACT_MODEL, EXACT_MODEL]
    assert [decision["target_count"] for decision in decisions] == [1, 2]
    assert diagnostics["extra_proposals_new"] == 1
    assert _volume_signature(volume) == [((0, 0, 0), "minecraft:cutout")]


def test_second_pass_falls_back_whole_call_when_a_proposal_is_uncovered():
    cutout, dark, _mid, palette = _cutout_palette()
    cases = {
        "missing_state": ({(0, 0, 0): cutout}, STATES_INCOMPLETE),
        "missing_model": ({(0, 0, 0): cutout, (0, 0, 1): dark}, MODEL_METADATA_UNAVAILABLE),
    }

    for name, (states, reason) in cases.items():
        diagnostics: dict[str, object] = {}
        _cutout_reconstruct(
            _cutout_frame(20), states=states, palette=palette,
            scene=SceneTransform((0, 0, 0), (1, 1, 2)),
            proposals={(0, 0, 1)}, diagnostics=diagnostics)

        first, second = diagnostics["sampling_geometry"]
        assert first["mode"] == EXACT_MODEL, name
        # The first pass stays exact; the second keeps the whole call on cube
        # geometry rather than mixing ownership inside one depth buffer.
        assert second["mode"] == CUBE_GEOMETRY, name
        assert second["fallback_reason"] == reason, name
        assert second["target_count"] == 2, name


def test_sampling_geometry_diagnostics_json_round_trip_over_both_passes():
    """`diagnostics["sampling_geometry"]` must survive ``json.dumps`` intact.

    Diagnostics are serialized and compared by callers, so the stored value is a
    tuple of primitive records rather than the live frozen decisions. Exact and
    whole-call fallback decisions are both walked through the real two-pass
    reconstruction and compared field for field after a real encode/decode.
    """
    import json

    cutout, dark, _mid, palette = _cutout_palette()
    exact = {
        "pass_name": PASS_SURFACE,
        "mode": EXACT_MODEL,
        "target_count": 1,
        "missing_state_count": 0,
        "missing_model_count": 0,
        "fallback_reason": None,
    }
    exact_proposed = {
        "pass_name": PASS_SURFACE_PROPOSED,
        "mode": EXACT_MODEL,
        "target_count": 2,
        "missing_state_count": 0,
        "missing_model_count": 0,
        "fallback_reason": None,
    }
    cases = {
        "exact": (
            {(0, 0, 0): cutout, (0, 0, 1): cutout},
            [exact, exact_proposed],
        ),
        "states_incomplete": (
            {(0, 0, 0): cutout},
            [
                exact,
                {
                    "pass_name": PASS_SURFACE_PROPOSED,
                    "mode": CUBE_GEOMETRY,
                    "target_count": 2,
                    "missing_state_count": 1,
                    "missing_model_count": 0,
                    "fallback_reason": STATES_INCOMPLETE,
                },
            ],
        ),
        "model_metadata_unavailable": (
            {(0, 0, 0): cutout, (0, 0, 1): dark},
            [
                exact,
                {
                    "pass_name": PASS_SURFACE_PROPOSED,
                    "mode": CUBE_GEOMETRY,
                    "target_count": 2,
                    "missing_state_count": 0,
                    "missing_model_count": 1,
                    "fallback_reason": MODEL_METADATA_UNAVAILABLE,
                },
            ],
        ),
    }

    for name, (states, expected) in cases.items():
        diagnostics: dict[str, object] = {}
        _cutout_reconstruct(
            _cutout_frame(20), states=states, palette=palette,
            scene=SceneTransform((0, 0, 0), (1, 1, 2)),
            proposals={(0, 0, 1)}, diagnostics=diagnostics)

        records = diagnostics["sampling_geometry"]
        # No `default=` hook: the standard library encoder must accept it as-is.
        decoded = json.loads(json.dumps(records, sort_keys=True))
        assert decoded == expected, name
        assert [record["pass_name"] for record in decoded] == [
            PASS_SURFACE, PASS_SURFACE_PROPOSED], name
        for record in decoded:
            assert all(
                value is None or type(value) in (str, int)
                for value in record.values()
            ), name

    # A call with no proposal pass records exactly one JSON-safe entry, not a
    # synthetic second one.
    diagnostics = {}
    _cutout_reconstruct(_cutout_frame(20), states=None, palette=palette, diagnostics=diagnostics)
    single = json.loads(json.dumps(diagnostics["sampling_geometry"]))
    assert single == [
        {
            "pass_name": PASS_SURFACE,
            "mode": CUBE_GEOMETRY,
            "target_count": 1,
            "missing_state_count": 1,
            "missing_model_count": 0,
            "fallback_reason": STATES_UNAVAILABLE,
        }
    ]


def test_sampling_states_cannot_create_coordinates_or_seed_scores():
    cutout, dark, _mid, palette = _cutout_palette()
    diagnostics: dict[str, object] = {}

    baseline = _cutout_reconstruct(
        _cutout_frame(20), states={(0, 0, 0): cutout}, palette=palette)
    injected = _cutout_reconstruct(
        _cutout_frame(20), states={(0, 0, 0): cutout, (5, 5, 5): dark},
        palette=palette, diagnostics=diagnostics)

    assert _volume_signature(injected) == _volume_signature(baseline)
    assert (5, 5, 5) not in injected.blocks
    decision = diagnostics["sampling_geometry"][0]
    assert decision["mode"] == EXACT_MODEL
    assert decision["target_count"] == 1


def test_cube_geometry_passes_no_blocks_and_exact_threads_the_target_set(monkeypatch):
    cutout, _dark, _mid, palette = _cutout_palette()
    calls = []

    def spy(coordinates, camera, scene, image, **kwargs):
        calls.append((set(coordinates), kwargs.get("blocks"), kwargs.get("palette")))
        return []

    monkeypatch.setattr(raycast, "sample_visible_faces", spy)
    common = dict(
        palette=palette,
        propose_colour_gap=0.0,
        adjacency_strength=0.0,
        simulate_lighting=False,
        correct_light_cast=False,
        reconcile_exposure=False,
        self_calibrate=False,
        surface_only=False,
        fit_partial_shapes=False,
    )

    # Legacy default: no snapshot, so no frame ever receives exact-model inputs.
    raycast.reconstruct_scene_surfaces(
        {"a.png": _cutout_frame(20)}, {"a.png": _cutout_camera()},
        SceneTransform((0, 0, 0), (1, 1, 1)), **common)
    assert len(calls) == 1
    assert calls[0][1] is None and calls[0][2] is None

    # Exact: the sampler receives a complete mapping of exactly this call's
    # targets, keyed by the same palette object the call was given.
    calls.clear()
    raycast.reconstruct_scene_surfaces(
        {"a.png": _cutout_frame(20)}, {"a.png": _cutout_camera()},
        SceneTransform((0, 0, 0), (1, 1, 1)),
        sampling_states={(0, 0, 0): cutout}, **common)
    assert len(calls) == 1
    coordinates, blocks, passed_palette = calls[0]
    assert set(blocks) == coordinates
    assert set(blocks.values()) == {cutout}
    assert passed_palette is palette

    # With a proposal the second call runs too, and its exact mapping covers the
    # whole ``surface | proposed`` target set - the surface target is not lost.
    calls.clear()
    raycast.reconstruct_scene_surfaces(
        {"a.png": _cutout_frame(20)}, {"a.png": _cutout_camera()},
        SceneTransform((0, 0, 0), (1, 1, 2)),
        extra_proposals={(0, 0, 1)},
        sampling_states={(0, 0, 0): cutout, (0, 0, 1): cutout}, **common)
    assert len(calls) == 2
    assert calls[0][0] == {(0, 0, 0)}
    assert set(calls[1][0]) == {(0, 0, 0), (0, 0, 1)}
    assert set(calls[1][1]) == calls[1][0]


def test_invalid_sampling_states_raise_before_any_sampling(monkeypatch):
    _cutout, _dark, _mid, palette = _cutout_palette()
    calls = []
    monkeypatch.setattr(raycast, "sample_visible_faces", lambda *a, **k: calls.append(1) or [])
    common = dict(
        palette=palette,
        propose_colour_gap=0.0,
        adjacency_strength=0.0,
        simulate_lighting=False,
        correct_light_cast=False,
        reconcile_exposure=False,
        self_calibrate=False,
        surface_only=False,
        fit_partial_shapes=False,
    )
    bad_inputs = (
        [(0, 0, 0)],
        {(0, 0): BlockState("minecraft:stone")},
        {(0, 0, 0): "minecraft:stone"},
    )

    for bad in bad_inputs:
        with pytest.raises((TypeError, ValueError)):
            raycast.reconstruct_scene_surfaces(
                {"a.png": _cutout_frame(20)}, {"a.png": _cutout_camera()},
                SceneTransform((0, 0, 0), (1, 1, 1)), sampling_states=bad, **common)
    assert calls == []


# ---------------------------------------------------------------------------
# DS-R29: optional, non-mutating stagewise visible-state loss instrumentation.
# The trace observes the pipeline; it never feeds it. These tests pin the
# observation-only contract (trace-on/off equality), JSON-safe bounds, the
# per-stage classification inputs and the fail-closed default.


def _uniform_reconstruct(image_value=125, *, palette, scene=None, camera=None, **kwargs):
    image = np.full((128, 128, 3), image_value, dtype=np.uint8)
    return raycast.reconstruct_scene_surfaces(
        {"a.png": image}, {"a.png": camera or CameraModel((4.0, 4.0, -10.0), 0.0, 0.0, eye_height=0.0)},
        scene or SceneTransform((0, 0, 0), (4, 4, 4)),
        palette=palette,
        correct_light_cast=False,
        reconcile_exposure=False,
        self_calibrate=False,
        simulate_lighting=False,
        surface_only=False,
        fit_partial_shapes=False,
        **kwargs,
    )


def _unreachable_palette():
    """A palette whose last state can never win a face, plus enough rivals to prune.

    With one camera each face votes once, so support pruning keeps the winner
    outright: the candidates that win nothing fall below the two-percent share
    and are dropped, while the six-or-more floor keeps the rest.
    """
    return (
        PaletteColor(BlockState("minecraft:stone"), (125, 125, 125)),
        PaletteColor(BlockState("minecraft:dirt"), (134, 96, 67)),
        PaletteColor(BlockState("minecraft:sand"), (219, 207, 163)),
        PaletteColor(BlockState("minecraft:glass"), (170, 210, 220)),
        PaletteColor(BlockState("minecraft:bricks"), (150, 75, 65)),
        PaletteColor(BlockState("minecraft:oak_planks"), (162, 130, 79)),
        PaletteColor(BlockState("minecraft:unreachable"), (255, 0, 255)),
    )


def test_stage_trace_does_not_change_reconstruction_output_or_diagnostics():
    """Trace-on must be decision-identical to trace-off, including diagnostics."""
    palette = _unreachable_palette()
    baseline_diagnostics: dict[str, object] = {}
    traced_diagnostics: dict[str, object] = {}
    trace: dict[str, object] = {}

    baseline = _uniform_reconstruct(palette=palette, diagnostics=baseline_diagnostics)
    traced = _uniform_reconstruct(
        palette=palette, diagnostics=traced_diagnostics, stage_trace=trace)

    assert _volume_signature(traced) == _volume_signature(baseline)
    assert baseline_diagnostics == traced_diagnostics
    # The trace is populated by that same run, not by a second reconstruction.
    assert trace["coordinates"]
    assert trace["coordinate_evidence"]


def test_stage_trace_is_json_safe_and_bounded():
    """The whole trace survives ``json.dumps`` unchanged, with the states intact."""
    import json

    palette = _unreachable_palette()
    trace: dict[str, object] = {}
    _uniform_reconstruct(palette=palette, stage_trace=trace)

    decoded = json.loads(json.dumps(trace, sort_keys=True))
    assert decoded == trace
    assert trace["schema"] == 2
    assert trace["candidate_limit"] == raycast.STAGE_TRACE_CANDIDATE_LIMIT

    # Canonical palette availability: every state is recorded, including the one
    # that never wins a face and is dropped by support pruning.
    palette_names = {state["name"] for state in trace["palette"]["states"]}
    assert "minecraft:unreachable" in palette_names
    assert "minecraft:stone" in palette_names
    # ``dark_oak_log``-style axis states must keep their exact properties.
    assert all(isinstance(state["properties"], list) for state in trace["palette"]["states"])


def test_stage_trace_support_pruning_membership_is_per_round_and_per_state():
    palette = _unreachable_palette()
    trace: dict[str, object] = {}
    _uniform_reconstruct(palette=palette, stage_trace=trace)

    rounds = trace["support_rounds"]
    assert len(rounds) == 2
    first = rounds[0]
    assert first["round"] == 0 and first["support_prior_applied"] is False
    assert first["candidates_before"] > first["candidates_after"]
    dropped = {state["name"] for state in first["states_dropped"]}
    kept = {state["name"] for state in first["states_kept"]}
    before = {state["name"] for state in first["states_before"]}
    # Every pre-round state is accounted for exactly once, the counts agree with
    # the recorded sets, and the states that won faces survive pruning.
    assert dropped.isdisjoint(kept)
    assert kept | dropped == before
    assert len(kept) == first["candidates_after"]
    assert len(before) == first["candidates_before"]
    assert "minecraft:stone" in kept
    assert "minecraft:stone" not in dropped
    assert len(kept) >= raycast.PALETTE_MINIMUM

    # The final palette is exactly the last round's survivors.
    final_names = {state["name"] for state in trace["final_palette"]["states"]}
    assert final_names == kept
    for coordinate in trace["coordinates"]:
        markers = coordinate["support"]
        assert [marker["round"] for marker in markers] == [0, 1]
        # Per-coordinate provenance partitions *this cell's own* ranked
        # candidates by each round's global membership; it never records a
        # privileged current winner, so a reference state that survived pruning
        # but lost the ranking is still visible here.
        names = {candidate["state"]["name"] for candidate in coordinate["top_candidates"]}
        for marker in markers:
            kept_now = set(marker["candidate_states_kept"])
            dropped_now = set(marker["candidate_states_dropped"])
            assert kept_now.isdisjoint(dropped_now)
            assert kept_now | dropped_now == names
        assert set(markers[-1]["candidate_states_kept"]) == names & final_names
        assert set(markers[0]["candidate_states_kept"]) == names & kept


def test_stage_trace_records_evidence_provenance_and_candidate_ranking():
    palette = _unreachable_palette()
    trace: dict[str, object] = {}
    _uniform_reconstruct(palette=palette, stage_trace=trace)

    evidence = {tuple(entry["coordinate"]): entry for entry in trace["coordinate_evidence"]}
    assert evidence
    sample = next(entry for entry in evidence.values() if entry["resolved_samples"])
    assert sample["has_resolved_face"] is True
    assert sample["observation_count"] == sample["resolved_samples"] + sample["fallback_samples"]
    assert sample["observation_count"] == 1
    assert sample["frames"] == ["a.png"]
    assert set(sample["slots"]) == set(sample["slots_resolved"]) | set(sample["slots_fallback"])
    assert set(sample["slots_resolved"]) <= set(sample["slots"])
    assert sample["pattern_samples"] == 0  # a rectified centre patch carries no pattern
    assert sample["resolved_area"] > 0.0 and sample["total_weight"] > 0.0
    assert len(sample["mean_rgb"]) == 3

    coordinates = {tuple(entry["coordinate"]): entry for entry in trace["coordinates"]}
    assert set(coordinates) == set(evidence)
    entry = next(values for values in coordinates.values() if values["raw_winner"])
    candidates = entry["top_candidates"]
    assert 0 < len(candidates) <= trace["candidate_limit"]
    assert entry["candidate_count"] == len(candidates)
    # Raw pre-smoothing ranking is best-first by ascending mean cost, and the
    # recorded winner is that ranking's head.
    costs = [candidate["cost"] for candidate in candidates]
    assert costs == sorted(costs)
    assert entry["raw_winner"] == candidates[0]["state"]
    assert entry["post_smoothing_label"] is not None
    assert entry["selected_state"] is not None
    assert entry["final_state"] == entry["selected_state"]


def test_stage_trace_absent_by_default_and_rejects_a_bad_candidate_limit():
    palette = _unreachable_palette()
    volume = _uniform_reconstruct(palette=palette)
    assert volume.blocks  # no trace requested, no trace work done

    with pytest.raises(ValueError):
        _uniform_reconstruct(palette=palette, stage_trace={}, stage_trace_candidate_limit=0)


# DS-R36: the trace-only candidate depth. Raising the limit records more of the
# cell's already-computed cheapest-first order; it must never change the volume,
# the diagnostics, the production winner or the ranking itself.


def _candidate_depth_palette(rivals=32):
    """One winning state plus enough distinct losing states to exceed the default depth.

    Every state is a flat colour, so a uniform image keeps one winner while the
    rest of the palette still has a defined mean cost order for that cell. The
    support-pruning ablation is what keeps the losers in the candidate universe;
    without it they are pruned away before ranking and the depth is bounded by
    the surviving six-state floor.
    """
    return (PaletteColor(BlockState("minecraft:stone"), (125, 125, 125)),) + tuple(
        PaletteColor(BlockState(f"minecraft:rival_{index}"), (index + 1, 0, 0))
        for index in range(rivals)
    )


def _deep_candidate_reconstruct(rivals=32, **kwargs):
    """Reconstruct a deep palette with support pruning disabled for this call."""
    return _uniform_reconstruct(
        palette=_candidate_depth_palette(rivals), palette_support=0.0, **kwargs)


def test_raised_stage_trace_candidate_limit_deepens_the_ranked_list_only():
    deep: dict[str, object] = {}
    baseline = _deep_candidate_reconstruct()
    deep_volume = _deep_candidate_reconstruct(stage_trace=deep, stage_trace_candidate_limit=32)

    assert _volume_signature(deep_volume) == _volume_signature(baseline)
    assert deep["candidate_limit"] == 32
    entry = next(
        values for values in deep["coordinates"] if values.get("candidate_count", 0) > 1)
    # Production's top-K map is untouched at 8; only the trace-only deeper list grew.
    assert len(entry["top_candidates"]) == 8
    assert len(entry["ranked_candidates"]) == 32
    # The trace reports the depth it reached (32 here) and, conservatively, that
    # the ranking may continue: the 33-state palette has one state left over.
    assert entry["candidate_count"] == 32
    assert entry["top_k_truncated"] is True
    # The order is the same cheapest-first ranking at any depth: only its length moved.
    costs = [candidate["cost"] for candidate in entry["ranked_candidates"]]
    assert costs == sorted(costs)
    assert entry["raw_winner"] == entry["top_candidates"][0]["state"]
    assert entry["raw_winner"] == entry["ranked_candidates"][0]["state"]


def test_stage_trace_candidate_limit_flags_a_truncated_ranking():
    trace: dict[str, object] = {}
    _deep_candidate_reconstruct(rivals=60, stage_trace=trace, stage_trace_candidate_limit=3)

    entry = next(
        values for values in trace["coordinates"] if values.get("candidate_count", 0) > 1)
    assert len(entry["top_candidates"]) == 8
    assert len(entry["ranked_candidates"]) == 3
    assert entry["candidate_count"] == 3
    assert entry["top_k_truncated"] is True


@pytest.mark.parametrize("limit", [8, 16, 32])
def test_stage_trace_candidate_limit_leaves_output_and_winner_identical(limit):
    baseline_diagnostics: dict[str, object] = {}
    traced_diagnostics: dict[str, object] = {}
    trace: dict[str, object] = {}

    baseline = _deep_candidate_reconstruct(diagnostics=baseline_diagnostics)
    traced = _deep_candidate_reconstruct(
        diagnostics=traced_diagnostics, stage_trace=trace,
        stage_trace_candidate_limit=limit)

    assert _volume_signature(traced) == _volume_signature(baseline)
    assert traced_diagnostics == baseline_diagnostics
    # A deeper trace never reorders the ranking whose head the pipeline places.
    for entry in trace["coordinates"]:
        if not entry.get("raw_winner"):
            continue
        assert len(entry["top_candidates"]) == 8
        assert entry["raw_winner"] == entry["top_candidates"][0]["state"]
        assert entry["raw_winner"] == entry["ranked_candidates"][0]["state"]


# ---------------------------------------------------------------------------
# E2a: the schema-2 observation-level trace. It is additive and diagnostic: the
# schema-1 keys keep their exact meaning, the new records are read back by
# nothing in production, and replaying the stored cost vectors and weights must
# reproduce the aggregation and the production top-8 winners exactly.


def _states_from_records(records):
    return tuple(
        BlockState(record["name"], tuple(tuple(pair) for pair in record["properties"]))
        for record in records
    )


def _replay_accumulation(cost_records):
    """Rebuild ``_accumulate_costs`` from the stored vectors and weights."""
    totals: dict[tuple[int, int, int], np.ndarray] = {}
    weights: dict[tuple[int, int, int], float] = {}
    for record in sorted(cost_records, key=lambda item: item["ordinal"]):
        effective = record["effective_weight"]
        if effective <= 0.0:
            continue
        coordinate = tuple(record["coordinate"])
        costs = np.asarray(record["costs"], dtype=np.float64)
        if coordinate in totals:
            totals[coordinate] = totals[coordinate] + costs * effective
        else:
            totals[coordinate] = costs * effective
        weights[coordinate] = weights.get(coordinate, 0.0) + effective
    return totals, weights


def test_stage_trace_v2_schema_and_keys_are_additive():
    """Schema 1 keys survive unchanged; only the number and v2 keys are added."""
    trace: dict[str, object] = {}
    _uniform_reconstruct(palette=_unreachable_palette(), stage_trace=trace)

    schema_v1_keys = {
        "schema", "candidate_limit", "palette", "support_rounds",
        "final_palette", "coordinates", "coordinate_evidence",
    }
    schema_v2_keys = {"observations", "observation_costs", "aggregation"}
    assert trace["schema"] == 2
    assert set(trace) == schema_v1_keys | schema_v2_keys
    # The v1 truth tables are still present in their original shape: a
    # coordinate entry keeps its ranking and its stage labels.
    for entry in trace["coordinates"]:
        assert "coordinate" in entry
        assert "final_state" in entry
        if entry.get("raw_winner"):
            assert "top_candidates" in entry and "ranked_candidates" in entry
    assert trace["final_palette"]["states"]
    assert trace["coordinate_evidence"]


def test_stage_trace_v2_observations_and_costs_are_ordinal_aligned():
    """One final cost record per retained raw observation, same identity."""
    palette = _unreachable_palette()
    trace: dict[str, object] = {}
    _uniform_reconstruct(palette=palette, stage_trace=trace)

    observations = trace["observations"]
    costs = trace["observation_costs"]
    retained = sum(entry["observation_count"] for entry in trace["coordinate_evidence"])
    assert len(observations) == retained == len(costs)
    assert [record["ordinal"] for record in observations] == list(range(retained))
    assert [record["ordinal"] for record in costs] == list(range(retained))

    for observation, cost in zip(observations, costs):
        assert cost["ordinal"] == observation["ordinal"]
        assert cost["frame_id"] == observation["frame_id"]
        assert cost["coordinate"] == observation["coordinate"]
        assert cost["slot"] == observation["slot"]
        assert cost["source"] == observation["source"]
        assert observation["source"] in {"resolved_face", "centre_patch_fallback"}
        # The vector is interpretable on its own: the current palette state
        # order is recorded and the vector matches its length.
        assert cost["palette_order"] == trace["final_palette"]["states"]
        assert len(cost["costs"]) == trace["final_palette"]["size"]
        assert cost["best_cost"] >= 0.0
        assert cost["weight"] >= 0.0
        assert cost["effective_weight"] == pytest.approx(
            cost["weight"] * cost["view_weight"], rel=1e-12, abs=1e-12)

    resolved = [record for record in observations if record["source"] == "resolved_face"]
    fallbacks = [
        record for record in observations if record["source"] == "centre_patch_fallback"
    ]
    assert resolved and fallbacks
    for record in resolved:
        assert record["area"] > 0.0
        assert record["depth"] > 0.0
        assert record["visible"] is not None and record["facing"] is not None
        assert record["observed_patch"] is not None
        assert record["valid"] is not None
        assert record["coverage"] is not None
    for record in fallbacks:
        # A centre patch has no rectified patch, mask or coverage, so those
        # digests are explicit nulls rather than digests of nothing.
        assert record["area"] == 0.0
        assert record["visible"] is None and record["facing"] is None
        assert record["observed_patch"] is None
        assert record["valid"] is None
        assert record["coverage"] is None
        assert record["pattern"] is None
        assert record["rgb"] is not None and record["descriptor"] is not None


def test_array_digest_binds_shape_dtype_and_bytes_without_mutating_inputs():
    """The digest is deterministic, shape/dtype bound, and never mutates."""
    import hashlib

    array = np.arange(24, dtype=np.float32).reshape(4, 6)
    original = array.copy()
    expected = hashlib.sha256(
        f"{array.shape}|{array.dtype.str}|".encode("ascii") + array.tobytes()
    ).hexdigest()
    assert raycast._array_digest(array) == expected
    assert raycast._array_digest(array) == raycast._array_digest(array)
    assert np.array_equal(array, original) and array.dtype == original.dtype

    # Same bytes, different shape or dtype, different digest.
    assert raycast._array_digest(array) != raycast._array_digest(array.reshape(6, 4))
    assert raycast._array_digest(array) != raycast._array_digest(array.astype(np.float64))

    # A non-contiguous view digests like its contiguous copy and leaves both the
    # view and its source untouched.
    view = array[:, ::2]
    view_before = view.copy()
    assert raycast._array_digest(view) == raycast._array_digest(np.ascontiguousarray(view))
    assert np.array_equal(view, view_before)
    assert np.array_equal(array, original)
    assert raycast._array_digest(None) is None


def test_stage_trace_v2_digests_are_deterministic_across_runs():
    first: dict[str, object] = {}
    second: dict[str, object] = {}
    palette = _unreachable_palette()
    _uniform_reconstruct(palette=palette, stage_trace=first)
    _uniform_reconstruct(palette=palette, stage_trace=second)

    assert first["observations"] == second["observations"]
    assert first["observation_costs"] == second["observation_costs"]
    assert first["aggregation"] == second["aggregation"]
    # Every digest is either an explicit null or a 64-character hex string.
    for record in first["observations"]:
        for key in ("observed_patch", "valid", "coverage", "rgb", "descriptor", "pattern"):
            value = record[key]
            assert value is None or (
                isinstance(value, str) and len(value) == 64 and value == value.lower()
            )


def test_stage_trace_v2_replay_reproduces_aggregation_and_top8_winners():
    """Stored vectors and weights replay aggregation and the raw top-8 map."""
    import types

    trace: dict[str, object] = {}
    _deep_candidate_reconstruct(stage_trace=trace, stage_trace_candidate_limit=32)
    assert trace["observation_costs"]
    # Clipped-cost ties are real in this fixture: the outlier ceiling is hit.
    assert any(
        abs(value - raycast.OUTLIER_CEILING) < 1e-9
        for record in trace["observation_costs"]
        for value in record["costs"]
    )

    totals, weights = _replay_accumulation(trace["observation_costs"])
    aggregation = {tuple(entry["coordinate"]): entry for entry in trace["aggregation"]}
    assert set(aggregation) == set(totals)
    for coordinate, entry in aggregation.items():
        assert entry["weight"] == pytest.approx(
            weights[coordinate], rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(
            np.asarray(entry["total"], dtype=np.float64), totals[coordinate],
            rtol=1e-12, atol=1e-12)

    index = types.SimpleNamespace(
        states=_states_from_records(trace["final_palette"]["states"]))
    coordinates = {tuple(entry["coordinate"]): entry for entry in trace["coordinates"]}
    checked = 0
    for coordinate, entry in coordinates.items():
        if not entry.get("raw_winner") or coordinate not in totals:
            continue
        mean = totals[coordinate] / max(weights[coordinate], 1e-6)
        scores = raycast._top_candidates(mean, index)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0].name))
        replayed = [
            {"state": raycast._state_record(state), "cost": -float(cost)}
            for state, cost in ranked
        ]
        # Same argpartition result at K=8, in the same ranked order.
        assert replayed == entry["top_candidates"]
        assert replayed[0]["state"] == entry["raw_winner"]
        checked += 1
    assert checked


def test_stage_trace_v2_survives_json_round_trip():
    import json

    trace: dict[str, object] = {}
    _deep_candidate_reconstruct(stage_trace=trace, stage_trace_candidate_limit=32)
    decoded = json.loads(json.dumps(trace, sort_keys=True))
    assert decoded == trace
    assert decoded["schema"] == 2
    assert decoded["observations"] == trace["observations"]
    assert decoded["observation_costs"] == trace["observation_costs"]
    assert decoded["aggregation"] == trace["aggregation"]


def test_stage_trace_v2_cost_records_carry_the_effective_view_weight():
    """A per-frame trust below one is recorded and replays aggregation exactly."""
    import types

    index = types.SimpleNamespace(
        states=(BlockState("minecraft:stone"), BlockState("minecraft:dirt")))
    observations = [
        ("a.png", (0, 0, 0), np.asarray([0.0, 24.0]), 2.0),
        ("b.png", (0, 0, 0), np.asarray([1.0, 4.0]), 3.0),
        ("a.png", (1, 0, 0), np.asarray([5.0, 0.0]), 1.0),
    ]
    observation_records = [
        {"slot": "north", "source": "resolved_face"},
        {"slot": "north", "source": "centre_patch_fallback"},
        {"slot": "up", "source": "resolved_face"},
    ]
    view_weights = {"a.png": 0.5, "b.png": 1.0}
    records = raycast._observation_cost_records(
        observations, index, view_weights, observation_records, [0.5, 0.25, 1.5])

    assert [record["weight"] for record in records] == [2.0, 3.0, 1.0]
    assert [record["view_weight"] for record in records] == [0.5, 1.0, 0.5]
    assert [record["effective_weight"] for record in records] == [1.0, 3.0, 0.5]
    assert [record["best_cost"] for record in records] == [0.5, 0.25, 1.5]
    assert records[0]["palette_order"] == [
        {"name": "minecraft:stone", "properties": []},
        {"name": "minecraft:dirt", "properties": []},
    ]

    totals, weights = raycast._accumulate_costs(observations, view_weights)
    assert raycast._aggregation_records(totals, weights) == [
        {
            "coordinate": [0, 0, 0],
            "total": [3.0, 36.0],
            "weight": 4.0,
        },
        {
            "coordinate": [1, 0, 0],
            "total": [2.5, 0.0],
            "weight": 0.5,
        },
    ]
    replayed_totals, replayed_weights = _replay_accumulation(records)
    for coordinate, total in totals.items():
        np.testing.assert_allclose(
            replayed_totals[coordinate], total, rtol=1e-12, atol=1e-12)
        assert replayed_weights[coordinate] == pytest.approx(
            weights[coordinate], rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# E2c: the opt-in, bounded resolved-face payload sidecar. It is separate from the
# E2A trace, carries the sampler's own patch/valid/coverage bytes, mirrors the
# trace's observation ordinals and digests, and is read back by no decision.


def _decode_sidecar_array(record):
    """Restore one sidecar array payload from its self-describing record."""
    import base64

    dtype = np.dtype(record["dtype"])
    shape = tuple(int(size) for size in record["shape"])
    raw = base64.b64decode(record["base64"], validate=True)
    assert len(raw) == int(np.prod(shape)) * dtype.itemsize
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def test_stage_sidecar_requires_the_e2a_trace():
    with pytest.raises(ValueError, match="stage_sidecar requires stage_trace"):
        _uniform_reconstruct(palette=_unreachable_palette(), stage_sidecar={})


def test_stage_sidecar_carries_only_resolved_faces_and_mirrors_the_trace():
    trace: dict[str, object] = {}
    sidecar: dict[str, object] = {}
    _uniform_reconstruct(
        palette=_unreachable_palette(), stage_trace=trace, stage_sidecar=sidecar)

    observations = trace["observations"]
    by_ordinal = {record["ordinal"]: record for record in observations}
    resolved = [record for record in observations if record["source"] == "resolved_face"]
    fallbacks = [
        record for record in observations if record["source"] == "centre_patch_fallback"
    ]
    records = sidecar["records"]

    assert sidecar["schema"] == raycast.STAGE_SIDECAR_SCHEMA
    assert sidecar["trace_schema"] == 2
    assert sidecar["observation_count"] == len(observations)
    assert sidecar["resolved_face_records"] == len(records) == len(resolved)
    assert sidecar["excluded_fallback_observations"] == len(fallbacks)
    # The fixture genuinely exercises both branches, so the exclusion is real.
    assert resolved and fallbacks
    assert all(record["source"] == "resolved_face" for record in records)
    assert len(records) == len(observations) - len(fallbacks)

    for record in records:
        observation = by_ordinal[record["ordinal"]]
        assert observation["source"] == "resolved_face"
        assert record["frame_id"] == observation["frame_id"]
        assert record["coordinate"] == observation["coordinate"]
        assert record["slot"] == observation["slot"]
        # The observed-patch/mask digests equal the E2A observation entry's, so the
        # scorer-facing payload parity can be checked without decoding anything.
        digests = record["digests"]
        assert set(digests) == {"raw_patch", "observed_patch", "valid", "coverage"}
        assert digests["observed_patch"] == observation["observed_patch"]
        assert digests["valid"] == observation["valid"]
        assert digests["coverage"] == observation["coverage"]
        # The raw sampler patch is carried beside the observed patch and has its
        # own shape/dtype-bound digest; it never substitutes for the trace digest.
        assert record["raw_patch"]["shape"] == record["observed_patch"]["shape"]
        assert record["raw_patch"]["dtype"] == record["observed_patch"]["dtype"]
        assert digests["raw_patch"] == raycast._array_digest(
            _decode_sidecar_array(record["raw_patch"]))
    # Fallback observations never leak a face payload into the sidecar.
    fallback_ordinals = {record["ordinal"] for record in fallbacks}
    assert fallback_ordinals.isdisjoint({record["ordinal"] for record in records})


def test_stage_sidecar_round_trips_exact_arrays_and_digest_parity():
    import json

    trace: dict[str, object] = {}
    sidecar: dict[str, object] = {}
    _uniform_reconstruct(
        palette=_unreachable_palette(), stage_trace=trace, stage_sidecar=sidecar)

    decoded = json.loads(json.dumps(sidecar, sort_keys=True))
    assert decoded == sidecar
    records = decoded["records"]
    assert records
    for record in records:
        raw_patch = _decode_sidecar_array(record["raw_patch"])
        patch = _decode_sidecar_array(record["observed_patch"])
        valid = _decode_sidecar_array(record["valid"])
        coverage = _decode_sidecar_array(record["coverage"])
        # Exact lossless domain: float32 raw/observed patch and coverage plus a
        # bool validity mask - the raw and observed arrays share that domain.
        assert raw_patch.dtype == np.float32
        assert patch.dtype == np.float32
        assert valid.dtype == np.bool_
        assert coverage.dtype == np.float32
        assert raw_patch.shape == (raycast.PATCH_SIZE, raycast.PATCH_SIZE, 3)
        assert patch.shape == (raycast.PATCH_SIZE, raycast.PATCH_SIZE, 3)
        assert valid.shape == (raycast.PATCH_SIZE, raycast.PATCH_SIZE)
        assert coverage.shape == (raycast.PATCH_SIZE, raycast.PATCH_SIZE)
        # Re-deriving the production digest from the decoded bytes matches both
        # the sidecar's own digest block and the E2A trace entry. The raw patch
        # digest is re-derived the same way; it has no trace counterpart.
        observation = trace["observations"][record["ordinal"]]
        assert raycast._array_digest(raw_patch) == record["digests"]["raw_patch"]
        assert raycast._array_digest(patch) == record["digests"]["observed_patch"]
        assert raycast._array_digest(valid) == record["digests"]["valid"]
        assert raycast._array_digest(coverage) == record["digests"]["coverage"]
        assert record["digests"]["observed_patch"] == observation["observed_patch"]
        assert record["digests"]["valid"] == observation["valid"]
        assert record["digests"]["coverage"] == observation["coverage"]
    # No reference label or evaluator truth can appear in a payload record.
    forbidden = {"truth", "label", "truth_state", "reference"}
    assert all(forbidden.isdisjoint(record) for record in records)


def test_stage_sidecar_raw_and_observed_patch_come_from_the_same_sample(monkeypatch):
    """The raw sampler patch and the scorer-facing observed patch are both real.

    A synthetic ``FaceSample`` whose mask invalidates a few pixels is the only
    deterministic way to make the two arrays differ: for an all-valid patch the
    observed property returns the raw array unchanged. The sidecar must carry the
    raw array E1 parity needs *and* the neutralised array the trace digests, each
    losslessly, so the test asserts the exact arrays rather than only their
    digests.
    """
    import base64

    raw = np.arange(raycast.PATCH_SIZE * raycast.PATCH_SIZE * 3, dtype=np.float32)
    raw = (raw * 0.25 + 7.0).reshape(raycast.PATCH_SIZE, raycast.PATCH_SIZE, 3)
    valid = np.ones(raw.shape[:2], dtype=bool)
    valid[0, 0] = False
    valid[1, 2] = False
    coverage = np.ones(raw.shape[:2], dtype=np.float32)
    coverage[0, 0] = 0.25
    sample = FaceSample(
        (0, 0, 0), "south", raw.copy(), 64.0, 4.0,
        visible=1.0, facing=1.0, valid=valid, coverage=coverage)

    monkeypatch.setattr(raycast, "sample_visible_faces", lambda *a, **k: [sample])
    trace: dict[str, object] = {}
    sidecar: dict[str, object] = {}
    # The synthetic patch is a gradient, so the uniformity gate is widened for
    # this test only; the gate is not what this test exercises.
    _uniform_reconstruct(
        palette=_unreachable_palette(), stage_trace=trace, stage_sidecar=sidecar,
        max_face_variation=1e9)

    records = sidecar["records"]
    assert len(records) == 1
    record = records[0]
    assert record["source"] == "resolved_face"
    # Decode through the payload's own dtype/shape so the round trip is bound to
    # the record, not to the arrays the test happens to know.
    def decode(payload):
        dtype = np.dtype(payload["dtype"])
        shape = tuple(int(size) for size in payload["shape"])
        return np.frombuffer(
            base64.b64decode(payload["base64"], validate=True), dtype=dtype
        ).reshape(shape).copy()

    # ``np.array_equal`` is exact, so this proves byte-level reproduction.
    np.testing.assert_array_equal(decode(record["raw_patch"]), sample.patch)
    expected_observed = np.array(raw, copy=True)
    expected_observed[~valid] = sample.rgb
    np.testing.assert_array_equal(decode(record["observed_patch"]), expected_observed)
    np.testing.assert_array_equal(decode(record["valid"]), valid)
    np.testing.assert_array_equal(decode(record["coverage"]), coverage)
    assert record["digests"]["raw_patch"] == raycast._array_digest(sample.patch)
    assert record["digests"]["observed_patch"] == raycast._array_digest(expected_observed)
    # The fixture genuinely distinguishes the two arrays, so a sidecar that stored
    # one twice would fail the exact comparisons above.
    assert not np.array_equal(raw, expected_observed)
    assert record["digests"]["raw_patch"] != record["digests"]["observed_patch"]
    assert record["raw_patch"]["base64"] != record["observed_patch"]["base64"]
    # ``observed_patch`` remains tied to the E2A trace observation, while the new
    # raw field is sidecar-only and has no trace counterpart.
    observation = trace["observations"][record["ordinal"]]
    assert record["digests"]["observed_patch"] == observation["observed_patch"]
    assert "raw_patch" not in observation


def test_stage_sidecar_raw_patch_matches_the_real_sampler_iteration(monkeypatch):
    """Source-array fidelity on the real sampler path, not a synthetic fixture.

    The sampler is wrapped (never replaced) so the test can capture every real
    ``FaceSample`` it returns. The sidecar's resolved-face records are then paired
    with the samples by their unique ``(coordinate, slot)`` identity - the same
    key ``seen_here`` deduplicates on - and each decoded raw patch is required to
    equal the captured ``sample.patch`` exactly. That is the E1 parity claim, and
    an implementation that fell back to ``observed_patch`` for the raw field would
    fail whenever a retained sample carries an invalid pixel.
    """
    import base64

    real_sampler = raycast.sample_visible_faces
    captured = []

    def capturing_sampler(*args, **kwargs):
        found = list(real_sampler(*args, **kwargs))
        captured.extend(found)
        return found

    monkeypatch.setattr(raycast, "sample_visible_faces", capturing_sampler)
    trace: dict[str, object] = {}
    sidecar: dict[str, object] = {}
    _uniform_reconstruct(
        palette=_unreachable_palette(), stage_trace=trace, stage_sidecar=sidecar)

    # The sidecar keeps exactly the first retained sample per coordinate/slot.
    unique = []
    seen = set()
    for item in captured:
        key = (tuple(item.coordinate), item.slot)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    records = sidecar["records"]
    assert records
    assert len(records) == len(unique) == len({
        (tuple(record["coordinate"]), record["slot"]) for record in records
    })
    by_key = {(tuple(item.coordinate), item.slot): item for item in unique}
    for record in records:
        sample = by_key[(tuple(record["coordinate"]), record["slot"])]
        payload = record["raw_patch"]
        decoded = np.frombuffer(
            base64.b64decode(payload["base64"], validate=True),
            dtype=np.dtype(payload["dtype"]),
        ).reshape(tuple(int(size) for size in payload["shape"])).copy()
        np.testing.assert_array_equal(decoded, sample.patch)
        assert record["digests"]["raw_patch"] == raycast._array_digest(sample.patch)
        assert record["digests"]["observed_patch"] == raycast._array_digest(
            sample.observed_patch)


def test_stage_sidecar_leaves_output_diagnostics_and_trace_identical():
    """Sidecar-on must be decision- and trace-identical to sidecar-off."""
    import json

    palette = _unreachable_palette()
    baseline_diagnostics: dict[str, object] = {}
    sidecar_diagnostics: dict[str, object] = {}
    baseline_trace: dict[str, object] = {}
    sidecar_trace: dict[str, object] = {}

    baseline = _uniform_reconstruct(
        palette=palette, diagnostics=baseline_diagnostics, stage_trace=baseline_trace)
    sidecar_run = _uniform_reconstruct(
        palette=palette, diagnostics=sidecar_diagnostics, stage_trace=sidecar_trace,
        stage_sidecar={})

    assert _volume_signature(sidecar_run) == _volume_signature(baseline)
    assert sidecar_diagnostics == baseline_diagnostics
    assert json.dumps(sidecar_trace, sort_keys=True) == json.dumps(
        baseline_trace, sort_keys=True)
    assert sidecar_trace == baseline_trace


def test_stage_sidecar_is_not_requested_by_default():
    # No sidecar mapping, no payload work: an ordinary traced run still produces
    # the unchanged v2 trace and reconstruction.
    trace: dict[str, object] = {}
    volume = _uniform_reconstruct(palette=_unreachable_palette(), stage_trace=trace)

    assert volume.blocks
    assert trace["schema"] == 2
    assert trace["observations"]
    # The sidecar lives outside the trace and is absent unless asked for.
    assert "records" not in trace
    assert "stage_sidecar" not in trace


def test_default_path_never_runs_the_sidecar_record_helper(monkeypatch):
    """Without an opt-in, no sidecar payload helper can even be reached.

    ``sidecar_records`` is ``None`` - not an empty list - on the ordinary traced
    path, so the capture branch never runs. Making ``_sidecar_record`` explode
    turns any accidental call into a hard failure and pins the no-allocation,
    no-sidecar-work contract for the default path.
    """
    def unreachable(**_kwargs):
        raise AssertionError("_sidecar_record ran without a sidecar opt-in")

    monkeypatch.setattr(raycast, "_sidecar_record", unreachable)
    trace: dict[str, object] = {}

    volume = _uniform_reconstruct(palette=_unreachable_palette(), stage_trace=trace)

    assert volume.blocks
    assert trace["observations"]

    # Positive control: the same exploding replacement is reachable once a
    # sidecar is opted in, so the default-path pass above genuinely proved the
    # helper was skipped rather than the replacement simply being inert.
    with pytest.raises(AssertionError, match="without a sidecar opt-in"):
        _uniform_reconstruct(
            palette=_unreachable_palette(), stage_trace={}, stage_sidecar={})


def test_stage_sidecar_enforces_the_record_bound_at_capture(monkeypatch):
    """The fixed record cap is enforced while records are built, not after.

    A baseline run fixes how many resolved faces this fixture produces. Capping
    one below that must refuse on the record that would exceed the bound, before
    its arrays are copied, digested or encoded; ``_sidecar_record`` therefore runs
    exactly ``cap`` times. The caller-owned sidecar mapping stays empty, so no
    truncated sidecar can be emitted.
    """
    baseline_trace: dict[str, object] = {}
    _uniform_reconstruct(palette=_unreachable_palette(), stage_trace=baseline_trace)
    resolved = [
        record for record in baseline_trace["observations"]
        if record["source"] == "resolved_face"
    ]
    assert len(resolved) >= 2

    constructed = {"count": 0}
    real_record = raycast._sidecar_record

    def counting_record(**kwargs):
        constructed["count"] += 1
        return real_record(**kwargs)

    monkeypatch.setattr(raycast, "_sidecar_record", counting_record)
    monkeypatch.setattr(raycast, "STAGE_SIDECAR_MAX_RECORDS", len(resolved) - 1)

    trace: dict[str, object] = {}
    sidecar: dict[str, object] = {}
    with pytest.raises(ValueError, match="record bound"):
        _uniform_reconstruct(
            palette=_unreachable_palette(), stage_trace=trace, stage_sidecar=sidecar)

    assert constructed["count"] == len(resolved) - 1
    assert sidecar == {}
