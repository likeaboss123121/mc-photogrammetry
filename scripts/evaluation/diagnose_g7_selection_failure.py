"""Observe one default delivery's G7 result; never replay or change selection."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import struct
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mc_photogrammetry.pipeline import image_derived_geometry_orchestration as orchestration
from mc_photogrammetry.pipeline import image_derived_joint_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice_measure
from mc_photogrammetry.pipeline.image_derived_production_delivery import derive_and_deliver_image_output
from mc_photogrammetry.pipeline import image_derived_graph as legacy
from mc_photogrammetry.reconstruction import plane_graph, endpoint_hypothesis as csp, region_growth

SCHEMA = "mc-photogrammetry-g7-selection-diagnostic-v1"
PRODUCER = "g7-selection-diagnostic-1"
MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_PROOF_BYTES = 4 * 1024 * 1024
MAX_ITEMS = 524288
MAX_DEPTH = 32
SOURCE_PINS = {
    "src/mc_photogrammetry/pipeline/image_derived_geometry_orchestration.py": "cbd2d8e2e8e744799628159b4eb307207024435272441d4b90215bb6f10add1e",
    "src/mc_photogrammetry/pipeline/image_derived_joint_graph.py": "535c1010aaf41bfad052e27d691038ac44e6bcf92ea98f61c73ac0a3a38bd878",
    "src/mc_photogrammetry/pipeline/image_derived_endpoint_selection.py": "6d4a59dd7725fd50a1bc8f85646939907cdbb2a361462fc5f652e8e093b91e1e",
    "src/mc_photogrammetry/reconstruction/endpoint_hypothesis.py": "86f8ccfa1665eb523830dce79c8f3593ed4afdb1b1993966c8d82b2036db9b0f",
    "src/mc_photogrammetry/reconstruction/plane_graph.py": "916111dbaf0440fd40e752e38664477b8f0a0858cd501d8e2b570598f0d83073",
}


class DiagnosticError(RuntimeError):
    def __init__(self, reason, field=""):
        self.reason, self.field = reason, field
        super().__init__(f"{reason}: {field}")


ATTRITION_STAGES = ("seed", "proposal", "reading")
ATTRITION_ITEMS = 4096
ATTRITION_BYTES = 512 * 1024
ATTRITION_SCHEMA = "g7-reading-attrition-v2"
ATTRITION_ROW_SCHEMA = "g7-reading-attrition-row-v2"
SEED_COUNT_FIELDS = ("node_count", "side_match_count", "normal_score_pass_count",
    "margin_pass_count", "tangent_support_pass_count", "after_side_count",
    "after_score_count", "after_margin_count", "after_support_count")
SEED_MAX_FIELDS = ("max_side_normal_score", "max_side_margin", "max_side_tangent_support")
SUPPORT_SCHEMA = "g7-support-score-attrition-v1"
SUPPORT_ROW_SCHEMA = "g7-support-score-row-v1"
SUPPORT_NODES = 20000
SUPPORT_MASS = 1e12
SUPPORT_STATES = ("measured", "not_reached", "graph_unavailable", "failed", "missing")
PAIR_SCHEMA = "g7-thin-centerline-paired-support-v2"
UNIT_WEIGHT_SCHEMA = "g7-unit-weight-sparse-support-v1"
UNIT_WEIGHT_SCOPE = "unit_line_weight_only_same_sparse_ids_and_denominator"
PAIR_EVIDENCE_SCHEMA = "g7-post-clipping-evidence-census-v1"
PAIR_EVIDENCE_STAGE = "post_clipping_retained_evidence_pre_allocation"
PAIR_INVALID_SCOPE = "dense_centerline_invalid_label_or_usable_mask_samples"
PAIR_TARGETS = (("image-0000", 0, -1), ("image-0000", 2, -1), ("image-0000", 2, 1))
PAIR_BYTES = 96 * 1024
PAIR_TEMP_BYTES = 4 * 1024 * 1024
PAIR_ROW_BYTES = 2048


def _pair_transient_bound(nodes):
    # Source arrays are borrowed, fingerprints hash their buffers without copying.
    # Per-node ceiling covers retained raster/new scores (24B), row masks/unions
    # (16B), native score/gate arithmetic (48B), and the larger existing support
    # validation's selected rows/index/pattern temporaries (104B), conservatively
    # treated as simultaneous. Fixed allowance covers 56,640B allocator vectors,
    # bounded <=5000-row census checks, scalar/JSON records and Python bookkeeping.
    return nodes * 192 + 56_640 + 128 * 1024


def _pair_timed(function):
    def measured(self, *args, **kwargs):
        started, failed = time.monotonic(), True
        try:
            if self.incomplete:
                raise ValueError("paired observer disabled")
            result = function(self, *args, **kwargs)
            failed = False
            return result
        except BaseException:
            self.discard("callback_failure")
            raise
        finally:
            elapsed = time.monotonic() - started
            self.frame_seconds += elapsed
            self.total_seconds += elapsed
            if not failed and (self.frame_seconds > 30 or self.total_seconds > 180):
                self.discard("time_budget")
                raise ValueError("paired time budget")
    return measured


def _pair_partition(old, new, denominator=None):
    np = region_growth.np
    both = int(np.count_nonzero(old & new))
    old_only = int(np.count_nonzero(old)) - both
    new_only = int(np.count_nonzero(new)) - both
    n = len(old) if denominator is None else denominator
    return dict(both=both, old_only=old_only, new_only=new_only,
                neither=n-both-old_only-new_only)


def _pair_metrics(supports, scores, sides, axis, side, config):
    np = region_growth.np
    tangent = [column for column in range(3) if column != axis]
    side_mask = sides[:, axis] == side
    denominator = int(np.count_nonzero(side_mask))
    best = scores[:, axis]
    # Match actual _seed_and_grow dtype and operation order.
    other = np.max(np.delete(scores, axis, axis=1), axis=1)
    tangent_support = supports[:, tangent].sum(axis=1)
    raw, sequential, current = [], [], side_mask.copy()
    for values, threshold in ((best, config.min_seed_score),
                              (best-other, config.min_seed_margin),
                              (tangent_support, config.min_seed_support)):
        passed = values >= threshold
        raw.append(int(np.count_nonzero(passed & side_mask)))
        np.logical_and(current, passed, out=current)
        sequential.append(int(np.count_nonzero(current)))
    mass = [float(np.sum(supports[:, column], where=side_mask, dtype=np.float64))
            for column in range(3)]
    positive = [int(np.count_nonzero((supports[:, column] > 0) & side_mask))
                for column in range(3)]
    cooccurrence = int(np.count_nonzero((supports[:, tangent[0]] > 0)
                       & (supports[:, tangent[1]] > 0) & side_mask))
    return current, dict(mass=mass, positive=positive, cooccurrence=cooccurrence,
                         raw=raw, sequential=sequential, eligible=sequential[-1]), denominator


def _pair_witness_values(support, scores, axis, source_side, side, thresholds):
    """Use one-row arrays: native array comparisons cast thresholds as source does."""
    np = region_growth.np
    other = np.max(np.delete(scores, axis, axis=1), axis=1)
    margin = scores[:, axis] - other
    tangent = support[:, [a for a in range(3) if a != axis]].sum(axis=1)
    passes = [bool((value >= threshold)[0]) for value, threshold in
              zip((scores[:, axis], margin, tangent), thresholds)]
    return dict(support=[float(value) for value in support[0]],
        score=float(scores[0, axis]), margin=float(margin[0]), tangent=float(tangent[0]),
        side=int(source_side), passes=passes+[int(source_side) == side])


def _pair_witness(node, old, new, old_scores, new_scores, axis, side, sides, config):
    np = region_growth.np
    result = {}
    thresholds = (config.min_seed_score, config.min_seed_margin, config.min_seed_support)
    for key, support, scores in (("old", old, old_scores), ("new", new, new_scores)):
        if support.dtype != np.float32 or not np.array_equal(
                scores[node:node+1], region_growth._normal_scores(support[node:node+1])):
            raise ValueError("paired witness source score")
        result[key] = _pair_witness_values(support[node:node+1], scores[node:node+1],
                    axis, sides[node, axis], side, thresholds)
    if (all(result["old"]["passes"]) or not all(result["new"]["passes"])
            or result["old"]["support"] == result["new"]["support"]):
        raise ValueError("paired witness membership")
    return result


def _pair_project(frame, graph, raster, config):
    np = region_growth.np
    old, old_scores, sides = graph.supports, graph.plane_scores, graph.sides
    if _pair_transient_bound(len(old)) > PAIR_TEMP_BYTES:
        raise ValueError("paired transient budget")
    _support_score_rows(frame, graph)  # Existing domain validation, no replay.
    if (raster.shape != old.shape or raster.dtype != np.float32
            or not np.all(np.isfinite(raster)) or np.any(raster < 0)):
        raise ValueError("paired raster matrix")
    new_scores = region_growth._normal_scores(raster)
    old_union, new_union = np.zeros(len(old), bool), np.zeros(len(old), bool)
    for axis in range(3):
        for side in (-1, 1):
            first = _pair_metrics(old, old_scores, sides, axis, side, config)[0]
            second = _pair_metrics(raster, new_scores, sides, axis, side, config)[0]
            np.logical_or(old_union, first, out=old_union)
            np.logical_or(new_union, second, out=new_union)
    rows = []
    for axis in range(3):
        tangent = [column for column in range(3) if column != axis]
        for side in (-1, 1):
            first, before, denominator = _pair_metrics(old, old_scores, sides, axis, side, config)
            second, after, _ = _pair_metrics(raster, new_scores, sides, axis, side, config)
            partition = _pair_partition(first, second, denominator)
            new_only = second & ~first
            globally_new = int(np.count_nonzero(new_only & ~old_union))
            coverage = ("positive" if before["eligible"] else "zero") + "_to_" + (
                        "positive" if after["eligible"] else "zero")
            old_co = (old[:, tangent[0]] > 0) & (old[:, tangent[1]] > 0)
            new_co = (raster[:, tangent[0]] > 0) & (raster[:, tangent[1]] > 0)
            matched = sides[:, axis] == side
            co = _pair_partition(old_co & matched, new_co & matched, denominator)
            witness = None
            if ((frame, axis, side) in PAIR_TARGETS and not before["eligible"]
                    and partition["new_only"]):
                # Argmax on the boolean mask returns the first true index.
                # Node identity is transient and never serialized.
                node = int(np.argmax(new_only))
                witness = _pair_witness(node, old, raster, old_scores, new_scores,
                                        axis, side, sides, config)
            rows.append(dict(frame=frame, axis=axis, side=side, state="measured",
                denominator=denominator, old=before, new=after, partition=partition,
                coverage=coverage, globally_new=globally_new,
                transferred=partition["new_only"]-globally_new,
                cooccurrence_changes=dict(gained=co["new_only"], lost=co["old_only"]),
                witness=witness))
    old_atoms, new_atoms = old.sum(axis=1) > 1e-6, raster.sum(axis=1) > 1e-6
    return rows, _pair_partition(old_union, new_union), _pair_partition(old_atoms, new_atoms)


class PairedSupportObserver:
    """One frame's transient allocation; completed entries contain scalars only."""
    def __init__(self, owner, *, unit_weight_sparse=False):
        self.owner = owner
        self.unit_weight_sparse = unit_weight_sparse is True
        self.frames, self.state = {}, None
        self.frame = None
        self.incomplete = False
        self.frame_seconds = self.total_seconds = 0.0
        self.peak_numeric_bytes = 0
        self.accounted_transient_bytes = 0
        self.items = 0

    def enter_frame(self, frame):
        if self.frame is not None or self.state is not None or frame in self.frames or len(self.frames) >= 6:
            self.discard("frame_census")
            raise ValueError("paired frame census")
        if type(frame) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", frame):
            self.discard("frame_identity")
            raise ValueError("paired frame identity")
        self.frame, self.frame_seconds = frame, 0.0
        self.frames[frame] = dict(frame=frame, state="not_reached", reason="", census=None,
                                 mass=None, union=None, atoms=None, thresholds=None, rows=None)

    def discard(self, reason):
        self.incomplete = True
        if self.frame in self.frames:
            self.frames[self.frame].update(state="failed", reason=str(reason)[:64],
                                          census=None, mass=None, union=None, atoms=None, rows=None)
            self.frames[self.frame]["thresholds"] = None
        if self.state is not None:
            self.state.clear()
        self.state = None

    def finish_frame(self):
        if self.state is not None:
            self.discard("frame_terminal")
        self.frame = None

    def _context(self):
        if (region_growth._attrition_frame.get() != self.frame
                or region_growth._attrition_state.get() is not self.owner):
            raise ValueError("paired source context")

    @staticmethod
    def _fingerprint(array):
        if (type(array) is not region_growth.np.ndarray or not array.flags.c_contiguous
                or array.dtype.kind not in "iufb"):
            raise ValueError("paired buffer domain")
        return (array.shape, array.strides, array.dtype.str, hashlib.sha256(array).digest())

    @staticmethod
    def _ordinal_bytes(index, segments, lengths, weights, associations):
        return struct.pack("!q9d", index, *segments[index], float(lengths[index]),
                           float(weights[index]), *associations[index])

    @_pair_timed
    def begin(self, grey, usable, labels, calibration, config, segments,
              lengths, weights, associations, node_count, scales):
        self._context()
        if self.state is not None or self.frames[self.frame]["state"] != "not_reached":
            raise ValueError("paired duplicate begin")
        np = region_growth.np
        actual_nodes = region_growth._raster_validate_frame(grey, usable, labels, config)
        if (node_count != actual_nodes or config.max_segment_samples > 128
                or config.max_segment_samples < 2 or len(segments) > 5000):
            raise ValueError("paired work census")
        bound = _pair_transient_bound(node_count)
        if bound > PAIR_TEMP_BYTES:
            raise ValueError("paired transient budget")
        self.accounted_transient_bytes = max(self.accounted_transient_bytes, bound)
        visits = region_growth._raster_validate_segments(segments, *grey.shape[::-1])
        if self.unit_weight_sparse:
            visits = 0  # Count only actual sparse sample IDs as they are observed.
        fingerprints = (self._fingerprint(labels), self._fingerprint(usable))
        expected = hashlib.sha256()
        evidence_fingerprints = None
        if len(segments):
            region_growth._raster_validate_evidence(lengths, weights, associations, len(segments))
            evidence_fingerprints = tuple(self._fingerprint(array) for array in
                                          (segments, lengths, weights, associations))
            for index in range(len(segments)):
                expected.update(self._ordinal_bytes(index, segments, lengths, weights, associations))
        raster = np.zeros((node_count, 3), np.float32)
        workspace = None if self.unit_weight_sparse else region_growth._RasterSupportWorkspace()
        self.peak_numeric_bytes = max(self.peak_numeric_bytes, raster.nbytes + (0 if workspace is None else workspace.numeric_bytes))
        self.state = dict(grey=grey, usable=usable, labels=labels, calibration=calibration,
            config=config, segments=segments, lengths=lengths, weights=weights,
            associations=associations, scales=scales, nodes=node_count, count=len(segments),
            visits=visits, observed=0, zero=0, expected=expected.digest(), digest=hashlib.sha256(),
            fingerprints=fingerprints, raster=raster, workspace=workspace, potential=np.zeros(3),
            associated_counts=np.zeros(3, dtype=np.int64), raw_mass=np.zeros(3), raw_invalid=np.zeros(3),
            evidence_fingerprints=evidence_fingerprints,
            old_unsupported=np.zeros(3), new_unsupported=np.zeros(3), support=None,
            finalized=False, returned=False)
        self.frames[self.frame]["state"] = "begun"

    @_pair_timed
    def segment(self, index, segments, lengths, weights, associations, ids, denominator):
        self._context()
        s = self.state
        if (s is None or s["finalized"] or type(index) is not int or index != s["observed"]
                or index >= s["count"] or any(s[key] is not value for key, value in (
                    ("segments", segments), ("lengths", lengths), ("weights", weights),
                    ("associations", associations)))):
            raise ValueError("paired ordinal identity")
        s["digest"].update(self._ordinal_bytes(index, segments, lengths, weights, associations))
        association = associations[index]
        raw_mass = float(lengths[index]) * association
        mass = float(lengths[index]) * float(weights[index]) * association
        if (not region_growth.np.all(region_growth.np.isfinite(mass))
                or region_growth.np.any(mass > SUPPORT_MASS)
                or not region_growth.np.all(region_growth.np.isfinite(raw_mass))
                or region_growth.np.any(raw_mass > SUPPORT_MASS)
                or region_growth.np.any(association > 1)
                or float(weights[index]) > 1.0):
            raise ValueError("paired mass domain")
        s["potential"] += mass
        s["raw_mass"] += raw_mass
        s["associated_counts"] += association > 0
        if not region_growth.np.any(association):
            if ids is not None or denominator is not None:
                raise ValueError("paired zero line")
            s["zero"] += 1
        else:
            if (type(ids) is not region_growth.np.ndarray or ids.ndim != 1
                    or type(denominator) is not int or len(ids) != denominator
                    or not 2 <= denominator <= 128):
                raise ValueError("paired actual samples")
            if self.unit_weight_sparse:
                np = region_growth.np
                if ids.dtype.kind not in "iu" or np.any(ids >= s["nodes"]):
                    raise ValueError("unit sparse sample domain")
                invalid = int(np.count_nonzero(ids < 0))
                s["visits"] += denominator
                if s["visits"] > 640000:
                    raise ValueError("unit sparse sample budget")
                s["old_unsupported"] += mass * invalid / denominator
                s["new_unsupported"] += raw_mass * invalid / denominator
                # Same original IDs and denominator; do not consult the mask,
                # resample the line, redistribute invalid share, or use raster.
                nodes, counts = np.unique(ids[ids >= 0], return_counts=True)
                for node, count in zip(nodes, counts):
                    contribution = float(lengths[index]) * int(count) / denominator
                    s["raster"][int(node)] += (contribution * association).astype(np.float32)
            else:
                s["old_unsupported"] += mass * region_growth.np.count_nonzero(ids < 0) / denominator
                region_growth._raster_allocate_segment(s["raster"], s["labels"], s["usable"],
                    segments[index], lengths[index], weights[index], association, s["workspace"])
                count = region_growth._raster_endpoints(segments[index], *s["grey"].shape[::-1])[-1]
                invalid = region_growth.np.count_nonzero(s["workspace"].ids[:count] < 0)
                s["new_unsupported"] += mass * invalid / count
                s["raw_invalid"] += raw_mass * invalid / count
        s["observed"] += 1

    def _unchanged(self):
        s = self.state
        if s["fingerprints"] != (self._fingerprint(s["labels"]), self._fingerprint(s["usable"])):
            raise ValueError("paired modified labels/mask")
        if s["evidence_fingerprints"] is not None and s["evidence_fingerprints"] != tuple(
                self._fingerprint(s[key]) for key in ("segments","lengths","weights","associations")):
            raise ValueError("paired modified segment evidence")
        if s["support"] is not None and s.get("support_fingerprint") != self._fingerprint(s["support"]):
            raise ValueError("paired modified sparse return")

    @_pair_timed
    def finalize(self, support):
        self._context()
        s = self.state
        if (s is None or s["finalized"] or s["observed"] != s["count"]
                or s["digest"].digest() != s["expected"] or type(support) is not region_growth.np.ndarray
                or support.shape != (s["nodes"], 3) or support.dtype.kind != "f"):
            raise ValueError("paired final census")
        self._unchanged()
        np = region_growth.np
        old_mass, new_mass = support.sum(axis=0, dtype=np.float64), s["raster"].sum(axis=0, dtype=np.float64)
        eps = float(np.finfo(np.float32).eps)
        gamma = 5000*eps/(1-5000*eps) + 2*eps
        allowance = gamma*s["potential"] + 1e-8
        new_potential = s["raw_mass"] if self.unit_weight_sparse else s["potential"]
        new_allowance = gamma*new_potential + 1e-8
        for actual, unsupported, potential, bound in (
                (old_mass, s["old_unsupported"], s["potential"], allowance),
                (new_mass, s["new_unsupported"], new_potential, new_allowance)):
            if (not np.all(np.isfinite(actual)) or np.any(actual < 0)
                    or np.any(np.abs(actual+unsupported-potential) > bound)):
                raise ValueError("paired conservation")
        s["mass"] = dict(potential=s["potential"].tolist(), old=old_mass.tolist(), new=new_mass.tolist(),
            old_unsupported=s["old_unsupported"].tolist(), new_unsupported=s["new_unsupported"].tolist(),
            old_ideal=(s["potential"]-s["old_unsupported"]).tolist(),
            new_ideal=(new_potential-s["new_unsupported"]).tolist(), allowance=allowance.tolist())
        if self.unit_weight_sparse:
            s["mass"].update(counterfactual_potential=new_potential.tolist(),
                             counterfactual_allowance=new_allowance.tolist())
        s["support"], s["finalized"] = support, True
        s["support_fingerprint"] = self._fingerprint(support)
        s["workspace"] = None

    @_pair_timed
    def returned(self, result):
        s = self.state
        if (s is None or not s["finalized"] or s["returned"] or type(result) is not tuple
                or len(result) != 2 or result[0] is not s["support"]):
            raise ValueError("paired actual return")
        self._context()
        self._unchanged()
        s["returned"] = True

    @_pair_timed
    def graph(self, result, args, kwargs):
        self._context()
        s = self.state
        if type(result) is tuple and len(result)==2 and result[0] is None:
            if s is not None: s.clear()
            self.state=None
            self.frames[self.frame].update(state="graph_unavailable",reason="",census=None,
                         mass=None,union=None,atoms=None,thresholds=None,rows=None)
            return
        if s is None or not s["returned"] or type(result) is not tuple or len(result) != 2:
            raise ValueError("paired graph stage")
        names = ("labels", "grey", "lab", "usable", "calibration", "config")
        actual = {name: args[index] if index < len(args) else kwargs.get(name)
                  for index, name in enumerate(names)}
        if any(actual[key] is not s[key] for key in ("labels", "grey", "usable", "calibration", "config")):
            raise ValueError("paired build inputs")
        graph = result[0]
        if (graph is None or graph.supports is not s["support"] or graph.labels is not s["labels"]
                or graph.usable is not s["usable"] or len(graph.node_area) != s["nodes"]
                or graph.lab is not actual["lab"]
                or kwargs.get("original_pixel_scale", (1., 1.)) != s["scales"]):
            raise ValueError("paired graph identity")
        self._unchanged()
        rows, union, atoms = _pair_project(self.frame, graph, s["raster"], s["config"])
        entry = dict(frame=self.frame, state="measured", reason="",
            census=dict(retained=s["count"], observed=s["observed"], zero=s["zero"],
                        visits=s["visits"], nodes=s["nodes"], evidence=dict(
                            schema=PAIR_EVIDENCE_SCHEMA, stage=PAIR_EVIDENCE_STAGE,
                            dense_invalid_scope=PAIR_INVALID_SCOPE,
                            nonzero_associated_segments=s["associated_counts"].tolist(),
                            length_association_mass=s["raw_mass"].tolist(),
                            length_weight_association_mass=s["potential"].tolist(),
                            dense_invalid_length_association_mass=s["raw_invalid"].tolist(),
                            dense_invalid_length_weight_association_mass=s["new_unsupported"].tolist())),
            mass=s["mass"], union=union,
            atoms=atoms, rows=rows)
        if self.unit_weight_sparse:
            entry["census"].pop("evidence")
            entry["census"]["sparse_samples"] = entry["census"].pop("visits")
        entry["thresholds"] = [float(s["config"].min_seed_score),
                                float(s["config"].min_seed_margin),
                                float(s["config"].min_seed_support)]
        if self.items + 7 > self.owner.max_items:
            raise ValueError("paired item cap")
        for row in [dict((k,v) for k,v in entry.items() if k != "rows")] + rows:
            encoded = self.owner._encode(row)
            if len(encoded) > PAIR_ROW_BYTES:
                raise ValueError("paired row byte cap")
            if not self.owner._reserve(len(encoded) + 128):
                raise ValueError("paired shared envelope")
        if len(canonical_bytes(entry)) > PAIR_BYTES:
            raise ValueError("paired byte cap")
        self.frames[self.frame] = entry
        self.items += 7
        s.clear()
        self.state = None


def _pair_int(value, maximum=20000):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("paired count domain")


def _pair_vector(values, count=3):
    if (type(values) is not list or len(values) != count
            or any(type(value) is not float or not math.isfinite(value)
                   or value < 0 or value > SUPPORT_MASS for value in values)):
        raise ValueError("paired finite vector")


def _pair_validate_partition(partition, denominator):
    if type(partition) is not dict or set(partition) != {"both", "old_only", "new_only", "neither"}:
        raise ValueError("paired partition fields")
    for value in partition.values():
        _pair_int(value)
    if sum(partition.values()) != denominator:
        raise ValueError("paired partition conservation")


def _pair_validate_evidence(evidence, census, mass):
    keys = {"schema", "stage", "dense_invalid_scope", "nonzero_associated_segments", "length_association_mass",
            "length_weight_association_mass", "dense_invalid_length_association_mass",
            "dense_invalid_length_weight_association_mass"}
    if (type(evidence) is not dict or set(evidence) != keys
            or evidence["schema"] != PAIR_EVIDENCE_SCHEMA or evidence["stage"] != PAIR_EVIDENCE_STAGE
            or evidence["dense_invalid_scope"] != PAIR_INVALID_SCOPE):
        raise ValueError("paired evidence schema/stage")
    counts = evidence["nonzero_associated_segments"]
    assigned = census["retained"] - census["zero"]
    if type(counts) is not list or len(counts) != 3:
        raise ValueError("paired evidence counts")
    for count in counts: _pair_int(count, assigned)
    if not max(counts, default=0) <= assigned <= sum(counts):
        raise ValueError("paired evidence associated census")
    for key in keys - {"schema", "stage", "dense_invalid_scope", "nonzero_associated_segments"}:
        _pair_vector(evidence[key])
    raw = evidence["length_association_mass"]
    weighted = evidence["length_weight_association_mass"]
    invalid = evidence["dense_invalid_length_association_mass"]
    invalid_weighted = evidence["dense_invalid_length_weight_association_mass"]
    if weighted != mass["potential"] or invalid_weighted != mass["new_unsupported"]:
        raise ValueError("paired evidence weighted reconciliation")
    for axis in range(3):
        # Retained screen-space segments fit the <=960x960 work domain.
        # 1358 conservatively bounds a diagonal even before endpoint rounding.
        allowance = mass["allowance"][axis]
        valid_raw, valid_weighted = raw[axis] - invalid[axis], weighted[axis] - invalid_weighted[axis]
        if (raw[axis] > counts[axis] * 1358 + 1e-8 or invalid[axis] > raw[axis] + allowance
                or invalid_weighted[axis] > weighted[axis] + 1e-8
                or weighted[axis] > raw[axis] + allowance
                or invalid_weighted[axis] > invalid[axis] + allowance
                or valid_weighted > valid_raw + allowance
                or (counts[axis] == 0 and any(values[axis] != 0
                    for values in (raw, weighted, invalid, invalid_weighted)))):
            raise ValueError("paired evidence mass bounds")


def _pair_validate_witness(witness, thresholds, axis, side):
    if type(witness) is not dict or set(witness) != {"old", "new"}:
        raise ValueError("paired witness fields")
    for name in ("old", "new"):
        row = witness[name]
        if type(row) is not dict or set(row) != {"support", "score", "margin", "tangent", "side", "passes"}:
            raise ValueError("paired witness row")
        _pair_vector(row["support"])
        if (type(row["score"]) is not float or not 0 <= row["score"] <= 1
                or type(row["margin"]) is not float or not -1 <= row["margin"] <= 1
                or type(row["tangent"]) is not float or not math.isfinite(row["tangent"])
                or row["tangent"] < 0 or type(row["side"]) is not int or row["side"] != side
                or type(row["passes"]) is not list or len(row["passes"]) != 4
                or any(type(v) is not bool for v in row["passes"])):
            raise ValueError("paired witness scalar")
        support = region_growth.np.asarray([row["support"]], dtype=region_growth.np.float32)
        expected = _pair_witness_values(support, region_growth._normal_scores(support),
                                       axis, row["side"], side, thresholds)
        if row != expected:
            raise ValueError("paired witness source formula")
    if (all(witness["old"]["passes"]) or not all(witness["new"]["passes"])
            or witness["old"]["support"] == witness["new"]["support"]):
        raise ValueError("paired witness qualification")


def _pair_account(record, attrition, *, unit_weight_sparse=False):
    keys = {"schema", "frames", "frame_rows", "rows", "limits", "incomplete", "status",
            "checks", "losses", "qualifying_targets", "decision", "causal_scope"}
    schema = UNIT_WEIGHT_SCHEMA if unit_weight_sparse else PAIR_SCHEMA
    scope = UNIT_WEIGHT_SCOPE if unit_weight_sparse else "combined_centerline_and_usable_mask_effect_only"
    if (type(record) is not dict or set(record) != keys or record["schema"] != schema
            or type(record["frames"]) is not list or len(record["frames"]) > 6
            or any(type(v) is not str for v in record["frames"])
            or len(set(record["frames"])) != len(record["frames"])
            or type(record["incomplete"]) is not bool
            or record["causal_scope"] != scope
            or record["limits"] != dict(items=4096, bytes=524288, pair_bytes=PAIR_BYTES)):
        raise ValueError("paired envelope")
    frames = record["frame_rows"]
    rows = record["rows"]
    if (type(frames) is not list or type(rows) is not list
            or [row.get("frame") for row in frames] != record["frames"]
            or len(rows) != 6*len(frames)):
        raise ValueError("paired frame census")
    if any(len(canonical_bytes(row)) > PAIR_ROW_BYTES for row in frames + rows):
        raise ValueError("paired row byte cap")
    if any(type(r.get("axis")) is not int or type(r.get("side")) is not int for r in rows):
        raise ValueError("paired axis/side type")
    if (type(record["losses"]) is not list or any(type(row) is not dict
            or type(row.get("axis")) is not int or type(row.get("side")) is not int
            for row in record["losses"])
            or type(record["qualifying_targets"]) is not list
            or any(type(row) is not list or len(row) != 3 or type(row[1]) is not int
                   or type(row[2]) is not int for row in record["qualifying_targets"])):
        raise ValueError("paired decision axis/side type")
    expected_keys = [(frame,axis,side) for frame in record["frames"]
                     for axis in range(3) for side in (-1,1)]
    if [(r.get("frame"),r.get("axis"),r.get("side")) for r in rows] != expected_keys:
        raise ValueError("paired row census")
    checks = dict(census=True, mass=True, partitions=True, frame_union=True,
                  source_seed_join=True, target_witness=True)
    measured = {}
    for entry in frames:
        if set(entry) != {"frame","state","reason","census","mass","union","atoms","thresholds"}:
            raise ValueError("paired frame fields")
        if type(entry["reason"]) is not str or len(entry["reason"]) > 64:
            raise ValueError("paired reason")
        if entry["state"] != "measured":
            if entry["state"] not in ("not_reached","graph_unavailable","failed","missing") or any(
                    entry[k] is not None for k in ("census","mass","union","atoms","thresholds")):
                raise ValueError("paired unmeasured frame")
            checks["census"] = False
            continue
        census = entry["census"]
        census_keys = ({"retained","observed","zero","sparse_samples","nodes"} if unit_weight_sparse else
                       {"retained","observed","zero","visits","nodes","evidence"})
        if type(census) is not dict or set(census) != census_keys:
            raise ValueError("paired census fields")
        for key,maximum in (("retained",5000),("observed",5000),("zero",5000),
                            ("sparse_samples" if unit_weight_sparse else "visits",
                             640000 if unit_weight_sparse else 4800000),("nodes",20000)):
            _pair_int(census[key], maximum)
        if census["observed"] != census["retained"] or census["zero"] > census["observed"]:
            raise ValueError("paired missing ordinals")
        _pair_vector(entry["thresholds"])
        _pair_validate_partition(entry["union"],census["nodes"])
        _pair_validate_partition(entry["atoms"],census["nodes"])
        mass = entry["mass"]
        mass_keys = {"potential","old","new","old_unsupported", "new_unsupported","old_ideal","new_ideal","allowance"}
        if unit_weight_sparse:
            mass_keys.update(("counterfactual_potential","counterfactual_allowance"))
        if type(mass) is not dict or set(mass) != mass_keys:
            raise ValueError("paired mass fields")
        for vector in mass.values(): _pair_vector(vector)
        if not unit_weight_sparse:
            _pair_validate_evidence(census["evidence"], census, mass)
        elif not 2*(census["retained"]-census["zero"]) <= census["sparse_samples"] <= 128*(census["retained"]-census["zero"]):
            raise ValueError("unit sparse sample census")
        eps=2**-23
        for axis in range(3):
            p = mass["potential"][axis]
            if unit_weight_sparse:
                assigned = census["retained"] - census["zero"]
                raw = mass["counterfactual_potential"][axis]
                # Source-independent work-domain bound; a resealed ledger
                # cannot justify arbitrary mass by raising its own allowance.
                if ((assigned == 0 and raw != 0.)
                        or raw > 1358*assigned + 1e-8):
                    raise ValueError("unit sparse raw potential bound")
            if unit_weight_sparse and (p > mass["counterfactual_potential"][axis] + 1e-8
                    or mass["old_unsupported"][axis] > mass["new_unsupported"][axis] + 1e-8
                    or mass["old_ideal"][axis] > mass["new_ideal"][axis] + 1e-8):
                raise ValueError("unit sparse weighted/raw bounds")
            expected_allowance = (5000*eps/(1-5000*eps)+2*eps)*p+1e-8
            if mass["allowance"][axis] != expected_allowance:
                raise ValueError("paired mass allowance")
            for version in ("old","new"):
                if unit_weight_sparse and version == "new":
                    p = mass["counterfactual_potential"][axis]
                    expected_allowance = (5000*eps/(1-5000*eps)+2*eps)*p+1e-8
                    if mass["counterfactual_allowance"][axis] != expected_allowance:
                        raise ValueError("unit sparse mass allowance")
                if (not math.isclose(mass[version+"_ideal"][axis]+mass[version+"_unsupported"][axis],p,rel_tol=1e-12,abs_tol=1e-8)
                        or abs(mass[version][axis]+mass[version+"_unsupported"][axis]-p) > expected_allowance):
                    raise ValueError("paired mass conservation")
        measured[entry["frame"]] = entry
    losses, targets = [], []
    seed_rows = {} if attrition is None else {tuple(r["identity"]): r for r in attrition["stages"]["seed"]["terminal"]}
    for row in rows:
        if set(row) != {"frame","axis","side","state","denominator","old","new","partition","coverage","globally_new","transferred","cooccurrence_changes","witness"}:
            raise ValueError("paired row fields")
        key = row["frame"],row["axis"],row["side"]
        if row["frame"] not in measured:
            if row["state"] != next(f["state"] for f in frames if f["frame"] == row["frame"]) or any(
                    row[k] is not None for k in row if k not in ("frame","axis","side","state")):
                raise ValueError("paired unmeasured row")
            continue
        entry = measured[row["frame"]]
        _pair_int(row["denominator"])
        if row["state"] != "measured" or row["denominator"] > entry["census"]["nodes"]:
            raise ValueError("paired denominator")
        for version in ("old","new"):
            value = row[version]
            if set(value) != {"mass","positive","cooccurrence","raw","sequential","eligible"}:
                raise ValueError("paired metric fields")
            _pair_vector(value["mass"])
            for vector in (value["positive"],value["raw"],value["sequential"]):
                if type(vector) is not list or len(vector) != 3: raise ValueError("paired counts")
                for item in vector:
                    _pair_int(item,row["denominator"])
            for item in (value["cooccurrence"],value["eligible"]): _pair_int(item,row["denominator"])
            if value["sequential"] != sorted(value["sequential"],reverse=True) or value["eligible"] != value["sequential"][-1]:
                raise ValueError("paired sequential counts")
            if any(a>b for a,b in zip(value["sequential"],value["raw"])):
                raise ValueError("paired raw intersections")
        _pair_validate_partition(row["partition"],row["denominator"])
        part=row["partition"]
        if row["old"]["eligible"] != part["both"]+part["old_only"] or row["new"]["eligible"] != part["both"]+part["new_only"]:
            raise ValueError("paired row eligibility")
        expected_coverage=("positive" if row["old"]["eligible"] else "zero")+"_to_"+("positive" if row["new"]["eligible"] else "zero")
        if row["coverage"] != expected_coverage: raise ValueError("paired coverage")
        _pair_int(row["globally_new"],entry["union"]["new_only"])
        _pair_int(row["transferred"])
        if row["globally_new"]+row["transferred"] != part["new_only"]:
            raise ValueError("paired transfer")
        changes=row["cooccurrence_changes"]
        if type(changes) is not dict or set(changes) != {"gained","lost"}: raise ValueError("paired tangent changes")
        for value in changes.values(): _pair_int(value,row["denominator"])
        if row["old"]["cooccurrence"]+changes["gained"]-changes["lost"] != row["new"]["cooccurrence"]:
            raise ValueError("paired tangent conservation")
        source=seed_rows.get(key)
        if source is None or source["disposition"] == "not_reached":
            checks["source_seed_join"]=False
        elif source["disposition"] != "not_reached":
            counts=source["seed_predicates"]
            if (source["eligible_seed_count"] != row["old"]["eligible"] or counts["side_match_count"] != row["denominator"]
                    or [counts[k] for k in ("after_score_count","after_margin_count","after_support_count")] != row["old"]["sequential"]):
                raise ValueError("paired source seed join")
        if part["old_only"]:
            losses.append(dict(frame=key[0],axis=key[1],side=key[2],old_only=part["old_only"],coverage=row["coverage"]))
        qualifies=key in PAIR_TARGETS and not row["old"]["eligible"] and part["new_only"]>0
        if qualifies:
            _pair_validate_witness(row["witness"],entry["thresholds"],key[1],key[2])
            targets.append(list(key))
        elif row["witness"] is not None: raise ValueError("paired non-target witness")
    for frame,entry in measured.items():
        current=[row for row in rows if row["frame"]==frame]
        for version in ("old","new"):
            counts=[row[version]["eligible"] for row in current]
            union=entry["union"]["both"]+entry["union"][version+"_only"]
            if not max(counts,default=0)<=union<=sum(counts): raise ValueError("paired union bounds")
        if not max((r["globally_new"] for r in current),default=0)<=entry["union"]["new_only"]<=sum(r["globally_new"] for r in current):
            raise ValueError("paired global-new union")
    complete=bool(frames) and all(checks.values()) and not record["incomplete"]
    qualified = targets if complete else []
    if record["losses"] != losses or record["qualifying_targets"] != qualified:
        raise ValueError("paired decision reconciliation")
    decision=("inconclusive" if not complete else "continue_candidate" if targets else
              "stop_no_new_only" if not any(r["partition"]["new_only"] for r in rows) else "stop_non_target_only")
    if record["decision"] != decision: raise ValueError("paired decision")
    return checks,"complete" if complete else "incomplete"


def _paired_support_record(observer, capture, attrition):
    pair=observer.paired
    unit = pair.unit_weight_sparse
    frames=list(capture.graph.frame_ids) if capture.graph is not None else list(pair.frames)
    frame_rows,rows=[],[]
    for frame in frames:
        entry=pair.frames.get(frame,dict(frame=frame,state="missing",reason="missing",census=None,
                      mass=None,union=None,atoms=None,thresholds=None,rows=None))
        frame_rows.append({k:v for k,v in entry.items() if k!="rows"})
        if entry["rows"] is not None: rows.extend(entry["rows"])
        else:
            for axis in range(3):
                for side in (-1,1): rows.append(dict(frame=frame,axis=axis,side=side,state=entry["state"],
                    denominator=None,old=None,new=None,partition=None,coverage=None,globally_new=None,
                    transferred=None,cooccurrence_changes=None,witness=None))
    targets=[list((r["frame"],r["axis"],r["side"])) for r in rows if r["state"]=="measured" and r["witness"] is not None]
    losses=[dict(frame=r["frame"],axis=r["axis"],side=r["side"],old_only=r["partition"]["old_only"],coverage=r["coverage"])
            for r in rows if r["state"]=="measured" and r["partition"]["old_only"]]
    record=dict(schema=UNIT_WEIGHT_SCHEMA if unit else PAIR_SCHEMA,frames=frames,frame_rows=frame_rows,rows=rows,
        limits=dict(items=4096,bytes=524288,pair_bytes=PAIR_BYTES),incomplete=pair.incomplete,
        status="incomplete",checks={},losses=losses,qualifying_targets=targets,decision="inconclusive",
        causal_scope=UNIT_WEIGHT_SCOPE if unit else "combined_centerline_and_usable_mask_effect_only")
    if list(pair.frames) != frames:
        record["incomplete"] = True
    # Determine completeness using the same validator, with decision set only
    # after measured frame and seed census reconciliation is known.
    if not record["incomplete"] and all(r["state"]=="measured" for r in rows) and frames:
        seed_keys=set() if attrition is None else {tuple(r["identity"]) for r in attrition["stages"]["seed"]["terminal"]
                                                if r["disposition"] != "not_reached"}
        if all((r["frame"],r["axis"],r["side"]) in seed_keys for r in rows):
            record["decision"]=("continue_candidate" if targets else "stop_no_new_only"
                                if not any(r["partition"]["new_only"] for r in rows) else "stop_non_target_only")
    if record["decision"] == "inconclusive":
        record["qualifying_targets"] = []
    record["checks"],record["status"]=_pair_account(record,attrition,unit_weight_sparse=unit)
    combined=dict(reading_attrition=attrition)
    combined["unit_weight_sparse_support" if unit else "paired_segment_support"] = record
    if observer.support_enabled:
        combined["support_score_attrition"]=_support_score_record(observer,capture,attrition)
    if (len(canonical_bytes(record))>PAIR_BYTES or len(canonical_bytes(combined))>observer.max_bytes
            or len(rows)+len(frame_rows)+observer.support_items+(0 if attrition is None else
                sum(len(s["terminal"]) for s in attrition["stages"].values()))>observer.max_items):
        raise ValueError("paired combined envelope")
    return record

ATTRITION_DISPOSITIONS = {
    "seed": frozenset(("available", "absent", "not_reached")),
    "proposal": frozenset(("mask_empty", "area_rejected", "polygon_rejected", "deduplicated", "returned", "diversity_cap")),
    "reading": frozenset(("window_incidence", "window_invalid", "periodicity", "structural_energy",
        "distance_alternative", "coverage", "per_axis_side_cap", "accepted", "reading_cap",
        "proposal_validation", "frame_rejection", "reading_stage_failed")),
}


class ReadingAttritionObserver(region_growth._ScalarAttritionObserver):
    """Evaluation-only projection of existing terminal hooks; never retain masks."""
    def __init__(self, *, max_items=ATTRITION_ITEMS, max_bytes=ATTRITION_BYTES,
                 paired_segment_support=False, unit_weight_sparse_support=False):
        super().__init__(max_items=max_items, max_bytes=max_bytes)
        self.scalar_rows = {stage: {} for stage in ATTRITION_STAGES}
        self.support_enabled = False
        self.support_incomplete = False
        self.support_frames = {}
        self.support_items = 0
        self.support_omitted = 0
        if paired_segment_support is True and unit_weight_sparse_support is True:
            raise ValueError("diagnostic counterfactual modes are separate")
        self.paired = (PairedSupportObserver(self, unit_weight_sparse=unit_weight_sparse_support)
                       if paired_segment_support is True or unit_weight_sparse_support is True else None)

    def support_begin(self, frame):
        self.support_enabled = True
        try:
            if type(frame) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", frame):
                raise ValueError("support frame")
            if frame in self.support_frames or len(self.support_frames) >= 32:
                raise ValueError("duplicate or excessive support frame")
            self.support_frames[frame] = dict(calls=0, state="not_reached", rows=None)
        except Exception:
            self.support_incomplete = True

    def support_graph(self, frame, result):
        try:
            entry = self.support_frames[frame]
            entry["calls"] += 1
            if entry["calls"] != 1 or region_growth._attrition_frame.get() != frame:
                raise ValueError("support graph identity or repeated call")
            if type(result) is not tuple or len(result) != 2:
                raise ValueError("support graph return")
            graph, _evidence = result
            if graph is None:
                entry["state"] = "graph_unavailable"
                return
            rows = _support_score_rows(frame, graph)
            if self.support_items + len(rows) > self.max_items:
                self.support_omitted += len(rows)
                raise ValueError("support item cap")
            for row in rows:
                if not self._reserve(len(self._encode(row)) + 128):
                    self.support_omitted += len(rows)
                    raise ValueError("support byte cap")
            self.support_items += len(rows)
            entry.update(state="measured", rows=rows)
        except Exception:
            self.support_incomplete = True
            if frame in self.support_frames and self.support_frames[frame]["rows"] is None:
                self.support_frames[frame]["state"] = "failed"

    def finish(self, stage, identity, disposition, **fields):
        # Preserve the original observer protocol, including its failure guard.
        super().finish(stage, identity, disposition, **fields)
        if stage not in ATTRITION_STAGES:
            return
        try:
            key = self._encode(identity).decode()
            if key in self.scalar_rows[stage]:
                self.incomplete = True
                return
            row = dict(identity=list(identity), disposition=disposition,
                       schema=ATTRITION_ROW_SCHEMA,
                       axis=fields.get("axis"), side=fields.get("side"),
                       plane_index=fields.get("plane_index"),
                       eligible_seed_count=fields.get("eligible_seed_count"), seed_predicates=None)
            if stage == "seed":
                if disposition == "not_reached":
                    if any(name in fields for name in SEED_COUNT_FIELDS + SEED_MAX_FIELDS):
                        raise ValueError("not reached seed predicates")
                else:
                    # Copy only the actual source hook's scalar fields. No array
                    # sink, inference object, or reconstructed predicate enters telemetry.
                    row["seed_predicates"] = {name: fields[name] for name in SEED_COUNT_FIELDS + SEED_MAX_FIELDS}
            _attrition_row(stage, row)
            if disposition not in ATTRITION_DISPOSITIONS[stage]:
                raise ValueError("unknown terminal disposition")
            if self._reserve(len(self._encode(row)) + 128):
                self.scalar_rows[stage][key] = row
        except Exception:
            self.incomplete = True


def _support_factors(components, axis):
    """Independent scalar diagnostic arithmetic; never substitutes source scores."""
    a, b = [other for other in range(3) if other != axis]
    tangent = components[a] + components[b]
    balance = 2.0 * math.sqrt(components[a] * components[b]) / max(tangent, 1e-6)
    normal_fraction = tangent / max(tangent + components[axis], 1e-6)
    reliability = tangent / (tangent + 24.0)
    return balance, normal_fraction, reliability, min(1.0, max(0.0, balance * normal_fraction * reliability))


def _support_score_rows(frame, graph):
    np = region_growth.np
    supports, scores, sides = graph.supports, graph.plane_scores, graph.sides
    if any(type(value) is not np.ndarray for value in (supports, scores, sides)):
        raise ValueError("support arrays")
    if (supports.ndim != 2 or supports.shape[1] != 3 or not 0 <= len(supports) <= SUPPORT_NODES
            or scores.shape != supports.shape or sides.shape != supports.shape
            or any(value.dtype.kind != "f" or value.dtype.itemsize not in (4, 8) for value in (supports, scores))
            or sides.dtype.kind not in "iu"):
        raise ValueError("support shape or dtype")
    if (not np.all(np.isfinite(supports)) or not np.all(np.isfinite(scores))
            or np.any(supports < 0) or np.any(supports > SUPPORT_MASS)
            or np.any(scores < 0) or np.any(scores > 1) or not np.all(np.isin(sides, (-1, 0, 1)))):
        raise ValueError("support scalar domain")
    rows = []
    for axis in range(3):
        tangent = [other for other in range(3) if other != axis]
        for side in (-1, 1):
            selected = np.flatnonzero(sides[:, axis] == side)
            values = supports[selected]
            positive = values > 0
            patterns = positive[:, 0].astype(int) + 2 * positive[:, 1] + 4 * positive[:, 2]
            witness = None
            if len(selected):
                # One actual source-score maximum; all witness fields refer to
                # that same node. Ties choose the first source array position.
                winner = int(selected[int(np.argmax(scores[selected, axis]))])
                components = [float(value) for value in supports[winner]]
                balance, normal_fraction, reliability, derived = _support_factors(components, axis)
                witness = dict(components=components, normal_score=float(scores[winner, axis]),
                    balance=balance, normal_fraction=normal_fraction, reliability=reliability, derived_score=derived)
            counts = [int(value) for value in positive.sum(axis=0)]
            summary = dict(node_count=len(supports), side_count=len(selected),
                component_mass=[float(value) for value in values.sum(axis=0, dtype=np.float64)],
                component_nonzero=counts, tangent_nonzero=[counts[index] for index in tangent],
                tangent_cooccurrence=int(np.count_nonzero(positive[:, tangent[0]] & positive[:, tangent[1]])),
                support_patterns=[int(np.count_nonzero(patterns == code)) for code in range(8)], witness=witness)
            row = dict(schema=SUPPORT_ROW_SCHEMA, identity=[frame, axis, side], state="measured", summary=summary)
            _support_score_row(row)
            rows.append(row)
    return rows


def _support_score_row(row):
    if type(row) is not dict or set(row) != {"schema", "identity", "state", "summary"} or row["schema"] != SUPPORT_ROW_SCHEMA:
        raise ValueError("support row schema")
    _attrition_row("seed", dict(schema=ATTRITION_ROW_SCHEMA, identity=row["identity"], disposition="pending",
        axis=None, side=None, plane_index=None, eligible_seed_count=None, seed_predicates=None))
    if type(row["state"]) is not str or row["state"] not in SUPPORT_STATES:
        raise ValueError("support state")
    summary = row["summary"]
    if row["state"] != "measured":
        if summary is not None:
            raise ValueError("unmeasured support")
        return
    if type(summary) is not dict or set(summary) != {"node_count", "side_count", "component_mass", "component_nonzero",
            "tangent_nonzero", "tangent_cooccurrence", "support_patterns", "witness"}:
        raise ValueError("support summary schema")
    nodes, count = summary["node_count"], summary["side_count"]
    if type(nodes) is not int or not 0 <= nodes <= SUPPORT_NODES or type(count) is not int or not 0 <= count <= nodes:
        raise ValueError("support counts")
    for key, length in (("component_mass", 3), ("component_nonzero", 3), ("tangent_nonzero", 2), ("support_patterns", 8)):
        values = summary[key]
        if type(values) is not list or len(values) != length:
            raise ValueError("support vector length")
        if key == "component_mass":
            if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= SUPPORT_MASS for value in values):
                raise ValueError("support mass")
        elif any(type(value) is not int or not 0 <= value <= count for value in values):
            raise ValueError("support positive count")
    patterns = summary["support_patterns"]
    if sum(patterns) != count:
        raise ValueError("support pattern conservation")
    axis = row["identity"][1]
    tangent = [other for other in range(3) if other != axis]
    if summary["component_nonzero"] != [sum(patterns[code] for code in range(8) if code & (1 << column)) for column in range(3)]:
        raise ValueError("support component conservation")
    if any((mass == 0) != (positive == 0) for mass, positive in zip(summary["component_mass"], summary["component_nonzero"])):
        raise ValueError("support mass presence")
    if summary["tangent_nonzero"] != [summary["component_nonzero"][column] for column in tangent]:
        raise ValueError("support tangent join")
    both = summary["tangent_cooccurrence"]
    if type(both) is not int or both != sum(patterns[code] for code in range(8) if all(code & (1 << column) for column in tangent)):
        raise ValueError("support cooccurrence")
    witness = summary["witness"]
    if not count:
        if witness is not None:
            raise ValueError("empty support witness")
        return
    if type(witness) is not dict or set(witness) != {"components", "normal_score", "balance", "normal_fraction", "reliability", "derived_score"}:
        raise ValueError("support witness schema")
    components = witness["components"]
    if (type(components) is not list or len(components) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= mass
                   for value, mass in zip(components, summary["component_mass"]))):
        raise ValueError("support witness components")
    if any(type(witness[key]) not in (int, float) or not math.isfinite(witness[key]) or not 0 <= witness[key] <= 1
           for key in witness if key != "components"):
        raise ValueError("support witness factors")
    expected = _support_factors(components, axis)
    if any(not math.isclose(witness[key], value, rel_tol=2e-6, abs_tol=5e-7)
           for key, value in zip(("balance", "normal_fraction", "reliability", "derived_score"), expected)):
        raise ValueError("support witness factor join")
    if not math.isclose(witness["normal_score"], expected[-1], rel_tol=2e-6, abs_tol=5e-7):
        raise ValueError("support source score join")
    code = sum(1 << column for column, value in enumerate(components) if value > 0)
    if not patterns[code]:
        raise ValueError("support witness pattern join")


def _support_score_account(record, attrition):
    if (type(record) is not dict or set(record) != {"schema", "status", "frames", "limits", "observer_incomplete", "omitted", "rows", "checks"}
            or record["schema"] != SUPPORT_SCHEMA or record["status"] not in ("complete", "incomplete")):
        raise ValueError("support envelope")
    frames, limits = record["frames"], record["limits"]
    if (type(frames) is not list or not 1 <= len(frames) <= 32
            or any(type(frame) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", frame) for frame in frames)
            or len(set(frames)) != len(frames) or type(limits) is not dict or set(limits) != {"items", "bytes"}
            or any(type(limits[key]) is not int or not 1 <= limits[key] <= cap for key, cap in (("items", ATTRITION_ITEMS), ("bytes", ATTRITION_BYTES)))
            or type(record["observer_incomplete"]) is not bool or type(record["omitted"]) is not int or record["omitted"] < 0
            or type(record["rows"]) is not list or len(record["rows"]) > min(limits["items"], 192)):
        raise ValueError("support bounds")
    rows = {}
    for row in record["rows"]:
        _support_score_row(row)
        key = tuple(row["identity"])
        if key[0] not in frames or key in rows:
            raise ValueError("support identity join")
        rows[key] = row
    seeds = {} if attrition is None else {tuple(row["identity"]): row for row in attrition["stages"]["seed"]["terminal"]}
    checks = dict(row_census=set(rows) == {(frame, axis, side) for frame in frames for axis in range(3) for side in (-1, 1)},
                  seed_join=attrition is not None and attrition["frames"] == frames)
    for key, row in rows.items():
        seed = seeds.get(key)
        summary = row["summary"]
        if summary is None:
            joined = row["state"] in ("not_reached", "graph_unavailable") and seed is not None and seed["disposition"] == "not_reached"
        else:
            fields = None if seed is None else seed["seed_predicates"]
            score = None if summary["witness"] is None else summary["witness"]["normal_score"]
            if seed is not None and seed["disposition"] == "not_reached":
                # Graph construction can precede a proposal early return. Its
                # measured supports remain evidence; seed predicates do not.
                joined = fields is None and seed["eligible_seed_count"] is None
            else:
                joined = (fields is not None and summary["node_count"] == fields["node_count"]
                          and summary["side_count"] == fields["side_match_count"] and score == fields["max_side_normal_score"])
        checks["seed_join"] = checks["seed_join"] and joined
    checks["frame_partition"] = True
    for frame in frames:
        frame_rows = [rows[key] for key in rows if key[0] == frame]
        if len({row["state"] for row in frame_rows}) != 1:
            checks["frame_partition"] = False
        measured = [row for row in frame_rows if row["summary"] is not None]
        if measured and len({row["summary"]["node_count"] for row in measured}) != 1:
            checks["frame_partition"] = False
        for axis in range(3):
            pair = [row["summary"] for row in measured if row["identity"][1] == axis]
            if pair and (len(pair) != 2 or sum(row["side_count"] for row in pair) > pair[0]["node_count"]):
                checks["frame_partition"] = False
    items = len(frames) + len(rows)
    if attrition is not None:
        items += sum(len(data[key]) for data in attrition["stages"].values() for key in ("expected", "terminal"))
        items += sum(len(attrition[key]) for key in ("frames", "handoffs", "returned_readings", "lineage", "counts"))
        if limits != attrition["limits"]:
            raise ValueError("support shared limits")
    if items > limits["items"] or len(canonical_bytes(dict(reading_attrition=attrition, support_score_attrition=record))) > limits["bytes"]:
        raise ValueError("support combined envelope cap")
    status = "complete" if all(checks.values()) and not record["observer_incomplete"] and not record["omitted"] else "incomplete"
    return checks, status


def _support_score_record(observer, capture, attrition):
    try:
        frames = list(capture.graph.frame_ids)
        rows = []
        for frame in frames:
            entry = observer.support_frames.get(frame)
            if entry is not None and entry["rows"] is not None:
                rows.extend(entry["rows"])
            else:
                state = "missing" if entry is None else entry["state"]
                rows.extend(dict(schema=SUPPORT_ROW_SCHEMA, identity=[frame, axis, side], state=state, summary=None)
                            for axis in range(3) for side in (-1, 1))
        record = dict(schema=SUPPORT_SCHEMA, status="incomplete", frames=frames,
            limits=dict(items=observer.max_items, bytes=observer.max_bytes), rows=rows, checks={},
            observer_incomplete=observer.incomplete or observer.support_incomplete or set(observer.support_frames) != set(frames),
            omitted=observer.omitted + observer.support_omitted)
        record["checks"], record["status"] = _support_score_account(record, attrition)
        _support_score_account(record, attrition)
        return record
    except Exception:
        return None


def _attrition_row(stage, row):
    if (type(row) is not dict or set(row) != {"schema", "identity", "disposition", "axis", "side",
            "plane_index", "eligible_seed_count", "seed_predicates"}
            or row["schema"] != ATTRITION_ROW_SCHEMA):
        raise ValueError("attrition row fields")
    identity = row["identity"]
    if (type(identity) is not list or len(identity) != (3 if stage == "seed" else 2)
            or type(identity[0]) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identity[0])
            or type(identity[1]) is not int or not 0 <= identity[1] < ATTRITION_ITEMS
            or stage == "seed" and (identity[1] not in (0, 1, 2) or type(identity[2]) is not int or identity[2] not in (-1, 1))):
        raise ValueError("attrition identity")
    if type(row["disposition"]) is not str or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", row["disposition"]):
        raise ValueError("attrition reason code")
    if row["axis"] is not None and (type(row["axis"]) is not int or row["axis"] not in (0, 1, 2)):
        raise ValueError("attrition axis")
    if row["side"] is not None and (type(row["side"]) not in (int, float) or row["side"] not in (-1, 1)):
        raise ValueError("attrition side")
    if row["plane_index"] is not None and (type(row["plane_index"]) is not int or not 0 <= row["plane_index"] < 128):
        raise ValueError("attrition plane index")
    if row["eligible_seed_count"] is not None and (stage != "seed" or type(row["eligible_seed_count"]) is not int or not 0 <= row["eligible_seed_count"] <= 16777216):
        raise ValueError("attrition seed count")
    predicates = row["seed_predicates"]
    if stage != "seed" or row["disposition"] in ("pending", "not_reached"):
        if predicates is not None or row["eligible_seed_count"] is not None:
            raise ValueError("unmeasured seed predicates")
        return
    if row["disposition"] not in ("available", "absent"):
        raise ValueError("seed terminal disposition")
    if type(predicates) is not dict or set(predicates) != set(SEED_COUNT_FIELDS + SEED_MAX_FIELDS):
        raise ValueError("seed predicate fields")
    nodes = predicates["node_count"]
    if type(nodes) is not int or not 0 <= nodes <= 16777216:
        raise ValueError("seed node count")
    if any(type(predicates[name]) is not int or not 0 <= predicates[name] <= nodes for name in SEED_COUNT_FIELDS):
        raise ValueError("seed predicate count")
    if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value))
           for value in (predicates[name] for name in SEED_MAX_FIELDS)):
        raise ValueError("seed predicate maximum")
    if (predicates["side_match_count"] != predicates["after_side_count"]
            or not predicates["after_side_count"] >= predicates["after_score_count"] >= predicates["after_margin_count"] >= predicates["after_support_count"]
            or predicates["after_support_count"] != row["eligible_seed_count"]):
        raise ValueError("seed predicate chain")
    if (row["disposition"] == "available") != (row["eligible_seed_count"] > 0):
        raise ValueError("seed predicate disposition")


def _attrition_account(record):
    """Strict scalar joins recomputed on readback, with no production replay."""
    expected_keys = {"schema", "status", "frames", "limits", "observer_incomplete", "omitted",
                     "stages", "handoffs", "returned_readings", "checks", "lineage", "counts"}
    if type(record) is not dict or set(record) != expected_keys or record["schema"] != ATTRITION_SCHEMA:
        raise ValueError("attrition envelope")
    limits = record["limits"]
    if (type(limits) is not dict or set(limits) != {"items", "bytes"}
            or any(type(limits[k]) is not int or not 1 <= limits[k] <= cap for k, cap in (("items", ATTRITION_ITEMS), ("bytes", ATTRITION_BYTES)))
            or type(record["observer_incomplete"]) is not bool or type(record["omitted"]) is not int or record["omitted"] < 0):
        raise ValueError("attrition limits")
    frames = record["frames"]
    if type(frames) is not list or not 1 <= len(frames) <= 32 or len(set(frames)) != len(frames):
        raise ValueError("attrition frames")
    if any(type(f) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", f) for f in frames):
        raise ValueError("attrition frame label")
    if type(record["stages"]) is not dict or set(record["stages"]) != set(ATTRITION_STAGES):
        raise ValueError("attrition stages")
    if any(type(record[k]) is not list or len(record[k]) > limits["items"] for k in ("lineage", "counts")):
        raise ValueError("attrition summary cap")
    # Charge the entire transported envelope, including derived summaries.
    if len(canonical_bytes(record)) > limits["bytes"]:
        raise ValueError("attrition full envelope byte cap")
    checks, rows, hist = {}, {}, {}
    total = 0
    for stage in ATTRITION_STAGES:
        data = record["stages"][stage]
        if type(data) is not dict or set(data) != {"expected", "terminal", "duplicates", "unexpected"}:
            raise ValueError("attrition stage fields")
        if any(type(data[k]) is not list or len(data[k]) > limits["items"] for k in ("expected", "terminal")):
            raise ValueError("attrition stage cap")
        if any(type(data[k]) is not int or data[k] < 0 for k in ("duplicates", "unexpected")):
            raise ValueError("attrition stage counters")
        expected = set()
        for identity in data["expected"]:
            _attrition_row(stage, dict(schema=ATTRITION_ROW_SCHEMA, identity=identity, disposition="pending", axis=None, side=None, plane_index=None, eligible_seed_count=None, seed_predicates=None))
            key = tuple(identity)
            if key[0] not in frames or key in expected:
                raise ValueError("attrition expected identity")
            expected.add(key)
        rows[stage] = {}
        for row in data["terminal"]:
            _attrition_row(stage, row)
            if row["disposition"] not in ATTRITION_DISPOSITIONS[stage]:
                raise ValueError("attrition terminal disposition")
            key = tuple(row["identity"])
            if key[0] not in frames or key in rows[stage]:
                raise ValueError("attrition terminal identity")
            rows[stage][key] = row
            bucket = stage, key[0], row["axis"], row["side"], row["disposition"]
            hist[bucket] = hist.get(bucket, 0) + 1
        checks[stage + "_conservation"] = expected == set(rows[stage]) and not data["duplicates"] and not data["unexpected"]
        total += len(data["expected"]) + len(data["terminal"])
    for key in ("handoffs", "returned_readings"):
        if type(record[key]) is not list or len(record[key]) > limits["items"]:
            raise ValueError("attrition handoff cap")
        total += len(record[key])
    if total > limits["items"] or len(canonical_bytes({k: record[k] for k in ("frames", "stages", "handoffs", "returned_readings")})) > limits["bytes"]:
        raise ValueError("attrition transport cap")
    lineage, seen_generated, seen_returned = [], set(), set()
    checks["seed_axis_identity"] = all(row["axis"] == key[1] and row["side"] == key[2] for key, row in rows["seed"].items())
    checks["seed_census"] = set(rows["seed"]) == {(f, a, side) for f in frames for a in range(3) for side in (-1, 1)}
    checks["seed_node_count_consistency"] = all(len({row["seed_predicates"]["node_count"]
        for key, row in rows["seed"].items() if key[0] == frame and row["seed_predicates"] is not None}) <= 1 for frame in frames)
    for handoff in record["handoffs"]:
        if type(handoff) is not dict or set(handoff) != {"generated", "returned"}:
            raise ValueError("attrition handoff fields")
        for identity in handoff.values():
            _attrition_row("proposal", dict(schema=ATTRITION_ROW_SCHEMA, identity=identity, disposition="pending", axis=None, side=None, plane_index=None, eligible_seed_count=None, seed_predicates=None))
        generated, returned = tuple(handoff["generated"]), tuple(handoff["returned"])
        if generated in seen_generated or returned in seen_returned or generated[0] != returned[0] or generated not in rows["proposal"]:
            raise ValueError("attrition handoff join")
        seen_generated.add(generated); seen_returned.add(returned)
        proposal, reading = rows["proposal"][generated], rows["reading"].get(returned)
        if proposal["disposition"] != "returned":
            raise ValueError("attrition nonreturned handoff")
        lineage.append(dict(frame=generated[0], generated=generated[1], returned=returned[1],
                            proposal_axis=proposal["axis"], proposal_side=proposal["side"],
                            reading_axis=None if reading is None else reading["axis"],
                            reading_side=None if reading is None else reading["side"],
                            disposition="pending" if reading is None else reading["disposition"],
                            plane_index=None if reading is None else reading["plane_index"]))
    checks["proposal_handoff"] = seen_generated == {k for k, r in rows["proposal"].items() if r["disposition"] == "returned"}
    checks["reading_handoff"] = seen_returned == {tuple(i) for i in record["stages"]["reading"]["expected"]}
    # A reader may rederive ray side. Local normal axis must be conserved.
    checks["axis_lineage"] = all(r["proposal_axis"] == r["reading_axis"] for r in lineage)
    output = {}
    for row in record["returned_readings"]:
        if type(row) is not dict or set(row) != {"frame", "plane", "axis"}:
            raise ValueError("attrition returned reading fields")
        _attrition_row("reading", dict(schema=ATTRITION_ROW_SCHEMA, identity=[row["frame"], row["plane"]], disposition="accepted", axis=row["axis"], side=None, plane_index=row["plane"], eligible_seed_count=None, seed_predicates=None))
        if row["axis"] is None:
            raise ValueError("attrition returned reading axis")
        key = row["frame"], row["plane"]
        if key[0] not in frames or key in output:
            raise ValueError("attrition returned reading identity")
        output[key] = row["axis"]
    accepted = {(r["frame"], r["plane_index"]): r["reading_axis"] for r in lineage if r["disposition"] == "accepted"}
    checks["accepted_reading_join"] = accepted == output and len(accepted) == sum(r["disposition"] == "accepted" for r in lineage)
    counts = [dict(stage=k[0], frame=k[1], axis=k[2], side=k[3], disposition=k[4], count=v)
              for k, v in sorted(hist.items(), key=lambda item: repr(item[0]))]
    lineage.sort(key=lambda r: (r["frame"], r["generated"]))
    status = "complete" if all(checks.values()) and not record["observer_incomplete"] and not record["omitted"] else "incomplete"
    return checks, lineage, counts, status


def _reading_attrition_record(observer, capture):
    try:
        frames = list(capture.graph.frame_ids)
        readings = capture.arguments[1]["lattice_result"].readings
        returned = []
        for reading in readings:
            binding = observer.bindings.get(("reading", id(reading)))
            if binding is None:
                observer.incomplete = True
                continue
            returned.append(dict(frame=binding[0], plane=binding[1], axis=int(reading.axis)))
        record = dict(schema=ATTRITION_SCHEMA, status="incomplete", frames=frames,
                      limits=dict(items=observer.max_items, bytes=observer.max_bytes),
                      observer_incomplete=observer.incomplete, omitted=observer.omitted,
                      stages={}, handoffs=[], returned_readings=returned, checks={}, lineage=[], counts=[])
        for stage in ATTRITION_STAGES:
            data = observer._stage(stage)
            record["stages"][stage] = dict(expected=[list(i) for i in data["expected"].values()],
                terminal=list(observer.scalar_rows[stage].values()), duplicates=data["duplicates"], unexpected=data["unexpected"])
        record["handoffs"] = [dict(generated=list(key[1:]), returned=list(value)) for key, value in observer.bindings.items() if key[0] == "proposal_return"]
        record["checks"], record["lineage"], record["counts"], record["status"] = _attrition_account(record)
        # Summaries contribute bytes too; enforce again after materializing them.
        _attrition_account(record)
        return record
    except Exception:
        # Observer projection failure is diagnostic only; no raw exception text.
        return None


def canonical_bytes(value):
    """Precharge exact ASCII-escaped JSON bytes before entering the encoder."""
    stack, count, size = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > MAX_DEPTH or count > MAX_ITEMS:
            raise DiagnosticError("report_budget", "structure")
        if type(item) in (dict, tuple, list):
            if count + len(stack) + len(item) > MAX_ITEMS:
                raise DiagnosticError("report_budget", "items")
            if type(item) is dict:
                if any(type(k) is not str for k in item):
                    raise DiagnosticError("malformed_report", "keys")
                for key in item:
                    size += _escaped_string_size(key) + 1  # quoted key and colon
                    if size > MAX_REPORT_BYTES:
                        raise DiagnosticError("report_budget", "bytes")
                children = item.values()
            else:
                children = item
            stack.extend((child, depth + 1) for child in children)
            size += 2 + max(0, len(item) - 1)  # braces/brackets and commas
        elif type(item) is str:
            size += _escaped_string_size(item)
        elif type(item) is int:
            if not -(2**63) <= item < 2**63:
                raise DiagnosticError("report_budget", "integer")
            size += len(str(item))
        elif type(item) is float:
            if not math.isfinite(item):
                raise DiagnosticError("malformed_report", "number")
            size += len(repr(item))
        elif type(item) is bool:
            size += 4 if item else 5
        elif item is None:
            size += 4
        else:
            raise DiagnosticError("malformed_report", "non_scalar")
        if size > MAX_REPORT_BYTES:
            raise DiagnosticError("report_budget", "bytes")
    data = bytearray()
    try:
        for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).iterencode(value):
            encoded = chunk.encode("utf-8")
            if len(data) + len(encoded) > MAX_REPORT_BYTES:
                raise DiagnosticError("report_budget", "bytes")
            data.extend(encoded)
    except (ValueError, OverflowError) as error:
        raise DiagnosticError("malformed_report", "number") from error
    return bytes(data)


def _escaped_string_size(value):
    size = 2  # opening and closing quotes
    for char in value:
        code = ord(char)
        size += (2 if char in '\\"\b\f\n\r\t' else
                 6 if code < 32 or 126 < code <= 65535 else 12 if code > 65535 else 1)
        if size > MAX_REPORT_BYTES:
            raise DiagnosticError("report_budget", "escaped_string")
    return size


def _json_preflight(raw, byte_limit):
    """Bound JSON lexical depth/tokens before any decoder or constructor.

    This is a resource preflight, not a replacement JSON parser. The existing
    decoder/record constructors still validate grammar and proof semantics.
    Keys count as tokens too, conservatively bounding parser allocations.
    """
    if type(raw) is bytes:
        if len(raw) > byte_limit:
            raise DiagnosticError("json_preflight", "bytes")
        try:
            text = raw.decode("utf-8")
        except UnicodeError as error:
            raise DiagnosticError("json_preflight", "utf8") from error
    elif type(raw) is str:
        text = raw
        if len(text) > byte_limit:
            raise DiagnosticError("json_preflight", "bytes")
    else:
        raise DiagnosticError("json_preflight", "type")
    stack, tokens, bytes_used = [], 0, 0
    quoted = escaped = scalar = False
    for char in text:
        code = ord(char)
        if 0xD800 <= code <= 0xDFFF:
            raise DiagnosticError("json_preflight", "utf8")
        bytes_used += 1 if code < 128 else 2 if code < 2048 else 3 if code < 65536 else 4
        if bytes_used > byte_limit:
            raise DiagnosticError("json_preflight", "bytes")
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char in " \t\r\n":
            scalar = False
            continue
        if char in ",:":
            scalar = False
            continue
        if char in "]}":
            scalar = False
            if not stack or stack.pop() != ("[" if char == "]" else "{"):
                raise DiagnosticError("json_preflight", "structure")
            continue
        if char in '[{"' or not scalar:
            tokens += 1
            if tokens > MAX_ITEMS:
                raise DiagnosticError("json_preflight", "items")
            if len(stack) > MAX_DEPTH:
                raise DiagnosticError("json_preflight", "depth")
        if char in "[{":
            stack.append(char)
            scalar = False
        elif char == '"':
            quoted = True
            scalar = False
        else:
            scalar = True
    if quoted or stack:
        raise DiagnosticError("json_preflight", "structure")


def _sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(report):
    report["result_sha256"] = _sha(report)
    return report


def _certificate_record(cert):
    payload = selection._payload(cert)
    payload.pop("proof_json")
    payload["proof"] = cert.proof_record()
    payload["result_sha256"] = cert.result_sha256
    return payload


def _certificate_from_record(record):
    fields = {f.name for f in dataclasses.fields(selection.EndpointSelectionCertificate)} - {"proof_json"} | {"proof"}
    if type(record) is not dict or set(record) != fields:
        raise ValueError("certificate fields differ")
    if type(record["reading_bindings"]) not in (tuple, list) or len(record["reading_bindings"]) > 128:
        raise ValueError("reading bindings exceed bound")
    if type(record["representatives"]) not in (tuple, list) or len(record["representatives"]) > 4096:
        raise ValueError("representatives exceed bound")
    # JSON arrays must regain the exact immutable source-certificate field.
    # Bound and validate before copying; never invent absent dispositions.
    dispositions = record["candidate_dispositions"]
    if (type(record["candidate_count"]) is not int or not 0 <= record["candidate_count"] <= 16384
            or type(dispositions) not in (tuple, list)
            or len(dispositions) != record["candidate_count"]
            or any(type(d) is not str or d not in ("admitted", plane_graph.SAME_AXIS_DISPOSITION_KEPT,
                                                  plane_graph.SAME_AXIS_DISPOSITION_DROPPED)
                   for d in dispositions)):
        raise ValueError("candidate dispositions shape/count/value differs")
    bindings = []
    for row in record["reading_bindings"]:
        if len(row) != 6 or len(row[4]) > 16 or len(row[5]) > 16:
            raise ValueError("reading binding shape/count differs")
        bindings.append((tuple(row[0]), row[1], row[2], row[3], tuple(tuple(a) for a in row[4]), tuple(row[5])))
    representatives = []
    for index, source, payload in record["representatives"]:
        if type(payload) is not list or len(payload) != 16:
            raise ValueError("representative match fields differ")
        values = list(payload)
        values[7] = legacy._psr_from_payload(values[7])
        values[15] = None if values[15] is None else tuple(values[15])
        match = plane_graph.PlaneMatch(*values)
        if legacy._match_payload(match) != payload:
            raise ValueError("representative match is not canonical")
        representatives.append(selection.RepresentativeConstraint(index, source, match))
    kwargs = {k: v for k, v in record.items() if k not in ("proof", "reading_bindings", "representatives", "budget", "frame_ids", "quadrant_constraints")}
    kwargs.update(candidate_dispositions=tuple(dispositions),
                  reading_bindings=tuple(bindings), representatives=tuple(representatives),
                  frame_ids=tuple(record["frame_ids"]), quadrant_constraints=tuple(tuple(q) for q in record["quadrant_constraints"]),
                  budget=csp.EndpointBudget(**record["budget"]), proof_json=selection._encode(record["proof"]).decode("utf-8"))
    return selection.EndpointSelectionCertificate(**kwargs)


def _validate_graph_record(graph, status):
    common = {"kind", "schema", "producer", "result_sha256", "status", "reason", "detail", "bindings", "frame_ids",
              "counts", "correlation_algorithm", "max_shift_cells", "geometry_evaluations", "geometry_settings", "selection"}
    if type(graph) is not dict or graph.get("kind") not in ("failure", "no_failure"):
        raise ValueError("graph record is missing")
    required = common | ({"per_frame_local_axis", "endpoint_incidence", "proof_scope"} if graph["kind"] == "failure" else set())
    if set(graph) != required or graph["schema"] != graph_mod.IMAGE_DERIVED_GRAPH_SCHEMA or graph["producer"] != graph_mod.IMAGE_DERIVED_GRAPH_PRODUCER:
        raise ValueError("graph fields/identity differ")
    if (not selection._hex(graph["result_sha256"]) or graph["status"] not in ("assembled", "unavailable")
            or graph["correlation_algorithm"] != plane_graph.PLANE_CORRELATION_ALGORITHM
            or graph["max_shift_cells"] != 32 or list(graph["geometry_settings"]) != [0.15, 0.90, 0.02]
            or type(graph["geometry_evaluations"]) is not int or graph["geometry_evaluations"] not in (0, 1)):
        raise ValueError("graph digest/settings differ")
    bindings = graph["bindings"]
    if type(bindings) is not dict or set(bindings) != {"session_sha256", "lattice_sha256", "signatures_sha256", "connection_sha256"}:
        raise ValueError("graph source fields differ")
    if any(not selection._hex(v) for k, v in bindings.items() if k != "connection_sha256") or bindings["connection_sha256"] != "" and not selection._hex(bindings["connection_sha256"]):
        raise ValueError("graph source digests differ")
    selection._frames(tuple(graph["frame_ids"]))
    counts = graph["counts"]
    if type(counts) is not dict or set(counts) != {"readings", "signatures", "candidates", "selected"}:
        raise ValueError("graph counts differ")
    if any(type(counts[k]) is not int or not 0 <= counts[k] <= cap for k, cap in (("readings",128),("signatures",2048),("candidates",16384),("selected",4096))):
        raise ValueError("graph count budget differs")
    if graph["kind"] == "no_failure":
        cert = graph["selection"]
        if (status != "no_failure" or graph["status"] != "assembled" or graph["reason"] or graph["detail"]
                or graph["geometry_evaluations"] != 1 or type(cert) is not dict
                or set(cert) != {"schema", "producer", "status", "result_sha256"}
                or cert["schema"] != selection.ENDPOINT_SELECTION_SCHEMA or cert["producer"] != selection.ENDPOINT_SELECTION_PRODUCER
                or cert["status"] != "unique" or not selection._hex(cert["result_sha256"])):
            raise ValueError("no_failure summary is inconsistent")
        return
    cert = _certificate_from_record(graph["selection"])
    if any(getattr(cert,k) != v for k,v in bindings.items()) or cert.frame_ids != tuple(graph["frame_ids"]):
        raise ValueError("certificate bindings differ")
    if (counts["readings"] != len(cert.reading_bindings) or counts["signatures"] != sum(len(r[5]) for r in cert.reading_bindings)
            or counts["candidates"] != cert.candidate_count or counts["selected"] != 0
            or graph["status"] != "unavailable" or not graph["reason"]):
        raise ValueError("failure graph counts/status differ")
    if graph["proof_scope"] != ("partial" if cert.status == "budget_refused" else "observed"):
        raise ValueError("proof scope differs")
    if cert.status == "budget_refused" and status != "refused" or cert.status != "budget_refused" and status != "complete":
        raise ValueError("diagnostic disposition differs")
    if cert.status != "unique" and (graph["reason"] != cert.reason or graph["detail"] != cert.status):
        raise ValueError("nonunique graph reason/status does not join selection")
    if graph["geometry_evaluations"] != int(cert.status == "unique"):
        raise ValueError("evaluator count differs")
    endpoints = graph["endpoint_incidence"]
    if type(endpoints) is not list or len(endpoints) != len(cert.reading_bindings):
        raise ValueError("endpoint count differs")
    axes = {}
    for row,binding in zip(endpoints,cert.reading_bindings):
        if set(row) != {"frame","plane","local_axis","alternatives","signatures","candidate_incidence","selected_incidence"}:
            raise ValueError("endpoint fields differ")
        if (row["frame"],row["plane"],row["local_axis"],row["alternatives"],row["signatures"]) != (*binding[0],binding[2],len(binding[4]),len(binding[5])):
            raise ValueError("endpoint bindings differ")
        if any(type(row[k]) is not int or not 0 <= row[k] <= 16384 for k in ("candidate_incidence","selected_incidence")):
            raise ValueError("endpoint incidence differs")
        key=(row["frame"],row["local_axis"])
        axis=axes.setdefault(key,dict(frame=key[0],local_axis=key[1],readings=0,alternatives=0,signatures=0,candidate_incidence=0,selected_incidence=0))
        axis["readings"]+=1
        for k in ("alternatives","signatures","candidate_incidence","selected_incidence"): axis[k]+=row[k]
    if (sum(e["candidate_incidence"] for e in endpoints) != 2*counts["candidates"]
            or sum(e["selected_incidence"] for e in endpoints) != 2*counts["selected"]
            or graph["per_frame_local_axis"] != [axes[k] for k in sorted(axes)]):
        raise ValueError("incidence accounting does not close")


def validate_report(report):
    keys = {"schema", "producer", "status", "reason", "field", "g_calls", "graph", "run", "notes", "result_sha256"}
    if type(report) is dict and "reading_attrition" in report:
        keys.add("reading_attrition")
    if type(report) is dict and "support_score_attrition" in report:
        keys.add("support_score_attrition")
    if type(report) is dict and "paired_segment_support" in report:
        keys.add("paired_segment_support")
    if type(report) is dict and "unit_weight_sparse_support" in report:
        keys.add("unit_weight_sparse_support")
    if type(report) is dict and "support_line_weight_policy" in report:
        keys.add("support_line_weight_policy")
    if (type(report) is not dict or set(report) != keys or report["schema"] != SCHEMA
            or report["producer"] != PRODUCER or report["status"] not in ("complete", "no_failure", "incomplete", "refused")
            or type(report["reason"]) is not str or type(report["field"]) is not str
            or type(report["g_calls"]) is not int or report["g_calls"] < 0
            or type(report["run"]) is not dict or type(report["notes"]) is not list):
        raise DiagnosticError("malformed_report", "envelope")
    canonical_bytes(report)
    body = {k: v for k, v in report.items() if k != "result_sha256"}
    if report["result_sha256"] != _sha(body):
        raise DiagnosticError("malformed_report", "digest")
    if "support_line_weight_policy" in report:
        policy = report["support_line_weight_policy"]
        if type(policy) is not str or policy not in ("structural", "unit"):
            raise DiagnosticError("malformed_report", "support_line_weight_policy")
        if policy == "unit" and "unit_weight_sparse_support" in report:
            raise DiagnosticError("malformed_report", "unit_policy_counterfactual_combination")
    if "reading_attrition" in report:
        attrition = report["reading_attrition"]
        try:
            if attrition is not None:
                checks, lineage, counts, status = _attrition_account(attrition)
                if (attrition["checks"], attrition["lineage"], attrition["counts"], attrition["status"]) != (checks, lineage, counts, status):
                    raise ValueError("attrition computed fields differ")
                if report["graph"] is not None and attrition["frames"] != list(report["graph"]["frame_ids"]):
                    raise ValueError("attrition graph frames differ")
                if report["graph"] is not None and report["graph"]["kind"] == "failure" and attrition["status"] == "complete":
                    actual = sorted((r["frame"], r["plane"], r["axis"]) for r in attrition["returned_readings"])
                    graph_rows = sorted((r["frame"], r["plane"], r["local_axis"]) for r in report["graph"]["endpoint_incidence"])
                    if actual != graph_rows:
                        raise ValueError("attrition graph reading axes differ")
            if attrition is None or attrition["status"] == "incomplete":
                if report["status"] != "incomplete" or report["reason"] not in ("reading_attrition_incomplete", "support_score_attrition_incomplete", "paired_segment_support_incomplete", "unit_weight_sparse_support_incomplete") or report["field"] != "observer":
                    raise ValueError("attrition incomplete disposition")
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            raise DiagnosticError("malformed_report", "reading_attrition") from error
    if "support_score_attrition" in report:
        try:
            support = report["support_score_attrition"]
            if "reading_attrition" not in report:
                raise ValueError("support missing seed source")
            if support is not None:
                checks, status = _support_score_account(support, report["reading_attrition"])
                if (support["checks"], support["status"]) != (checks, status):
                    raise ValueError("support computed fields differ")
            if support is None or support["status"] == "incomplete":
                if report["status"] != "incomplete" or report["reason"] not in ("reading_attrition_incomplete", "support_score_attrition_incomplete", "paired_segment_support_incomplete", "unit_weight_sparse_support_incomplete") or report["field"] != "observer":
                    raise ValueError("support incomplete disposition")
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            raise DiagnosticError("malformed_report", "support_score_attrition") from error
    if "paired_segment_support" in report and "unit_weight_sparse_support" in report:
        raise DiagnosticError("malformed_report", "counterfactual_modes_mixed")
    for pair_field in ("paired_segment_support", "unit_weight_sparse_support"):
        if pair_field not in report:
            continue
        try:
            pair = report[pair_field]
            if "reading_attrition" not in report:
                raise ValueError("paired missing source")
            if pair is not None:
                checks,status=_pair_account(pair,report["reading_attrition"],
                                           unit_weight_sparse=pair_field == "unit_weight_sparse_support")
                if (pair["checks"],pair["status"]) != (checks,status):
                    raise ValueError("paired computed fields")
                if report["graph"] is not None and pair["frames"] != list(report["graph"]["frame_ids"]):
                    raise ValueError("paired graph frame census")
                combined={name:report[name] for name in ("reading_attrition","support_score_attrition",pair_field) if name in report}
                if len(canonical_bytes(pair))>PAIR_BYTES or len(canonical_bytes(combined))>ATTRITION_BYTES:
                    raise ValueError("paired report cap")
            if pair is None or pair["status"] != "complete":
                if (report["status"],report["reason"],report["field"]) != ("incomplete",pair_field+"_incomplete","observer"):
                    raise ValueError("paired incomplete disposition")
        except (ValueError,TypeError,KeyError,IndexError,AttributeError) as error:
            raise DiagnosticError("malformed_report",pair_field) from error
    if report["status"] in ("complete", "no_failure") and (report["g_calls"] != 1 or type(report["graph"]) is not dict or report["reason"]):
        raise DiagnosticError("malformed_report", "completion")
    if report["graph"] is not None:
        if report["g_calls"] != 1:
            raise DiagnosticError("malformed_report", "graph_call_count")
        try:
            graph_status = report["status"]
            if report["reason"] in ("reading_attrition_incomplete", "support_score_attrition_incomplete", "paired_segment_support_incomplete", "unit_weight_sparse_support_incomplete"):
                graph_status = ("no_failure" if report["graph"]["kind"] == "no_failure" else
                                "refused" if report["graph"]["selection"]["status"] == "budget_refused" else "complete")
            _validate_graph_record(report["graph"], graph_status)
            _validate_run_record(report["run"])
            if report["graph"]["kind"] == "failure" and report["graph"]["selection"]["status"] == "budget_refused":
                if report["reason"] not in ("reading_attrition_incomplete", "support_score_attrition_incomplete", "paired_segment_support_incomplete", "unit_weight_sparse_support_incomplete") and (report["status"],report["reason"],report["field"]) != ("refused","selection_budget_refused","budget"):
                    raise ValueError("budget refusal outer envelope differs")
        except (ValueError,TypeError,KeyError,IndexError,AttributeError) as error:
            raise DiagnosticError("malformed_report", "graph_proof") from error
    elif report["status"] in ("complete","no_failure"):
        raise DiagnosticError("malformed_report", "graph_missing")
    return report


def _validate_run_record(run):
    if set(run) != {"inputs","runtime","production","outputs","elapsed_seconds","inputs_unchanged_after"} or run["inputs_unchanged_after"] is not True:
        raise ValueError("run fields/input rehash differ")
    inputs = run["inputs"]
    if type(inputs) is not list or not 1 <= len(inputs) <= 32:
        raise ValueError("input count differs")
    for i,row in enumerate(inputs):
        if (type(row) is not dict or set(row) != {"index","path","sha256"} or type(row["index"]) is not int or row["index"] != i
                or type(row["path"]) is not str or not row["path"] or not selection._hex(row["sha256"])):
            raise ValueError("ordered input binding differs")
    runtime=run["runtime"]
    if (type(runtime) is not dict or set(runtime) != {"python","source_sha256","pin_coverage","runner_sha256"}
            or runtime["source_sha256"] != SOURCE_PINS or not selection._hex(runtime["runner_sha256"])
            or type(runtime["python"]) is not str or type(runtime["pin_coverage"]) is not str):
        raise ValueError("runtime fields/pins differ")
    if type(run["elapsed_seconds"]) is not float or run["elapsed_seconds"] < 0:
        raise ValueError("runtime duration differs")
    if type(run["production"]) is not dict or set(run["production"]) != {"schema","producer","status","reason","result_sha256","production_sha256","delivery_sha256","refusal_sha256"}:
        raise ValueError("production fields differ")
    if any(v is not None and type(v) is not str for v in run["production"].values()):
        raise ValueError("production must be scalar")
    outputs=run["outputs"]
    if type(outputs) is not list or len(outputs) != 2:
        raise ValueError("terminal outputs differ")
    for row,kind in zip(outputs,("mcscan","litematic")):
        if (type(row) is not dict or set(row) != {"kind","path","exists","sha256","size_bytes"} or row["kind"] != kind
                or type(row["path"]) is not str or type(row["exists"]) is not bool
                or (row["exists"] and (not selection._hex(row["sha256"]) or type(row["size_bytes"]) is not int or row["size_bytes"] < 0))
                or (not row["exists"] and (row["sha256"] is not None or row["size_bytes"] is not None))):
            raise ValueError("terminal output binding differs")


def _support_line_weight_policy(policy):
    if type(policy) is not str or policy not in ("structural", "unit"):
        raise DiagnosticError("invalid_input", "support_line_weight_policy")
    return policy


class GraphCapture:
    """A temporary observation of the exact orchestration callable."""
    def __init__(self, reading_attrition=None, *, support_line_weight_policy="structural"):
        self.calls = 0
        self.graph = None
        self.arguments = None
        self.error = None
        self.reading_attrition = reading_attrition
        self.support_line_weight_policy = _support_line_weight_policy(support_line_weight_policy)
        if (self.support_line_weight_policy == "unit" and reading_attrition is not None
                and reading_attrition.paired is not None and reading_attrition.paired.unit_weight_sparse):
            raise DiagnosticError("invalid_input", "unit_policy_counterfactual_combination")

    def invoke(self, delivery, *args, **kwargs):
        original = orchestration.g_mod.derive_image_derived_graph
        original_measure = lattice_measure._measure_frame
        original_policy_build = region_growth._build_graph

        def policy_build(*build_args, **build_kwargs):
            incoming = _support_line_weight_policy(build_kwargs.get("_support_line_weight_policy", "structural"))
            if self.support_line_weight_policy == "structural" and incoming != "structural":
                raise DiagnosticError("invalid_input", "foreign_support_line_weight_policy")
            # Only the temporary evaluation seam can override ordinary proposal
            # calls. Config/schema, source arguments and inference gates stay fixed.
            if self.support_line_weight_policy == "unit":
                build_kwargs["_support_line_weight_policy"] = "unit"
            return original_policy_build(*build_args, **build_kwargs)

        def observed_measure(*measure_args, **measure_kwargs):
            # The existing reader call is the only observer scope. Signature,
            # matcher and certificate validation work retain their original context.
            observer = self.reading_attrition
            frame = None
            try:
                frame_object = measure_args[0] if measure_args else measure_kwargs.get("frame")
                frame = getattr(frame_object, "frame_id", None)
                if type(frame) is not str:
                    frame = None
                observer.support_begin(frame)
                if observer.paired is not None:
                    observer.paired.enter_frame(frame)
            except Exception:
                observer.support_enabled = True
                observer.support_incomplete = True
            original_build = region_growth._build_graph
            original_support = region_growth._segment_support

            def observed_support(*support_args, **support_kwargs):
                pair=observer.paired
                try:
                    if support_kwargs.get("_support_observer") is not None:
                        pair.discard("foreign_sink")
                    else:
                        support_kwargs["_support_observer"]=pair
                    result=original_support(*support_args,**support_kwargs)
                    try:
                        pair.returned(result)
                    except Exception:
                        pair.discard("return_failure")
                    return result
                except BaseException:
                    pair.discard("source_allocation_exception")
                    raise

            def observed_build(*build_args, **build_kwargs):
                result = original_build(*build_args, **build_kwargs)
                try:
                    if region_growth._attrition_state.get() is not observer:
                        raise ValueError("foreign support observer")
                    observer.support_graph(frame, result)
                    if observer.paired is not None:
                        observer.paired.graph(result,build_args,build_kwargs)
                except Exception:
                    observer.support_incomplete = True
                    if observer.paired is not None:
                        observer.paired.discard("graph_failure")
                return result

            region_growth._build_graph = observed_build
            if observer.paired is not None:
                region_growth._segment_support = observed_support
            failed = True
            try:
                with region_growth._scalar_attrition_context(observer):
                    result = original_measure(*measure_args, **measure_kwargs)
                failed = False
                return result
            finally:
                region_growth._build_graph = original_build
                if observer.paired is not None:
                    region_growth._segment_support = original_support
                    observer.paired.finish_frame()
                if failed:
                    observer.support_incomplete = True
                    if frame in observer.support_frames and observer.support_frames[frame]["rows"] is None:
                        observer.support_frames[frame]["state"] = "failed"

        def wrapped(*graph_args, **graph_kwargs):
            self.calls += 1
            if self.calls == 1:
                self.arguments = (graph_args, graph_kwargs)
            try:
                result = original(*graph_args, **graph_kwargs)
            except Exception as error:
                self.error = error
                raise
            if self.calls == 1:
                self.graph = result
            return result

        orchestration.g_mod.derive_image_derived_graph = wrapped
        region_growth._build_graph = policy_build if self.support_line_weight_policy == "unit" else original_policy_build
        if self.reading_attrition is not None:
            lattice_measure._measure_frame = observed_measure
        try:
            return delivery(*args, **kwargs)
        finally:
            orchestration.g_mod.derive_image_derived_graph = original
            region_growth._build_graph = original_policy_build
            if self.reading_attrition is not None:
                lattice_measure._measure_frame = original_measure


def _graph_record(capture):
    graph = capture.graph
    if type(graph) is not graph_mod.ImageDerivedGraphResult:
        raise DiagnosticError("foreign_or_upstream_refusal", "graph")
    cert = graph.selection
    if type(cert) is not selection.EndpointSelectionCertificate:
        raise DiagnosticError("foreign_proof", "selection")
    # Before validators, traversal, or reconstructing any nested records.
    if type(cert.proof_json) is not str or len(cert.proof_json) > MAX_PROOF_BYTES or len(cert.proof_json.encode("utf-8")) > MAX_PROOF_BYTES:
        raise DiagnosticError("proof_budget", "proof_json")
    _json_preflight(cert.proof_json, MAX_PROOF_BYTES)
    for value, cap, field in ((graph.frame_ids, 32, "frames"), (graph.readings, 128, "readings"),
                              (graph.signatures, 2048, "signatures"), (graph.candidate_matches, 16384, "candidates"),
                              (graph.matches, 4096, "selected")):
        if type(value) is not tuple or len(value) > cap:
            raise DiagnosticError("proof_budget", field)
    try:
        # Constructors validate scalar identities, digests and source-row joins.
        # These calls never replay the matcher, selector or numerical evaluator.
        graph_mod.ImageDerivedGraphResult(**{f.name: getattr(graph, f.name) for f in dataclasses.fields(graph)})
        _, arguments = capture.arguments
        expected = {
            "session_sha256": arguments["session"].session_sha256,
            "lattice_sha256": arguments["lattice_result"].result_sha256,
            "signatures_sha256": arguments["signatures_result"].result_sha256,
            "connection_sha256": "" if arguments.get("connection_graph") is None else arguments["connection_graph"].graph_sha256,
        }
        if any(getattr(graph, name) != digest or getattr(cert, name) != digest for name, digest in expected.items()):
            raise DiagnosticError("stale_binding", "sources")
        if graph.readings is not arguments["lattice_result"].readings or graph.signatures is not arguments["signatures_result"].signatures:
            raise DiagnosticError("stale_binding", "source_universe")
    except DiagnosticError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, RecursionError) as error:
        raise DiagnosticError("malformed_or_foreign_proof", type(error).__name__) from error
    proof = cert.proof_record()
    endpoints = {tuple(row[0]): {"frame": row[0][0], "plane": row[0][1], "local_axis": row[2],
                 "alternatives": len(row[4]), "signatures": len(row[5]), "candidate_incidence": 0,
                 "selected_incidence": 0} for row in cert.reading_bindings}
    for rows, field in ((graph.candidate_matches, "candidate_incidence"), (graph.matches, "selected_incidence")):
        for row in rows:
            endpoints[(row.left, row.left_plane)][field] += 1
            endpoints[(row.right, row.right_plane)][field] += 1
    axes = {}
    for endpoint in endpoints.values():
        key = endpoint["frame"], endpoint["local_axis"]
        row = axes.setdefault(key, dict(frame=key[0], local_axis=key[1], readings=0, alternatives=0,
                                       signatures=0, candidate_incidence=0, selected_incidence=0))
        row["readings"] += 1
        for name in ("alternatives", "signatures", "candidate_incidence", "selected_incidence"):
            row[name] += endpoint[name]
    common = {
        "kind": "failure" if graph.status != "assembled" else "no_failure",
        "schema": graph.schema, "producer": graph.producer, "result_sha256": graph.result_sha256,
        "status": graph.status, "reason": graph.reason, "detail": graph.detail, "bindings": expected,
        "frame_ids": graph.frame_ids, "counts": dict(readings=len(graph.readings), signatures=len(graph.signatures),
                candidates=len(graph.candidate_matches), selected=len(graph.matches)),
        "correlation_algorithm": graph.correlation_algorithm, "max_shift_cells": graph.max_shift_cells,
        "geometry_evaluations": graph.geometry_evaluations, "geometry_settings": graph.geometry_settings,
    }
    if graph.status == "assembled":
        common["selection"] = {k:getattr(cert,k) for k in ("schema","producer","status","result_sha256")}
    else:
        common.update(per_frame_local_axis=[axes[k] for k in sorted(axes)],
                      endpoint_incidence=[endpoints[k] for k in sorted(endpoints)],
                      proof_scope="partial" if cert.status == "budget_refused" else "observed",
                      selection=_certificate_record(cert))
    return common


def build_report(capture, run, production_error=None):
    report = dict(schema=SCHEMA, producer=PRODUCER, status="incomplete", reason="", field="", g_calls=capture.calls,
                  graph=None, run=run, support_line_weight_policy=capture.support_line_weight_policy, notes=[
        "Eligibility reasons count each quadrant assignment's first failed predicate, not every defect.",
        "Excluded quadrant assignments are not matcher-rejected candidate rows. Incidence includes every admitted candidate row unchanged; dispositions are annotations, not pruning.",
        "Structural validation of the observed result is not a second source replay or geometry acceptance."])
    if production_error is not None or capture.error is not None:
        report.update(reason="production_exception", field=type(production_error or capture.error).__name__)
    elif capture.calls != 1:
        report.update(reason="g_call_count", field="zero" if capture.calls == 0 else "multiple")
    else:
        try:
            report["graph"] = _graph_record(capture)
            if capture.graph.selection.status == "budget_refused":
                report.update(status="refused", reason="selection_budget_refused", field="budget")
            elif capture.graph.status == "assembled":
                report["status"] = "no_failure"
            else:
                report["status"] = "complete"
        except DiagnosticError as error:
            report.update(reason=error.reason, field=error.field)
    if capture.reading_attrition is not None:
        report["reading_attrition"] = _reading_attrition_record(capture.reading_attrition, capture)
        if report["reading_attrition"] is None or report["reading_attrition"]["status"] != "complete":
            report.update(status="incomplete", reason="reading_attrition_incomplete", field="observer")
        if capture.reading_attrition.support_enabled:
            report["support_score_attrition"] = _support_score_record(capture.reading_attrition, capture, report["reading_attrition"])
            if report["support_score_attrition"] is None or report["support_score_attrition"]["status"] != "complete":
                report.update(status="incomplete", reason="support_score_attrition_incomplete", field="observer")
        if capture.reading_attrition.paired is not None:
            pair_field = ("unit_weight_sparse_support" if capture.reading_attrition.paired.unit_weight_sparse
                          else "paired_segment_support")
            try:
                report[pair_field]=_paired_support_record(capture.reading_attrition,capture,report["reading_attrition"])
            except Exception:
                report[pair_field]=None
            if report[pair_field] is None or report[pair_field]["status"]!="complete":
                report.update(status="incomplete",reason=pair_field+"_incomplete",field="observer")
    return validate_report(_seal(report))


def _readback(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_REPORT_BYTES + 1)
    if len(data) > MAX_REPORT_BYTES:
        raise DiagnosticError("readback_failed", "size")
    _json_preflight(data, MAX_REPORT_BYTES)
    report = json.loads(data)
    validate_report(report)
    if canonical_bytes(report) != data:
        raise DiagnosticError("readback_failed", "canonical")
    return data


def publish_report(path, report):
    """Exclusive same-directory temp, fsync, no-clobber link, strict readback."""
    path = Path(path)
    data = canonical_bytes(validate_report(report))
    if path.exists():
        raise DiagnosticError("output_collision", "diagnostic")
    temporary, published = None, False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".g7-diagnostic-", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if _readback(temporary) != data:
            raise DiagnosticError("readback_failed", "temporary")
        os.link(temporary, path)
        published = True
        if _readback(path) != data:
            raise DiagnosticError("readback_failed", "diagnostic")
    except FileExistsError as error:
        raise DiagnosticError("output_collision", "diagnostic") from error
    except DiagnosticError:
        raise
    except (OSError, ValueError, TypeError, RecursionError) as error:
        raise DiagnosticError("publication_failed", type(error).__name__) from error
    finally:
        # Unlink only this invocation's exact files. Preserve any foreign racer.
        if temporary is not None:
            if published and sys.exc_info()[0] is not None and path.exists() and os.path.samefile(temporary, path):
                path.unlink()
            temporary.unlink(missing_ok=True)


@dataclasses.dataclass(frozen=True)
class ObservationOutcome:
    production_result: object
    production_error: Exception | None
    report: dict
    diagnostic_error: DiagnosticError | None


def run_once(images, *, project, destination, diagnostic, name, description, delivery=None,
             reading_attrition=False, paired_segment_support=False, unit_weight_sparse_support=False,
             support_line_weight_policy="structural"):
    """Ordinary delivery with one explicit evaluation-only support experiment."""
    support_line_weight_policy = _support_line_weight_policy(support_line_weight_policy)
    if support_line_weight_policy == "unit" and unit_weight_sparse_support is True:
        raise DiagnosticError("invalid_input", "unit_policy_counterfactual_combination")
    delivery = derive_and_deliver_image_output if delivery is None else delivery
    if paired_segment_support is True and reading_attrition is not True:
        raise DiagnosticError("invalid_input","paired_requires_reading_attrition")
    if unit_weight_sparse_support is True:
        if reading_attrition is not True or paired_segment_support is True:
            raise DiagnosticError("invalid_input","unit_sparse_requires_reading_attrition_and_separate_mode")
    bounded = []
    for image in images:
        if len(bounded) == 32:
            raise DiagnosticError("invalid_input", "image_count")
        bounded.append(Path(image))
    images = tuple(bounded)
    outputs = tuple(Path(p) for p in (project, destination, diagnostic))
    if not 1 <= len(images) <= 32 or any(not p.is_file() or p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff") for p in images):
        raise DiagnosticError("invalid_input", "ordinary_images")
    resolved = tuple(p.resolve() for p in outputs)
    if len(set(resolved)) != 3 or any(p.exists() for p in outputs) or set(resolved).intersection(p.resolve() for p in images):
        raise DiagnosticError("output_collision", "fresh_outputs")
    actual_pins = {p: _file_hash(REPO_ROOT / p) for p in SOURCE_PINS}
    if actual_pins != SOURCE_PINS:
        raise DiagnosticError("source_pin_mismatch", "runtime")
    run = {"inputs": [{"index": i, "path": str(p), "sha256": _file_hash(p)} for i, p in enumerate(images)],
           "runtime": {"python": platform.python_version(), "source_sha256": actual_pins,
                       "pin_coverage": "Partial execution-seam pins; not a complete transitive dependency/environment identity.",
                       "runner_sha256": _file_hash(Path(__file__))}, "production": {}, "outputs": []}
    capture = GraphCapture(ReadingAttritionObserver(paired_segment_support=paired_segment_support,
        unit_weight_sparse_support=unit_weight_sparse_support) if reading_attrition else None,
        support_line_weight_policy=support_line_weight_policy)
    production, error, start = None, None, time.monotonic()
    try:
        production = capture.invoke(delivery, images, destination=Path(destination), project_path=Path(project), name=name, description=description)
    except Exception as failure:
        error = failure
    run["elapsed_seconds"] = time.monotonic() - start
    if production is not None:
        run["production"] = {k: getattr(production, k, None) for k in ("schema", "producer", "status", "reason", "result_sha256", "production_sha256", "delivery_sha256", "refusal_sha256")}
    diagnostic_error = None
    try:
        for item in run["inputs"]:
            try:
                unchanged = _file_hash(item["path"]) == item["sha256"]
            except OSError:
                unchanged = False
            if not unchanged:
                raise DiagnosticError("input_changed", "ordered_inputs")
        run["inputs_unchanged_after"] = True
        for kind, path in zip(("mcscan", "litematic"), outputs[:2]):
            run["outputs"].append({"kind": kind, "path": str(path), "exists": path.exists(),
                "sha256": _file_hash(path) if path.is_file() else None, "size_bytes": path.stat().st_size if path.is_file() else None})
        report = build_report(capture, run, error)
        publish_report(diagnostic, report)
    except (DiagnosticError, OSError) as failure:
        diagnostic_error = failure if isinstance(failure, DiagnosticError) else DiagnosticError("diagnostic_io_failed", type(failure).__name__)
        report = validate_report(_seal(dict(schema=SCHEMA, producer=PRODUCER, status="refused", reason=diagnostic_error.reason,
            field=diagnostic_error.field, g_calls=capture.calls, graph=None, run={},
            support_line_weight_policy=capture.support_line_weight_policy,
            notes=["Production result/output unchanged; diagnostic not published."])))
    return ObservationOutcome(production, error, report, diagnostic_error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--reading-attrition", action="store_true", help="Observe existing seed/proposal/reading scalar hooks in the same pass.")
    parser.add_argument("--paired-segment-support",action="store_true",help="Diagnostic-only scalar sparse/raster comparison; requires --reading-attrition.")
    parser.add_argument("--unit-weight-sparse-support",action="store_true",help="Diagnostic-only unit-weight sparse comparison; requires --reading-attrition and excludes --paired-segment-support.")
    parser.add_argument("--support-line-weight-policy",choices=("structural","unit"),default="structural",
                        help="Evaluation-only actual allocation policy; unit excludes --unit-weight-sparse-support.")
    args = parser.parse_args(argv)
    try:
        outcome = run_once(args.image, project=args.project, destination=args.destination, diagnostic=args.diagnostic,
                           name=args.name, description=args.description, reading_attrition=args.reading_attrition,
                           paired_segment_support=args.paired_segment_support,
                           unit_weight_sparse_support=args.unit_weight_sparse_support,
                           support_line_weight_policy=args.support_line_weight_policy)
    except (DiagnosticError, OSError) as error:
        print(json.dumps({"status": "refused", "reason": getattr(error, "reason", "preflight_io_failed"), "field": getattr(error, "field", type(error).__name__)}))
        return 1
    print(json.dumps({"status": outcome.report["status"], "reason": outcome.report["reason"], "result_sha256": outcome.report["result_sha256"],
                      "production": outcome.report["run"].get("production", {})}, sort_keys=True))
    return 1 if outcome.production_error is not None or outcome.diagnostic_error is not None or outcome.report["status"] not in ("complete","no_failure") else 0


if __name__ == "__main__":
    raise SystemExit(main())
