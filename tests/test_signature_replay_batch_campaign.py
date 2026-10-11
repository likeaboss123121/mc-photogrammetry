from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import signature_replay_batch_campaign as subject
from scripts import run_signature_replay_batch_worker as worker
from mc_photogrammetry import evaluation_signature_sampling as sampling


def _artifact(manifest, index, *, accepted=True):
    rows = []
    for i in manifest.batches[index]:
        identity = manifest.identities[i]
        result = (sampling.SignatureSamplingBindingV1(
            sampling.SCHEMA, identity[0], identity[1], identity[2], identity[3],
            "a" * 64, "b" * 64) if accepted else
            sampling.SignatureSamplingReplayRefusalV1(
                sampling.SCHEMA, "refused", "budget_exhausted", "samples",
                "limit=1;planned=2;actual=0", identity[3], "a" * 64))
        rows.append(subject.BatchRow(identity, result))
    return subject.BatchArtifact(manifest.manifest_sha256, index, "complete", "",
                                 tuple(rows), 1, 2, 1, 3.0)


def test_manifest_freezes_exact_order_partition_and_limits():
    manifest = subject.CampaignManifest()
    assert len(manifest.identities) == 37
    assert [len(batch) for batch in manifest.batches] == [4] * 9 + [1]
    assert tuple(i for batch in manifest.batches for i in batch) == tuple(range(37))
    assert manifest.memory_limit_bytes == 8 * 1024**3
    assert manifest.deadline_seconds == 600 and manifest.manifest_sha256
    assert subject.CampaignManifest(**{
        name: getattr(manifest, name) for name in manifest.__dataclass_fields__}) == manifest


def test_manifest_tamper_and_batch_identity_changes_refuse():
    manifest = subject.CampaignManifest()
    with pytest.raises(ValueError, match="manifest mismatch"):
        replace(manifest, deadline_seconds=601, manifest_sha256="")
    artifact = _artifact(manifest, 0)
    changed = list(artifact.rows)
    changed[0] = subject.BatchRow(manifest.identities[2], changed[2].result)
    with pytest.raises(ValueError, match="identities mismatch"):
        replace(artifact, rows=tuple(changed), artifact_sha256="")


def test_regenerated_universe_must_match_before_subset_selection():
    manifest = subject.CampaignManifest()
    rows = tuple(type("Row", (), dict(frame_id=f, canonical_index=r,
                                      alternative_index=a, signature_sha256=s))()
                 for f, r, a, s in manifest.identities)
    assert tuple(row.signature_sha256 for row in subject.select_batch(
        manifest, 9, rows)) == (manifest.identities[-1][-1],)
    with pytest.raises(ValueError, match="universe mismatch"):
        subject.select_batch(manifest, 0, rows[::-1])


def test_timeout_is_atomic_typed_batch_refusal():
    manifest = subject.CampaignManifest()
    timeout = subject.BatchArtifact(manifest.manifest_sha256, 3, "refused",
                                    "deadline_exhausted", (), 1, 2, 1, 600.0)
    assert timeout.rows == () and timeout.reason == "deadline_exhausted"
    unavailable = replace(timeout, baseline_rss_bytes=None, peak_rss_bytes=None,
                          post_gc_rss_bytes=None, elapsed_seconds=601.0,
                          artifact_sha256="")
    assert unavailable.elapsed_seconds > manifest.deadline_seconds
    with pytest.raises(ValueError, match="malformed"):
        replace(timeout, rows=(object(),),
                artifact_sha256="")


def test_merge_requires_all_batches_once_and_in_parent_order():
    manifest = subject.CampaignManifest()
    artifacts = tuple(_artifact(manifest, i) for i in range(10))
    merged = subject.merge_campaign(manifest, artifacts)
    assert merged.execution_complete and merged.all_signatures_accepted
    assert tuple(row.identity for row in merged.rows) == manifest.identities
    assert not subject.merge_campaign(manifest, artifacts[:-1]).execution_complete
    duplicate = artifacts[:-1] + (artifacts[0],)
    assert subject.merge_campaign(manifest, duplicate).reason == "duplicate_batch"


def test_execution_complete_is_distinct_from_all_accepted():
    manifest = subject.CampaignManifest()
    artifacts = tuple(_artifact(manifest, i, accepted=(i != 4)) for i in range(10))
    merged = subject.merge_campaign(manifest, artifacts)
    assert merged.execution_complete is True
    assert merged.all_signatures_accepted is False
    assert merged.reason == "signature_refusal" and len(merged.rows) == 37


def test_artifact_digest_rejects_mutation():
    manifest = subject.CampaignManifest()
    artifact = _artifact(manifest, 9)
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(artifact, artifact_sha256="0" * 64)


def test_measurements_index_and_identity_only_rows_fail():
    manifest = subject.CampaignManifest()
    with pytest.raises(ValueError, match="malformed"):
        replace(_artifact(manifest, 0), peak_rss_bytes=0, artifact_sha256="")
    with pytest.raises(ValueError):
        subject.BatchArtifact(manifest.manifest_sha256, 10, "complete", "", (),
                              1, 2, 1, 1.0)
    with pytest.raises((AttributeError, ValueError, TypeError)):
        subject.BatchArtifact(manifest.manifest_sha256, 0, "complete", "",
                              ({"result_type": "SignatureSamplingBindingV1"},) * 4,
                              1, 2, 1, 1.0)


def test_source_verification_and_atomic_write(tmp_path):
    contents = (b"be", b"truth", b"jar", b"palette", b"worker", b"producer")
    paths = []
    for i, value in enumerate(contents):
        path = tmp_path / str(i); path.write_bytes(value); paths.append(path)
    expected = tuple(__import__("hashlib").sha256(v).hexdigest() for v in contents)
    original = (subject.BE_FILE_SHA256, subject.SOURCE_HASHES,
                subject.PRODUCER_HASHES)
    try:
        subject.BE_FILE_SHA256, subject.SOURCE_HASHES, subject.PRODUCER_HASHES = (
            expected[0], expected[1:4], expected[4:])
        subject.verify_sources(paths)
        paths[-1].write_bytes(b"changed")
        with pytest.raises(ValueError, match="source hash mismatch"):
            subject.verify_sources(paths)
    finally:
        subject.BE_FILE_SHA256, subject.SOURCE_HASHES, subject.PRODUCER_HASHES = original
    artifact = _artifact(subject.CampaignManifest(), 0)
    target = tmp_path / "artifact.json"
    subject.atomic_write_artifact(target, artifact)
    assert artifact.artifact_sha256 in target.read_text()


def test_foreign_parent_and_precontainment_contract_refuse():
    manifest = subject.CampaignManifest()
    with pytest.raises(ValueError, match="manifest mismatch"):
        replace(manifest, be_file_sha256="0" * 64, manifest_sha256="")
    with pytest.raises(ValueError, match="malformed"):
        replace(_artifact(manifest, 0), manifest_sha256="0" * 64,
                artifact_sha256="")
    source = open(subject.__file__, encoding="utf-8").read()
    assert "creationflags=0x4" in source
    assert source.index("AssignProcessToJobObject") < source.index("NtResumeProcess")


def test_real_batch_executor_selects_only_batch_zero_without_full_preflight():
    manifest = subject.CampaignManifest()
    rows = tuple(type("Evidence", (), dict(
        frame_id=f, canonical_index=r, alternative_index=a,
        signature_sha256=s))() for f, r, a, s in manifest.identities)
    calls = []
    def replay(*, signature_evidence, truth_evidence, witness_budget):
        calls.append(signature_evidence.signature_sha256)
        return sampling.SignatureSamplingBindingV1(
            sampling.SCHEMA, signature_evidence.frame_id,
            signature_evidence.canonical_index,
            signature_evidence.alternative_index,
            signature_evidence.signature_sha256, "a" * 64, "b" * 64)
    times = iter((10.0, 11.0))
    artifact = worker.execute_batch(
        manifest=manifest, batch_index=0, regenerated=rows,
        evidence_by_frame={f: object() for f, *_ in manifest.identities},
        replay=replay, rss=lambda: 100, clock=lambda: next(times))
    assert artifact.status == "complete" and len(artifact.rows) == 4
    assert calls == [identity[3] for identity in manifest.identities[:4]]
    assert "preflight_synthetic_targeted_capture" not in open(
        worker.__file__, encoding="utf-8").read()


def test_worker_cli_help_runs_from_repository():
    completed = subprocess.run(
        [sys.executable, "scripts/run_signature_replay_batch_worker.py", "--help"],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    assert "--batch-index" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object contract")
def test_launcher_nonzero_exit_writes_one_atomic_refusal(tmp_path):
    root = Path(subject.__file__).resolve().parent.parent
    paths = (root / ".tmp-synthetic-canal-be.json",
             root / "exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/ground_truth.json",
             Path.home() / "AppData/Roaming/.minecraft/versions/1.21.5/1.21.5.jar",
             Path.home() / "AppData/Roaming/.minecraft/versions/1.21.5/1.21.5.mc-textures.json",
             Path(worker.__file__),
             root / "src/mc_photogrammetry/synthetic_targeted_capture_preflight.py")
    output = tmp_path / "failed.json"
    _stdout, _stderr, code = subject.launch_reviewed_worker(
        (sys.executable, "-c", "raise MemoryError('controlled')"),
        verified_paths=paths, artifact_path=output, batch_index=0)
    assert code != 0 and output.exists()
    record = __import__("json").loads(output.read_text())
    assert record["status"] == "refused" and record["rows"] == []
    assert record["reason"].startswith("worker_exit_")
