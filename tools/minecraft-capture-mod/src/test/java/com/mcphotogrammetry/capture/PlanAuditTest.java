package com.mcphotogrammetry.capture;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Contract tests for the Minecraft-free plan audit. These run on a plain JVM as
 * part of {@code gradlew build}, so a safety rule cannot silently regress just
 * because nobody launched the game.
 */
class PlanAuditTest {

    private static final String STAGING_TOKEN =
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    private static String planWith(String jobs) {
        return """
                {
                  "schema_version": 1,
                  "mod_contract_version": 1,
                  "plan_id": "plan-abc123",
                  "scene_id": "template-2-house",
                  "dataset_id": "ds17",
                  "world_identity": "Template 2",
                  "staging_token": "%s",
                  "region": {"min": [0, 60, 0], "max": [23, 83, 23]},
                  "jobs": [%s],
                  "notes": []
                }
                """.formatted(STAGING_TOKEN, jobs);
    }

    private static String job(String frameId, String overrides) {
        return """
                {
                  "schema_version": 1,
                  "frame_id": "%s",
                  "resolution": [1920, 1080],
                  "target": [12.0, 72.0, 12.0],
                  "settle_ticks": 20,
                  "settings": {"gamma": 0.0, "time_of_day": 6000, "weather": "clear",
                               "hud_hidden": true, "resource_pack": null, "shader_pack": null},
                  "camera": {"position": [40.0, 90.0, 40.0], "yaw_degrees": 135.0,
                             "pitch_degrees": -25.0, "fov_degrees": 70.0, "eye_height": 1.62,
                             "roll_degrees": 0.0, "intrinsics": null}%s
                }
                """.formatted(frameId, overrides);
    }

    @Test
    void parsesAWellFormedPlanAndDerivesScreenshotFilenames() throws Exception {
        PlanAudit.AuditedPlan plan = PlanAudit.parse(planWith(job("view-house-a-00", "")));
        assertEquals("plan-abc123", plan.planId());
        assertEquals("Template 2", plan.worldIdentity());
        assertEquals(24 * 24 * 24L, plan.volume());
        assertEquals(1, plan.jobs().size());
        assertEquals("view-house-a-00.png", plan.jobs().get(0).filename());
        assertEquals(1920, plan.jobs().get(0).width());
        assertTrue(plan.jobs().get(0).settings().hudHidden());
    }

    @Test
    void rejectsUnknownSchemaAndContractVersions() {
        String bumped = planWith(job("view-house-a-00", "")).replace("\"schema_version\": 1,",
                "\"schema_version\": 2,");
        assertThrows(PlanAudit.PlanRejected.class, () -> PlanAudit.parse(bumped));

        String contract = planWith(job("view-house-a-00", ""))
                .replace("\"mod_contract_version\": 1", "\"mod_contract_version\": 9");
        assertThrows(PlanAudit.PlanRejected.class, () -> PlanAudit.parse(contract));
    }

    @Test
    void requiresARealStagingTokenAndRefusesTheInertOne() throws Exception {
        PlanAudit.AuditedPlan parsed = PlanAudit.parse(planWith(job("view-house-a-00", "")));
        assertEquals(STAGING_TOKEN, parsed.stagingToken());

        String missing = planWith(job("view-house-a-00", ""))
                .replace("\"staging_token\": \"" + STAGING_TOKEN + "\",", "");
        assertThrows(PlanAudit.PlanRejected.class, () -> PlanAudit.parse(missing));

        String shortToken = planWith(job("view-house-a-00", ""))
                .replace(STAGING_TOKEN, "abc");
        assertThrows(PlanAudit.PlanRejected.class, () -> PlanAudit.parse(shortToken));

        String inert = planWith(job("view-house-a-00", ""))
                .replace(STAGING_TOKEN, StagingMarker.UNSTAGED_TOKEN);
        assertThrows(PlanAudit.PlanRejected.class, () -> PlanAudit.parse(inert));
    }

    @Test
    void yawComparisonIsCircularRatherThanASubtraction() {
        // A client at 10 degrees is twenty degrees from 350, not 340 degrees away.
        assertEquals(20.0, PlanAudit.angleDeltaDegrees(10.0, 350.0), 1e-9);
        assertEquals(20.0, PlanAudit.angleDeltaDegrees(350.0, 10.0), 1e-9);
        assertEquals(10.0, PlanAudit.angleDeltaDegrees(300.0, -50.0), 1e-9);
        assertEquals(180.0, PlanAudit.angleDeltaDegrees(0.0, 180.0), 1e-9);
        assertEquals(180.0, PlanAudit.angleDeltaDegrees(90.0, -90.0), 1e-9);
        assertEquals(20.0, PlanAudit.angleDeltaDegrees(370.0, -10.0), 1e-9);
        assertTrue(PlanAudit.headingWithinTolerance(370.0, -10.0,
                PlanAudit.HEADING_TOLERANCE_DEGREES + 19.0));
        assertFalse(PlanAudit.headingWithinTolerance(30.0, -10.0,
                PlanAudit.HEADING_TOLERANCE_DEGREES));
    }

    @Test
    void rejectsUnsafeFrameIdsAndDuplicateJobs() {
        assertFalse(PlanAudit.isSafeFrameId("../escape"));
        assertFalse(PlanAudit.isSafeFrameId("frame/name"));
        assertTrue(PlanAudit.isSafeFrameId("view-house-a-00"));
        assertThrows(PlanAudit.PlanRejected.class,
                () -> PlanAudit.parse(planWith(job("../escape", ""))));
        assertThrows(PlanAudit.PlanRejected.class,
                () -> PlanAudit.parse(planWith(job("same", "") + "," + job("same", ""))));
    }

    @Test
    void acceptsZeroJobsOnlyForExplicitContextExportOnlyPlans() throws Exception {
        String empty = planWith("");
        assertThrows(PlanAudit.PlanRejected.class, () -> PlanAudit.parse(empty));

        String contextOnly = empty.replace("\"region\":",
                "\"context_export_only\": true,\n  \"region\":");
        PlanAudit.AuditedPlan parsed = PlanAudit.parse(contextOnly);
        assertTrue(parsed.contextExportOnly());
        assertTrue(parsed.jobs().isEmpty());

        String contextWithCaptureJob = planWith(job("view-house-a-00", ""))
                .replace("\"region\":", "\"context_export_only\": true,\n  \"region\":");
        assertThrows(PlanAudit.PlanRejected.class,
                () -> PlanAudit.parse(contextWithCaptureJob));

        String stringFlag = contextOnly.replace("true", "\"true\"");
        assertThrows(PlanAudit.PlanRejected.class, () -> PlanAudit.parse(stringFlag));
    }

    @Test
    void rejectsNonzeroRollBecauseTheClientCannotApplyIt() {
        // A rolled plan cannot be honoured, so accepting it would silently capture a
        // different pose than the frozen ground truth claims.
        String rolled = planWith(job("view-house-a-00", ""))
                .replace("\"roll_degrees\": 0.0", "\"roll_degrees\": 5.0");
        assertThrows(PlanAudit.PlanRejected.class, () -> PlanAudit.parse(rolled));
    }

    @Test
    void rejectsJobCountsAboveTheCap() {
        StringBuilder jobs = new StringBuilder();
        for (int index = 0; index < PlanAudit.MAX_JOBS + 1; index++) {
            if (index > 0) {
                jobs.append(',');
            }
            jobs.append(job(String.format("view-%03d", index), ""));
        }
        assertThrows(PlanAudit.PlanRejected.class, () -> PlanAudit.parse(planWith(jobs.toString())));
    }

    @Test
    void rejectsRegionVolumesAboveTheExportCap() {
        String huge = planWith(job("view-house-a-00", ""))
                .replace("\"min\": [0, 60, 0]", "\"min\": [0, 0, 0]")
                .replace("\"max\": [23, 83, 23]", "\"max\": [200, 200, 200]");
        assertThrows(PlanAudit.PlanRejected.class, () -> PlanAudit.parse(huge));
    }

    @Test
    void verifiesThePlanDigestAndRejectsATamperedFile() throws Exception {
        byte[] bytes = planWith(job("view-house-a-00", "")).getBytes(StandardCharsets.UTF_8);
        String digest = PlanAudit.sha256Hex(bytes);
        assertEquals(64, digest.length());
        PlanAudit.verifyDigest(bytes, digest);
        assertThrows(PlanAudit.PlanRejected.class,
                () -> PlanAudit.verifyDigest(bytes, "0".repeat(64)));
        assertThrows(PlanAudit.PlanRejected.class,
                () -> PlanAudit.verifyDigest(bytes, "not-a-digest"));
    }

    @Test
    void resolvesOnlyPathsInsideTheGameDirectory() throws Exception {
        Path root = Path.of("C:/staging-game").toAbsolutePath();
        assertEquals(root.resolve("mc-photo-capture/plans/p.json"),
                PlanAudit.resolveContained(root, "mc-photo-capture/plans/p.json"));
        assertThrows(PlanAudit.PlanRejected.class,
                () -> PlanAudit.resolveContained(root, "../outside.json"));
        assertThrows(PlanAudit.PlanRejected.class,
                () -> PlanAudit.resolveContained(root, "C:/Windows/system32/evil.json"));
        assertThrows(PlanAudit.PlanRejected.class,
                () -> PlanAudit.resolveContained(root, ""));
    }
}
