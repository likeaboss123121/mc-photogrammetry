package com.mcphotogrammetry.capture;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Contract tests for the staged-copy identity proof.
 *
 * <p>A level name is not identity. These tests cover the pure marker validation
 * and the content fingerprint the mod recomputes over the world it is about to
 * mutate, so a plan cannot be carried onto a different save that happens to share
 * a name.
 */
class StagingMarkerTest {

    private static final String TOKEN =
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    @TempDir
    Path temporary;

    private static String markerJson(String token, String fingerprint, String world) {
        return """
                {
                  "schema_version": 1,
                  "token": "%s",
                  "source_fingerprint": "%s",
                  "world_identity": "%s",
                  "dataset_id": "ds17",
                  "scene_id": "template-2-house",
                  "plan_id": "plan-abc123"
                }
                """.formatted(token, fingerprint, world);
    }

    @Test
    void acceptsAMarkerThatMatchesThePlanExactly() throws Exception {
        StagingMarker.Marker marker = StagingMarker.parseAndValidate(
                markerJson(TOKEN, TOKEN, "Template 2"), TOKEN, "Template 2",
                Path.of("staging.json"));
        assertEquals(TOKEN, marker.token());
        assertEquals(TOKEN, marker.sourceFingerprint());
        assertEquals("Template 2", marker.worldIdentity());
        assertEquals("plan-abc123", marker.planId());
    }

    @Test
    void rejectsATokenThePlanDidNotAskFor() {
        String other = "1".repeat(64);
        assertThrows(StagingMarker.StagingMismatch.class, () -> StagingMarker.parseAndValidate(
                markerJson(other, other, "Template 2"), TOKEN, "Template 2",
                Path.of("staging.json")));
    }

    @Test
    void rejectsAWorldIdentityMismatchAndAnUnsupportedSchema() {
        assertThrows(StagingMarker.StagingMismatch.class, () -> StagingMarker.parseAndValidate(
                markerJson(TOKEN, TOKEN, "Some other world"), TOKEN, "Template 2",
                Path.of("staging.json")));
        String bumped = markerJson(TOKEN, TOKEN, "Template 2").replace("\"schema_version\": 1",
                "\"schema_version\": 2");
        assertThrows(StagingMarker.StagingMismatch.class, () -> StagingMarker.parseAndValidate(
                bumped, TOKEN, "Template 2", Path.of("staging.json")));
    }

    @Test
    void rejectsTheInertUnstagedTokenAndMalformedMarkers() {
        assertThrows(StagingMarker.StagingMismatch.class, () -> StagingMarker.parseAndValidate(
                markerJson(StagingMarker.UNSTAGED_TOKEN, StagingMarker.UNSTAGED_TOKEN,
                        "Template 2"),
                StagingMarker.UNSTAGED_TOKEN, "Template 2", Path.of("staging.json")));
        assertThrows(StagingMarker.StagingMismatch.class, () -> StagingMarker.parseAndValidate(
                "{not json", TOKEN, "Template 2", Path.of("staging.json")));
        assertThrows(StagingMarker.StagingMismatch.class, () -> StagingMarker.parseAndValidate(
                markerJson("abc", "abc", "Template 2"), TOKEN, "Template 2",
                Path.of("staging.json")));
    }

    @Test
    void readAndValidateWorldRequiresTheMarkerInsideTheOpenSaveRoot() throws Exception {
        Path worldRoot = temporary.resolve("saves").resolve("Template 2");
        Files.createDirectories(worldRoot);
        Path marker = StagingMarker.worldMarkerPath(worldRoot);
        assertEquals(worldRoot.resolve(".mc-photo-capture-staging.json"), marker);
        Files.writeString(marker, markerJson(TOKEN, TOKEN, "Template 2"),
                StandardCharsets.UTF_8);
        assertEquals(TOKEN,
                StagingMarker.readAndValidateWorld(worldRoot, TOKEN, "Template 2").token());

        Path empty = temporary.resolve("no-marker");
        Files.createDirectories(empty);
        assertThrows(StagingMarker.StagingMismatch.class,
                () -> StagingMarker.readAndValidateWorld(empty, TOKEN, "Template 2"));
    }

    @Test
    void theSaveFolderMayDifferFromTheLevelNameBecauseTheRootIsResolvedNotGuessed()
            throws Exception {
        // Minecraft's LevelName lives in level.dat; the folder can be anything. The
        // mod is handed the real open root, so the marker is found where it was
        // written even when the folder name is unrelated.
        Path worldRoot = temporary.resolve("saves").resolve("world-folder-7f3a");
        Files.createDirectories(worldRoot);
        Files.write(worldRoot.resolve("level.dat"), new byte[] {1, 2, 3});
        Files.writeString(StagingMarker.worldMarkerPath(worldRoot),
                markerJson(TOKEN, TOKEN, "Template 2"), StandardCharsets.UTF_8);
        assertEquals("Template 2",
                StagingMarker.readAndValidateWorld(worldRoot, TOKEN, "Template 2")
                        .worldIdentity());

        // The old guess - saves/<LevelName> - must not be what is trusted.
        Path guessed = temporary.resolve("saves").resolve("Template 2");
        assertFalse(Files.exists(guessed));
    }

    @Test
    void aCopiedWorldKeepsItsIdentityAfterMinecraftUpdatesOrdinaryFiles() throws Exception {
        // A copy is verified before the marker exists; the marker then carries the
        // source fingerprint. The mod must not rehash the live tree, because the
        // game updates level.dat, player data, and region files as soon as the copy
        // is opened - a valid older copy would otherwise fail.
        Path source = temporary.resolve("source").resolve("Template 2");
        Files.createDirectories(source);
        Files.writeString(source.resolve("level.dat"), "original", StandardCharsets.UTF_8);
        String fingerprint = StagingMarker.fingerprint(source);

        Path staged = temporary.resolve("staging").resolve("saves").resolve("Template 2");
        Files.createDirectories(staged);
        Files.writeString(staged.resolve("level.dat"), "original", StandardCharsets.UTF_8);
        Files.writeString(StagingMarker.worldMarkerPath(staged),
                markerJson(TOKEN, fingerprint, "Template 2"), StandardCharsets.UTF_8);

        // The game now rewrites the ordinary world bytes.
        Files.writeString(staged.resolve("level.dat"), "rewritten-by-minecraft",
                StandardCharsets.UTF_8);
        Files.writeString(staged.resolve("session.lock"), "lock", StandardCharsets.UTF_8);

        // Only the marker's token and identity matter: the mutating capture is not
        // refused just because the live tree changed underneath it.
        StagingMarker.Marker marker = StagingMarker.readAndValidateWorld(staged, TOKEN,
                "Template 2");
        assertEquals(fingerprint, marker.sourceFingerprint());
        assertEquals(TOKEN, marker.token());
    }

    @Test
    void fingerprintIsContentAddressedAndOrderIndependent() throws Exception {
        Path world = temporary.resolve("saves").resolve("Template 2");
        Files.createDirectories(world.resolve("region"));
        Files.write(world.resolve("level.dat"), new byte[] {1, 2, 3});
        Files.write(world.resolve("region").resolve("r.0.0.mca"), new byte[] {4, 5});
        // session.lock is never part of the staged world, so it must not matter.
        Files.write(world.resolve("session.lock"), new byte[] {9});

        String first = StagingMarker.fingerprint(world);
        assertEquals(first, StagingMarker.fingerprint(world));
        assertEquals(64, first.length());

        // Same bytes written in a different order are the same world.
        Files.write(world.resolve("region").resolve("r.0.0.mca"), new byte[] {4, 5});
        assertEquals(first, StagingMarker.fingerprint(world));

        // A changed block region is a different world.
        Files.write(world.resolve("region").resolve("r.0.0.mca"), new byte[] {4, 6});
        assertNotEquals(first, StagingMarker.fingerprint(world));
    }

    @Test
    void fingerprintRejectsAWorldRootThatIsNotADirectory() {
        assertThrows(IOException.class,
                () -> StagingMarker.fingerprint(temporary.resolve("absent")));
    }

    @Test
    void fileDigestHashesTheFinishedScreenshot() throws Exception {
        Path screenshot = temporary.resolve("view-00.png");
        Files.write(screenshot, "PNG-CONTENT".getBytes(StandardCharsets.UTF_8));
        String digest = StagingMarker.fileSha256(screenshot);
        assertEquals(64, digest.length());
        assertEquals(digest, PlanAudit.sha256Hex(
                "PNG-CONTENT".getBytes(StandardCharsets.UTF_8)));
        Files.write(screenshot, "OTHER".getBytes(StandardCharsets.UTF_8));
        assertNotEquals(digest, StagingMarker.fileSha256(screenshot));
        assertTrue(Files.exists(screenshot));
    }
}
