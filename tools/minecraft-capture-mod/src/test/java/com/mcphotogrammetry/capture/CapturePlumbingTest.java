package com.mcphotogrammetry.capture;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Contract tests for the Minecraft-free capture plumbing: eye-vs-feet arithmetic,
 * stale-screenshot rejection, atomic JSON replacement and atomic disarm. These run
 * on a plain JVM so the durability rules cannot regress without a launch.
 */
class CapturePlumbingTest {

    @TempDir
    Path temporary;

    @Test
    void eyeIsFeetPlusEyeHeightAndNeverTheFeetItself() {
        double[] feet = {12.5, 64.0, -3.25};
        double[] eye = CapturePlumbing.eyeFromFeet(feet, CapturePlumbing.DEFAULT_EYE_HEIGHT);
        assertEquals(12.5, eye[0]);
        assertEquals(64.0 + 1.62, eye[1], 1e-9);
        assertEquals(-3.25, eye[2]);
        assertEquals(1.62, CapturePlumbing.distance(feet, eye), 1e-9);
    }

    @Test
    void toleranceAcceptsASettledPoseAndRejectsADriftedOne() {
        double[] requestedFeet = {0.0, 70.0, 0.0};
        double[] requestedEye = CapturePlumbing.eyeFromFeet(requestedFeet, 1.62);
        double tolerance = CapturePlumbing.DEFAULT_POSITION_TOLERANCE;

        assertTrue(CapturePlumbing.withinTolerance(new double[] {0.05, 70.0, -0.1},
                new double[] {0.05, 71.62, -0.1}, requestedFeet, requestedEye, tolerance));
        // A capture that put the feet where the eye should be must not pass: the
        // recorded eye would be 1.62 blocks too high.
        assertFalse(CapturePlumbing.withinTolerance(new double[] {0.0, 71.62, 0.0},
                new double[] {0.0, 73.24, 0.0}, requestedFeet, requestedEye, tolerance));
        assertFalse(CapturePlumbing.withinTolerance(new double[] {1.0, 70.0, 0.0},
                new double[] {1.0, 71.62, 0.0}, requestedFeet, requestedEye, tolerance));
        assertFalse(CapturePlumbing.withinTolerance(new double[] {Double.NaN, 70.0, 0.0},
                requestedEye, requestedFeet, requestedEye, tolerance));
    }

    @Test
    void groundTruthExportStatusIsIndependentOfFrameCaptureStatus() {
        assertTrue(CapturePlumbing.isGroundTruthExported(true, false),
                "a written block export remains exported even when separate frame results failed");
        assertFalse(CapturePlumbing.isGroundTruthExported(false, false),
                "a failed payload write is not an export");
        assertFalse(CapturePlumbing.isGroundTruthExported(true, true),
                "a failure sentinel is never reported as exported ground truth");
    }

    @Test
    void readsOnlyWellFormedFiniteTriples() {
        JsonObject object = JsonParser.parseString(
                "{\"eye_position\": [1, 2.5, -3], \"bad\": [1, 2], \"text\": [\"a\",\"b\",\"c\"]}")
                .getAsJsonObject();
        Optional<double[]> values = CapturePlumbing.readTriple(object, "eye_position");
        assertTrue(values.isPresent());
        assertEquals(2.5, values.get()[1], 1e-9);
        assertTrue(CapturePlumbing.readTriple(object, "bad").isEmpty());
        assertTrue(CapturePlumbing.readTriple(object, "text").isEmpty());
        assertTrue(CapturePlumbing.readTriple(object, "missing").isEmpty());
    }

    @Test
    void removesAStaleScreenshotButNeverADirectory() throws Exception {
        Path stale = temporary.resolve("view-00.png");
        Files.write(stale, new byte[] {1, 2, 3});
        assertTrue(CapturePlumbing.removeStaleScreenshot(stale));
        assertFalse(Files.exists(stale));
        assertFalse(CapturePlumbing.removeStaleScreenshot(stale));

        Path directory = temporary.resolve("not-a-file.png");
        Files.createDirectory(directory);
        assertFalse(CapturePlumbing.removeStaleScreenshot(directory));
        assertTrue(Files.isDirectory(directory));
    }

    @Test
    void acceptsAScreenshotOnlyAfterTheCallbackAndANonEmptyFile() throws Exception {
        Path target = temporary.resolve("view-01.png");
        assertFalse(CapturePlumbing.isScreenshotAccepted(target, true));

        Files.write(target, new byte[0]);
        assertFalse(CapturePlumbing.isScreenshotAccepted(target, true),
                "an empty file is not a finished screenshot");
        assertFalse(CapturePlumbing.isScreenshotAccepted(target, false),
                "a file without the callback is not a finished screenshot");

        Files.write(target, new byte[] {9});
        assertTrue(CapturePlumbing.isScreenshotAccepted(target, true));
    }

    @Test
    void replacesJsonAtomicallyAndLeavesNoTemporaryFile() throws Exception {
        Path target = temporary.resolve("out").resolve("results.json");
        // A missing parent is created rather than throwing: the controller writes
        // into a fresh plan directory.
        CapturePlumbing.writeJsonAtomic(target, "{\"first\": true}\n");
        assertEquals("{\"first\": true}\n", Files.readString(target, StandardCharsets.UTF_8));
        CapturePlumbing.writeJsonAtomic(target, "{\"second\": \"\\u00e9\"}\n");
        assertEquals("{\"second\": \"\\u00e9\"}\n", Files.readString(target, StandardCharsets.UTF_8));
        assertFalse(Files.exists(target.resolveSibling("results.json.tmp")),
                "the temporary sibling must not survive the replace");
    }

    @Test
    void disarmIsAnAtomicSameDirectoryRenameNotAPlainDelete() throws Exception {
        Path armed = temporary.resolve("armed.json");
        Files.writeString(armed, "{\"plan_path\": \"p.json\"}", StandardCharsets.UTF_8);

        Optional<Path> marker = CapturePlumbing.disarmAtomic(armed, "consumed");
        assertTrue(marker.isPresent());
        assertEquals(temporary.resolve("armed.json.consumed"), marker.get());
        assertFalse(Files.exists(armed));
        assertFalse(Files.exists(marker.get()), "the marker is deleted after the rename");

        Optional<Path> rejected = CapturePlumbing.disarmAtomic(armed, "rejected");
        assertTrue(rejected.isEmpty(), "an already disarmed file is a no-op");
    }

    @Test
    void disarmOverwritesAStaleMarkerInsteadOfFailing() throws Exception {
        Path armed = temporary.resolve("armed.json");
        Files.writeString(armed, "{}", StandardCharsets.UTF_8);
        Files.writeString(CapturePlumbing.disarmMarker(armed, "consumed"), "old",
                StandardCharsets.UTF_8);
        assertTrue(CapturePlumbing.disarmAtomic(armed, "consumed").isPresent());
        assertFalse(Files.exists(armed));
        assertFalse(Files.exists(CapturePlumbing.disarmMarker(armed, "consumed")));
    }
}
