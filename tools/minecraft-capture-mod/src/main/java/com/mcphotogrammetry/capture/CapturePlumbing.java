package com.mcphotogrammetry.capture;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.Optional;

/**
 * Minecraft-free file and geometry plumbing for the capture controller.
 *
 * <p>Everything here is deliberately pure so the parts of the correction that can
 * be wrong without Minecraft - eye-vs-feet arithmetic, stale-screenshot removal,
 * atomic JSON replacement, atomic disarm - are unit testable on a plain JVM.
 * The controller owns the state machine; this class owns the arithmetic and the
 * durability rules.
 *
 * <p>Coordinates: {@code CameraModel.position} is the player's <em>feet</em>; the
 * camera eye is {@code feet + eye_height}. The mod therefore teleports the
 * integrated-server player to the requested feet and compares the recorded eye to
 * the requested eye, never mistaking one for the other.
 */
public final class CapturePlumbing {

    /** Default player eye height in blocks, matching the Python planner. */
    public static final double DEFAULT_EYE_HEIGHT = 1.62;

    /**
     * Requested-vs-actual tolerance in blocks for a settled camera position.
     *
     * <p>The integrated server can nudge a player a fraction of a block while the
     * chunk under it loads, so the planner promises a settled capture only within
     * this bound; the Python finalizer rejects anything larger.
     */
    public static final double DEFAULT_POSITION_TOLERANCE = 0.25;

    private CapturePlumbing() {
    }

    /** The camera eye for a requested feet position: {@code feet + eye_height}. */
    public static double[] eyeFromFeet(double[] feet, double eyeHeight) {
        return new double[] {feet[0], feet[1] + eyeHeight, feet[2]};
    }

    public static double distance(double[] first, double[] second) {
        double dx = first[0] - second[0];
        double dy = first[1] - second[1];
        double dz = first[2] - second[2];
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /** True when every component is a finite number. */
    public static boolean isFiniteTriple(double[] values) {
        if (values == null || values.length != 3) {
            return false;
        }
        for (double value : values) {
            if (!Double.isFinite(value)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Split a JSON triple such as {@code actual.eye_position}; empty when absent or
     * not exactly three finite numbers.
     */
    public static Optional<double[]> readTriple(JsonObject object, String field) {
        JsonElement element = object.get(field);
        if (element == null || !element.isJsonArray()) {
            return Optional.empty();
        }
        var array = element.getAsJsonArray();
        if (array.size() != 3) {
            return Optional.empty();
        }
        double[] values = new double[3];
        for (int axis = 0; axis < 3; axis++) {
            JsonElement value = array.get(axis);
            if (value == null || !value.isJsonPrimitive()
                    || !value.getAsJsonPrimitive().isNumber()) {
                return Optional.empty();
            }
            values[axis] = value.getAsDouble();
        }
        return isFiniteTriple(values) ? Optional.of(values) : Optional.empty();
    }

    /**
     * Remove a stale screenshot target before a new capture.
     *
     * <p>An old file with the same name would otherwise satisfy the "the file
     * exists and is non-empty" check before the asynchronous writer had produced
     * anything. Only a regular file is removed; a directory or other non-file is
     * reported instead of being deleted.
     *
     * @return true when a stale file was removed
     */
    public static boolean removeStaleScreenshot(Path target) throws IOException {
        if (target == null) {
            return false;
        }
        if (Files.isRegularFile(target)) {
            Files.delete(target);
            return true;
        }
        return false;
    }

    /**
     * Accept a screenshot only once its callback signalled completion and the
     * final file is a non-empty regular file.
     */
    public static boolean isScreenshotAccepted(Path target, boolean callbackSignalled) {
        if (!callbackSignalled || target == null) {
            return false;
        }
        try {
            return Files.isRegularFile(target) && Files.size(target) > 0L;
        } catch (IOException error) {
            return false;
        }
    }

    /**
     * True when a settled camera is within tolerance of the requested pose.
     *
     * @param actualFeet where the player actually stands
     * @param actualEye  the rendered client's camera position
     * @param requestedFeet the plan's {@code CameraModel.position}
     * @param requestedEye  {@code requestedFeet + eye_height}
     * @param tolerance blocks; the plan's declared bound
     */
    public static boolean withinTolerance(double[] actualFeet, double[] actualEye,
                                          double[] requestedFeet, double[] requestedEye,
                                          double tolerance) {
        if (!isFiniteTriple(actualFeet) || !isFiniteTriple(actualEye)
                || !isFiniteTriple(requestedFeet) || !isFiniteTriple(requestedEye)) {
            return false;
        }
        return distance(actualFeet, requestedFeet) <= tolerance
                && distance(actualEye, requestedEye) <= tolerance;
    }

    /**
     * Record whether a serialized payload really represents exported ground truth.
     * Frame failures are tracked separately by {@code run_failed} and
     * {@code all_frames_captured}; they do not invalidate a successful block export.
     */
    public static boolean isGroundTruthExported(boolean payloadWriteSucceeded,
                                                 boolean failureSentinel) {
        return payloadWriteSucceeded && !failureSentinel;
    }

    /**
     * Write JSON durably: a sibling {@code .tmp} file, then an atomic replace.
     *
     * <p>A crash can therefore leave the previous complete file or the new complete
     * file, never a half-written one. When the filesystem refuses an atomic move the
     * replace falls back to a plain move, which is still far safer than truncating
     * the destination in place.
     */
    public static void writeJsonAtomic(Path target, String json) throws IOException {
        Path parent = target.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temporary = target.resolveSibling(target.getFileName() + ".tmp");
        Files.writeString(temporary, json, StandardCharsets.UTF_8);
        try {
            Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /** Same-directory marker a consumed or rejected {@code armed.json} becomes. */
    public static Path disarmMarker(Path armedPath, String reason) {
        String normalized = reason == null || reason.isBlank()
                ? "consumed" : reason.trim().toLowerCase(Locale.ROOT);
        return armedPath.resolveSibling(armedPath.getFileName() + "." + normalized);
    }

    /**
     * Disarm by an atomic same-directory rename, then best-effort delete the marker.
     *
     * <p>A plain {@code deleteIfExists} cannot meet the contract: a crash between
     * the decision and the unlink would leave a live arming file behind. The rename
     * is the atomic step; the deletion afterwards is only tidiness, so a failure to
     * delete is reported rather than thrown.
     *
     * @return the marker path that now holds the disarmed arming file, or empty when
     *         no arming file existed
     */
    public static Optional<Path> disarmAtomic(Path armedPath, String reason)
            throws IOException {
        if (armedPath == null || !Files.exists(armedPath)) {
            return Optional.empty();
        }
        Path marker = disarmMarker(armedPath, reason);
        if (Files.exists(marker)) {
            Files.delete(marker);
        }
        try {
            Files.move(armedPath, marker, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(armedPath, marker, StandardCopyOption.REPLACE_EXISTING);
        }
        try {
            Files.deleteIfExists(marker);
        } catch (IOException ignored) {
            // The arming file is already atomically out of the way; leaving the
            // marker behind is harmless and must not mask the disarm.
        }
        return Optional.of(marker);
    }
}
