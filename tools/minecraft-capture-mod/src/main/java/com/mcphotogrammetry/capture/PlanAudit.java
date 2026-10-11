package com.mcphotogrammetry.capture;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Minecraft-free validation of an armed capture plan.
 *
 * <p>Everything here is deliberately pure so the safety rules - bounded jobs,
 * bounded region volume, filename safety, path containment, plan digest - can be
 * unit tested on a plain JVM. The client-side controller only decides <em>when</em>
 * to apply a job; this class decides whether a job may exist at all.
 */
public final class PlanAudit {

    public static final int SUPPORTED_SCHEMA_VERSION = 1;
    public static final int SUPPORTED_MOD_CONTRACT_VERSION = 1;
    public static final int MAX_JOBS = 64;
    /** Bounded so one ground-truth export cannot stall a client tick for minutes. */
    public static final long MAX_REGION_VOLUME = 250_000L;
    public static final int MAX_SETTLE_TICKS = 600;
    /** Requested-vs-actual heading tolerance in degrees; yaw uses a circular delta. */
    public static final double HEADING_TOLERANCE_DEGREES = 1.0;
    public static final String PLAN_SUBDIRECTORY = "mc-photo-capture";

    private PlanAudit() {
    }

    /** A plan was missing, malformed, or outside the supported contract. */
    public static final class PlanRejected extends Exception {
        public PlanRejected(String message) {
            super(message);
        }
    }

    public record CaptureSettings(double gamma, int timeOfDay, String weather, boolean hudHidden,
                                  String resourcePack, String shaderPack) {
    }

    public record CaptureJob(String frameId, String filename, double x, double y, double z,
                             double yawDegrees, double pitchDegrees, double fovDegrees,
                             double eyeHeight, double rollDegrees, double targetX, double targetY,
                             double targetZ, int width, int height, int settleTicks,
                             CaptureSettings settings) {
    }

    public record AuditedPlan(int schemaVersion, int modContractVersion, String planId,
                              String sceneId, String datasetId, String worldIdentity,
                              String stagingToken, int[] minimum, int[] maximum,
                              List<CaptureJob> jobs, boolean contextExportOnly) {

        public long volume() {
            long total = 1L;
            for (int axis = 0; axis < 3; axis++) {
                total *= (long) (maximum[axis] - minimum[axis] + 1);
            }
            return total;
        }
    }

    public static AuditedPlan parse(String json) throws PlanRejected {
        JsonObject root = object(parseJson(json), "plan root");
        requireVersion(root, "schema_version", SUPPORTED_SCHEMA_VERSION, "plan");
        requireVersion(root, "mod_contract_version", SUPPORTED_MOD_CONTRACT_VERSION, "plan");

        String planId = requiredString(root, "plan_id");
        String sceneId = requiredString(root, "scene_id");
        String datasetId = requiredString(root, "dataset_id");
        String worldIdentity = requiredString(root, "world_identity");
        String stagingToken = requiredDigest(root, "staging_token");

        JsonObject region = object(root.get("region"), "plan region");
        int[] minimum = intTriple(region.get("min"), "region min");
        int[] maximum = intTriple(region.get("max"), "region max");
        for (int axis = 0; axis < 3; axis++) {
            if (maximum[axis] < minimum[axis]) {
                throw new PlanRejected("region maximum is below its minimum on axis " + axis);
            }
        }
        long volume = 1L;
        for (int axis = 0; axis < 3; axis++) {
            volume *= (long) (maximum[axis] - minimum[axis] + 1);
        }
        if (volume > MAX_REGION_VOLUME) {
            throw new PlanRejected("region volume " + volume + " exceeds " + MAX_REGION_VOLUME);
        }

        JsonArray rawJobs = array(root.get("jobs"), "plan jobs");
        boolean contextExportOnly = false;
        if (root.has("context_export_only")) {
            JsonElement contextFlag = root.get("context_export_only");
            if (contextFlag == null || !contextFlag.isJsonPrimitive()
                    || !contextFlag.getAsJsonPrimitive().isBoolean()) {
                throw new PlanRejected("context_export_only must be a boolean");
            }
            contextExportOnly = contextFlag.getAsBoolean();
        }
        if (contextExportOnly && !rawJobs.isEmpty()) {
            throw new PlanRejected("context-export-only plans must contain zero jobs");
        }
        if (!contextExportOnly && rawJobs.isEmpty()) {
            throw new PlanRejected("a plan must contain at least one job unless "
                    + "context_export_only is true");
        }
        if (rawJobs.size() > MAX_JOBS) {
            throw new PlanRejected("plan exceeds " + MAX_JOBS + " jobs");
        }

        List<CaptureJob> jobs = new ArrayList<>(rawJobs.size());
        Set<String> frameIds = new HashSet<>();
        for (JsonElement element : rawJobs) {
            CaptureJob job = parseJob(object(element, "job"));
            if (!frameIds.add(job.frameId())) {
                throw new PlanRejected("duplicate frame_id " + job.frameId());
            }
            jobs.add(job);
        }
        return new AuditedPlan(SUPPORTED_SCHEMA_VERSION, SUPPORTED_MOD_CONTRACT_VERSION, planId,
                sceneId, datasetId, worldIdentity, stagingToken, minimum, maximum,
                List.copyOf(jobs), contextExportOnly);
    }

    private static CaptureJob parseJob(JsonObject job) throws PlanRejected {
        String frameId = requiredString(job, "frame_id");
        if (!isSafeFrameId(frameId)) {
            throw new PlanRejected("frame_id " + frameId + " is not filename-safe");
        }
        JsonArray resolution = array(job.get("resolution"), "job resolution");
        if (resolution.size() != 2) {
            throw new PlanRejected("job resolution must have two values");
        }
        int width = resolution.get(0).getAsInt();
        int height = resolution.get(1).getAsInt();
        if (width <= 0 || height <= 0 || width > 7680 || height > 4320) {
            throw new PlanRejected("job resolution is out of range: " + width + "x" + height);
        }
        JsonArray target = array(job.get("target"), "job target");
        if (target.size() != 3) {
            throw new PlanRejected("job target must have three values");
        }
        int settleTicks = job.has("settle_ticks") ? job.get("settle_ticks").getAsInt() : 20;
        if (settleTicks < 0 || settleTicks > MAX_SETTLE_TICKS) {
            throw new PlanRejected("settle_ticks out of range: " + settleTicks);
        }
        JsonObject settings = object(job.get("settings"), "job settings");
        String weather = settings.has("weather") ? settings.get("weather").getAsString() : "clear";
        if (!weather.equals("clear") && !weather.equals("rain") && !weather.equals("thunder")) {
            throw new PlanRejected("unsupported weather " + weather);
        }
        int timeOfDay = settings.has("time_of_day") ? settings.get("time_of_day").getAsInt() : 6000;
        if (timeOfDay < 0 || timeOfDay >= 24000) {
            throw new PlanRejected("time_of_day out of range: " + timeOfDay);
        }
        CaptureSettings captureSettings = new CaptureSettings(
                settings.has("gamma") ? settings.get("gamma").getAsDouble() : 0.0,
                timeOfDay,
                weather,
                settings.has("hud_hidden") && settings.get("hud_hidden").getAsBoolean(),
                optionalString(settings, "resource_pack"),
                optionalString(settings, "shader_pack"));

        JsonObject camera = object(job.get("camera"), "job camera");
        JsonArray position = array(camera.get("position"), "camera position");
        if (position.size() != 3) {
            throw new PlanRejected("camera position must have three values");
        }
        double yaw = finite(camera, "yaw_degrees");
        double pitch = finite(camera, "pitch_degrees");
        double fov = finite(camera, "fov_degrees");
        double eyeHeight = camera.has("eye_height") ? finite(camera, "eye_height") : 1.62;
        double roll = camera.has("roll_degrees") ? finite(camera, "roll_degrees") : 0.0;
        if (pitch < -90.0 || pitch > 90.0) {
            throw new PlanRejected("camera pitch out of range: " + pitch);
        }
        // Minecraft's client has no roll control, so a plan that asks for one
        // cannot be honoured honestly. Reject it instead of silently capturing a
        // different pose than the frozen ground truth claims.
        if (roll != 0.0) {
            throw new PlanRejected("camera roll is not supported by the client: " + roll);
        }
        if (fov < 1.0 || fov >= 180.0) {
            throw new PlanRejected("camera fov out of range: " + fov);
        }
        for (int axis = 0; axis < 3; axis++) {
            double value = position.get(axis).getAsDouble();
            if (!Double.isFinite(value)) {
                throw new PlanRejected("camera position is not finite");
            }
        }
        return new CaptureJob(frameId, frameId + ".png",
                position.get(0).getAsDouble(), position.get(1).getAsDouble(),
                position.get(2).getAsDouble(), yaw, pitch, fov, eyeHeight, roll,
                target.get(0).getAsDouble(), target.get(1).getAsDouble(),
                target.get(2).getAsDouble(), width, height, settleTicks, captureSettings);
    }

    /**
     * Smallest absolute angle between two headings, in degrees.
     *
     * <p>Yaw wraps, so a client reporting 350 degrees where the plan asked for 10 is
     * twenty degrees away, not three hundred and forty. A plain subtraction would
     * report the whole turn.
     */
    public static double angleDeltaDegrees(double first, double second) {
        // Round before the modulo so a full turn that floating point represents as
        // 360.00000000000006 lands on 0 rather than on a whole turn.
        double difference = Math.abs(Math.round(((first - second) % 360.0) * 1e9) / 1e9) % 360.0;
        if (difference > 180.0) {
            difference = 360.0 - difference;
        }
        return difference;
    }

    /** True when a recorded heading is within tolerance of the requested one. */
    public static boolean headingWithinTolerance(double actual, double requested,
                                                 double tolerance) {
        return Double.isFinite(actual) && Double.isFinite(requested)
                && angleDeltaDegrees(actual, requested) <= tolerance;
    }

    /** True when a frame id can be used verbatim as a screenshot file name. */
    public static boolean isSafeFrameId(String value) {
        if (value == null || value.isEmpty() || value.length() > 96) {
            return false;
        }
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            boolean allowed = (character >= 'a' && character <= 'z')
                    || (character >= 'A' && character <= 'Z')
                    || (character >= '0' && character <= '9')
                    || character == '-' || character == '_';
            if (!allowed) {
                return false;
            }
        }
        return true;
    }

    /** Resolve a plan-relative path and refuse anything that escapes {@code root}. */
    public static Path resolveContained(Path root, String relative) throws PlanRejected {
        if (relative == null || relative.isBlank()) {
            throw new PlanRejected("plan path must not be empty");
        }
        if (relative.contains("..") || relative.startsWith("/") || relative.startsWith("\\")) {
            throw new PlanRejected("plan path must be relative and free of '..': " + relative);
        }
        Path base = root.toAbsolutePath().normalize();
        Path candidate = base.resolve(relative.replace('\\', '/')).normalize();
        if (!candidate.startsWith(base)) {
            throw new PlanRejected("plan path escapes the game directory: " + relative);
        }
        return candidate;
    }

    public static String sha256Hex(byte[] bytes) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(bytes);
            StringBuilder builder = new StringBuilder(hash.length * 2);
            for (byte value : hash) {
                builder.append(String.format(Locale.ROOT, "%02x", value));
            }
            return builder.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
    }

    public static void verifyDigest(byte[] planBytes, String expectedSha256) throws PlanRejected {
        if (expectedSha256 == null || !expectedSha256.matches("[0-9a-fA-F]{64}")) {
            throw new PlanRejected("armed.json must carry a 64-character plan_sha256");
        }
        String actual = sha256Hex(planBytes);
        if (!actual.equalsIgnoreCase(expectedSha256)) {
            throw new PlanRejected("plan digest mismatch: armed.json says " + expectedSha256
                    + " but the plan file hashes to " + actual);
        }
    }

    private static JsonElement parseJson(String json) throws PlanRejected {
        try {
            return JsonParser.parseString(json);
        } catch (RuntimeException error) {
            throw new PlanRejected("plan is not valid JSON: " + error.getMessage());
        }
    }

    private static JsonObject object(JsonElement element, String label) throws PlanRejected {
        if (element == null || !element.isJsonObject()) {
            throw new PlanRejected(label + " must be a JSON object");
        }
        return element.getAsJsonObject();
    }

    private static JsonArray array(JsonElement element, String label) throws PlanRejected {
        if (element == null || !element.isJsonArray()) {
            throw new PlanRejected(label + " must be a JSON array");
        }
        return element.getAsJsonArray();
    }

    private static void requireVersion(JsonObject root, String field, int expected, String label)
            throws PlanRejected {
        if (!root.has(field)) {
            throw new PlanRejected(label + " is missing " + field);
        }
        int actual = root.get(field).getAsInt();
        if (actual != expected) {
            throw new PlanRejected(label + " has unsupported " + field + " " + actual
                    + " (this mod supports " + expected + ")");
        }
    }

    private static String requiredString(JsonObject object, String field) throws PlanRejected {
        if (!object.has(field) || object.get(field).isJsonNull()) {
            throw new PlanRejected("missing required field " + field);
        }
        String value = object.get(field).getAsString();
        if (value.isBlank()) {
            throw new PlanRejected("field " + field + " must not be blank");
        }
        return value;
    }

    /**
     * A required 64-character lowercase hex digest.
     *
     * <p>The plan's {@code staging_token} is validated this way: a plan without a
     * real staged-world fingerprint must never run, and the inert all-zero token a
     * synthetic plan carries is refused here rather than only at the world check.
     */
    private static String requiredDigest(JsonObject object, String field) throws PlanRejected {
        String value = requiredString(object, field).toLowerCase(Locale.ROOT);
        if (value.length() != 64) {
            throw new PlanRejected(field + " must be a 64-character SHA-256 digest");
        }
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            boolean hex = (character >= '0' && character <= '9')
                    || (character >= 'a' && character <= 'f');
            if (!hex) {
                throw new PlanRejected(field + " must be a lowercase hex digest");
            }
        }
        if (StagingMarker.UNSTAGED_TOKEN.equals(value)) {
            throw new PlanRejected("plan carries the inert unstaged staging_token; "
                    + "produce the plan from a staged world copy");
        }
        return value;
    }

    private static String optionalString(JsonObject object, String field) {
        if (!object.has(field) || object.get(field).isJsonNull()) {
            return null;
        }
        return object.get(field).getAsString();
    }

    private static int[] intTriple(JsonElement element, String label) throws PlanRejected {
        JsonArray array = array(element, label);
        if (array.size() != 3) {
            throw new PlanRejected(label + " must have three values");
        }
        int[] values = new int[3];
        for (int axis = 0; axis < 3; axis++) {
            values[axis] = array.get(axis).getAsInt();
        }
        return values;
    }

    private static double finite(JsonObject object, String field) throws PlanRejected {
        if (!object.has(field)) {
            throw new PlanRejected("camera is missing " + field);
        }
        double value = object.get(field).getAsDouble();
        if (!Double.isFinite(value)) {
            throw new PlanRejected("camera " + field + " is not finite");
        }
        return value;
    }

    static String utf8(byte[] bytes) {
        return new String(bytes, StandardCharsets.UTF_8);
    }
}
