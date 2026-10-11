package com.mcphotogrammetry.capture;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

/**
 * Proves that the world the client opened is the world Python staged.
 *
 * <p>A level name alone is not identity, and neither is the save <em>folder</em>
 * name: the integrated server reports the level name from {@code level.dat} while
 * the folder may be called something else entirely, so resolving
 * {@code <game>/saves/<LevelName>} can point at a directory Minecraft is not even
 * using. Nor is a content hash of the live tree safe: Minecraft creates and updates
 * files (session lock, player data, region data) after the copy, so a valid older
 * copy would fail the check and hashing a multi-GB save on the client thread would
 * freeze the game.
 *
 * <p>The stager therefore writes
 * {@code <copied world root>/.mc-photo-capture-staging.json} containing a
 * plan-required token (the SHA-256 content fingerprint of the source save) plus the
 * world identity. The mod resolves the real open save root with the 1.21.5
 * integrated-server save-path API and reads that marker before it mutates anything;
 * the live tree is never rehashed. The game-directory
 * {@code config/mc-photo-capture/staging.json} copy is retained only as provenance
 * and is never accepted as proof of the opened world.
 *
 * <p>Everything in this class is Minecraft-free so the rule is unit-testable on a
 * plain JVM. The row format must stay byte-identical to the Python
 * {@code staging_fingerprint} helper: {@code sha256<TAB>size<TAB>relative\n} per
 * file, sorted by relative path.
 */
public final class StagingMarker {

    public static final int SUPPORTED_SCHEMA_VERSION = 1;
    /** Provenance-only copy beside the game configuration; never proof of a world. */
    public static final String MARKER_RELATIVE = "config/mc-photo-capture/staging.json";
    /** The authoritative marker, written into the copied world's own root. */
    public static final String WORLD_MARKER_NAME = ".mc-photo-capture-staging.json";
    /** The inert token a synthetic plan carries; production preparation never uses it. */
    public static final String UNSTAGED_TOKEN = "0".repeat(64);

    /** A staging marker was absent, malformed, or did not describe this world. */
    public static final class StagingMismatch extends Exception {
        public StagingMismatch(String message) {
            super(message);
        }
    }

    /** The validated marker contents, plus where it was read from. */
    public record Marker(int schemaVersion, String token, String sourceFingerprint,
                         String worldIdentity, String datasetId, String sceneId,
                         String planId, Path path) {
    }

    private StagingMarker() {
    }

    /** The provenance-only marker file inside a game directory. */
    public static Path markerPath(Path gameDir) {
        return gameDir.resolve(MARKER_RELATIVE);
    }

    /** The authoritative staging marker inside one save root (or copied world root). */
    public static Path worldMarkerPath(Path worldRoot) {
        return worldRoot.resolve(WORLD_MARKER_NAME);
    }

    /**
     * Parse a marker and require it to describe the plan exactly.
     *
     * @param jsonText the raw marker file
     * @param expectedToken the plan's {@code staging_token}
     * @param expectedWorldIdentity the plan's {@code world_identity}
     * @param markerLocation the file the text came from, recorded for diagnostics
     */
    public static Marker parseAndValidate(String jsonText, String expectedToken,
                                          String expectedWorldIdentity, Path markerLocation)
            throws StagingMismatch {
        JsonObject root = parseObject(jsonText);
        int version = requireInt(root, "schema_version");
        if (version != SUPPORTED_SCHEMA_VERSION) {
            throw new StagingMismatch("staging marker has unsupported schema_version " + version
                    + " (this mod supports " + SUPPORTED_SCHEMA_VERSION + ")");
        }
        String token = requireHex(root, "token");
        String fingerprint = requireHex(root, "source_fingerprint");
        String worldIdentity = requireText(root, "world_identity");
        if (expectedToken == null || expectedToken.isBlank()) {
            throw new StagingMismatch("the plan does not carry a staging token");
        }
        if (UNSTAGED_TOKEN.equalsIgnoreCase(expectedToken)) {
            throw new StagingMismatch(
                    "the plan carries the inert unstaged token; run prepare_capture_dataset.py "
                            + "against the staged world instead of arming a synthetic plan");
        }
        if (!token.equalsIgnoreCase(expectedToken)) {
            throw new StagingMismatch("staging token mismatch: the plan requires "
                    + expectedToken + " but " + markerLocation + " carries " + token);
        }
        if (!worldIdentity.equals(expectedWorldIdentity)) {
            throw new StagingMismatch("staging marker world identity '" + worldIdentity
                    + "' does not match the plan's '" + expectedWorldIdentity + "'");
        }
        return new Marker(version, token, fingerprint, worldIdentity,
                optionalText(root, "dataset_id"), optionalText(root, "scene_id"),
                optionalText(root, "plan_id"), markerLocation);
    }

    /** Read and validate the provenance marker for one game directory. */
    public static Marker readAndValidate(Path gameDir, String expectedToken,
                                         String expectedWorldIdentity) throws StagingMismatch {
        return readMarker(markerPath(gameDir), expectedToken, expectedWorldIdentity,
                "the game directory was not staged by prepare_capture_dataset.py");
    }

    /**
     * Read and validate the authoritative marker inside the world the client opened.
     *
     * <p>{@code worldRoot} must be the real open save root resolved through the
     * integrated server ({@code MinecraftServer#getSavePath(WorldSavePath#ROOT)}),
     * which is correct even when the save folder name differs from the level name.
     * The world's own files are never rehashed: only this marker is read, so a world
     * Minecraft has since updated still validates.
     */
    public static Marker readAndValidateWorld(Path worldRoot, String expectedToken,
                                              String expectedWorldIdentity)
            throws StagingMismatch {
        return readMarker(worldMarkerPath(worldRoot), expectedToken, expectedWorldIdentity,
                "this world was not staged by prepare_capture_dataset.py");
    }

    private static Marker readMarker(Path location, String expectedToken,
                                     String expectedWorldIdentity, String missingHint)
            throws StagingMismatch {
        if (!Files.isRegularFile(location)) {
            throw new StagingMismatch("no staging marker at " + location + "; " + missingHint);
        }
        try {
            return parseAndValidate(Files.readString(location, StandardCharsets.UTF_8),
                    expectedToken, expectedWorldIdentity, location);
        } catch (IOException error) {
            throw new StagingMismatch("could not read " + location + ": " + error);
        }
    }

    /**
     * SHA-256 content fingerprint of a save tree, matching the Python helper.
     *
     * <p>Files are visited in sorted relative-path order; {@code session.lock} is
     * skipped because it is not part of the staged world. The fingerprint covers
     * content, size, and the relative path, so neither a renamed file nor an
     * edited one can collide.
     */
    public static String fingerprint(Path root) throws IOException {
        Path base = root.toAbsolutePath().normalize();
        if (!Files.isDirectory(base)) {
            throw new IOException("world root is not a directory: " + base);
        }
        List<Path> files = new ArrayList<>();
        try (Stream<Path> walk = Files.walk(base)) {
            walk.filter(Files::isRegularFile)
                    .filter(path -> !path.getFileName().toString().equals("session.lock"))
                    .forEach(files::add);
        }
        List<String> rows = new ArrayList<>(files.size());
        for (Path file : files) {
            String relative = base.relativize(file).toString().replace('\\', '/');
            rows.add(sha256(file) + "\t" + Files.size(file) + "\t" + relative);
        }
        rows.sort(Comparator.naturalOrder());
        StringBuilder builder = new StringBuilder();
        for (String row : rows) {
            builder.append(row).append('\n');
        }
        return sha256(builder.toString().getBytes(StandardCharsets.UTF_8));
    }

    /** SHA-256 of a file's bytes, streamed so a multi-GB save stays affordable. */
    public static String fileSha256(Path file) throws IOException {
        return sha256(file);
    }

    private static String sha256(Path file) throws IOException {
        MessageDigest digest = digest();
        byte[] buffer = new byte[1 << 20];
        try (InputStream stream = Files.newInputStream(file)) {
            int read;
            while ((read = stream.read(buffer)) > 0) {
                digest.update(buffer, 0, read);
            }
        }
        return hex(digest.digest());
    }

    private static String sha256(byte[] bytes) {
        MessageDigest digest = digest();
        digest.update(bytes);
        return hex(digest.digest());
    }

    private static MessageDigest digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder builder = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            builder.append(String.format(Locale.ROOT, "%02x", value));
        }
        return builder.toString();
    }

    private static JsonObject parseObject(String jsonText) throws StagingMismatch {
        try {
            JsonElement parsed = JsonParser.parseString(jsonText);
            if (parsed == null || !parsed.isJsonObject()) {
                throw new StagingMismatch("staging marker must be a JSON object");
            }
            return parsed.getAsJsonObject();
        } catch (RuntimeException error) {
            throw new StagingMismatch("staging marker is not valid JSON: " + error.getMessage());
        }
    }

    private static int requireInt(JsonObject object, String field) throws StagingMismatch {
        JsonElement element = object.get(field);
        if (element == null || element.isJsonNull() || !element.isJsonPrimitive()
                || !element.getAsJsonPrimitive().isNumber()) {
            throw new StagingMismatch("staging marker is missing a numeric " + field);
        }
        return element.getAsInt();
    }

    private static String requireText(JsonObject object, String field) throws StagingMismatch {
        String value = optionalText(object, field);
        if (value == null || value.isBlank()) {
            throw new StagingMismatch("staging marker is missing " + field);
        }
        return value;
    }

    private static String optionalText(JsonObject object, String field) {
        JsonElement element = object.get(field);
        if (element == null || element.isJsonNull()) {
            return null;
        }
        return element.getAsString();
    }

    private static String requireHex(JsonObject object, String field) throws StagingMismatch {
        String value = requireText(object, field).toLowerCase(Locale.ROOT);
        if (value.length() != 64) {
            throw new StagingMismatch("staging marker " + field
                    + " must be a 64-character SHA-256 digest");
        }
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            boolean hex = (character >= '0' && character <= '9')
                    || (character >= 'a' && character <= 'f');
            if (!hex) {
                throw new StagingMismatch("staging marker " + field
                        + " must be a lowercase hex digest");
            }
        }
        return value;
    }
}
