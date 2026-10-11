package com.mcphotogrammetry.capture;

import com.google.gson.JsonElement;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.fabricmc.loader.api.FabricLoader;
import net.fabricmc.loader.api.ModContainer;
import net.minecraft.client.MinecraftClient;
import net.minecraft.resource.ResourcePackProfile;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL20;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.Collection;
import java.util.List;
import java.util.Optional;

/**
 * Minecraft-side observations for the advanced factory provenance sidecar.
 * Missing adapters are represented explicitly; requested settings are never
 * substituted for runtime observations.
 */
public final class RuntimeRenderProvenance {
    public static final int SCHEMA_VERSION = 1;
    private static final int MAX_CANONICAL_BYTES = 1_048_576;
    private static final int MAX_CANONICAL_DEPTH = 64;
    private static final int MAX_CANONICAL_OBJECT_KEYS = 8_192;
    private static final int MAX_LOADED_MODS = 2_048;
    private static final int MAX_RESOURCE_PROFILES = 8_192;

    private RuntimeRenderProvenance() { }

    public static JsonObject create(FabricLoader loader, JsonObject expectation) {
        JsonObject root = new JsonObject();
        root.addProperty("schema_version", SCHEMA_VERSION);
        for (String key : List.of("attempt_nonce", "plan_id", "plan_sha256",
                "staging_token", "capture_mod_sha256", "capture_mod_staged_sha256",
                "capture_mod_staged_path", "mode", "expectation_sha256")) {
            if (!expectation.has(key) || expectation.get(key).isJsonNull()) {
                throw new IllegalArgumentException("provenance expectation lacks " + key);
            }
            root.add(key, expectation.get(key).deepCopy());
        }
        for (String key : List.of("capture_mod_source_sha256", "capture_mod_source_path",
                "capture_mod_expected_sha256")) {
            root.add(key, expectation.has(key) ? expectation.get(key).deepCopy()
                    : com.google.gson.JsonNull.INSTANCE);
        }
        for (String key : List.of("dataset_plan_sha256", "scene_id", "world_identity",
                "staging_game_dir")) {
            root.add(key, expectation.has(key) ? expectation.get(key).deepCopy()
                    : com.google.gson.JsonNull.INSTANCE);
        }
        root.addProperty("mode", expectation.get("mode").getAsString());
        root.addProperty("expectation_sha256", expectation.get("expectation_sha256").getAsString());
        JsonObject runtime = new JsonObject();
        runtime.addProperty("minecraft_version", modVersion(loader, "minecraft"));
        runtime.addProperty("fabric_loader_version", modVersion(loader, "fabricloader"));
        runtime.addProperty("java_version", System.getProperty("java.version", "unknown"));
        runtime.addProperty("java_vendor", System.getProperty("java.vendor", "unknown"));
        runtime.addProperty("capture_mod_sha256", expectation.get("capture_mod_sha256").getAsString());
        runtime.add("capture_mod_source_sha256", expectation.get("capture_mod_source_sha256").deepCopy());
        runtime.add("capture_mod_staged_sha256", expectation.get("capture_mod_staged_sha256").deepCopy());
        runtime.add("capture_mod_loaded_sha256", com.google.gson.JsonNull.INSTANCE);
        runtime.addProperty("capture_mod_digest_source", "runner_verified_staged_artifact");
        runtime.addProperty("loaded_capture_mod_origin_status", "unbound");
        runtime.addProperty("identity_status", "incomplete");
        runtime.addProperty("client_artifacts_status", "content_hash_adapter_unavailable");
        JsonArray mods = new JsonArray();
        JsonArray artifacts = new JsonArray();
        for (ModContainer mod : loader.getAllMods()) {
            if (mods.size() >= MAX_LOADED_MODS) {
                runtime.addProperty("loaded_mods_status", "unavailable_entry_cap_exceeded");
                runtime.addProperty("client_artifacts_status", "unavailable_entry_cap_exceeded");
                break;
            }
            JsonObject row = new JsonObject();
            String id = mod.getMetadata().getId();
            String version = mod.getMetadata().getVersion().getFriendlyString();
            row.addProperty("id", id);
            row.addProperty("version", version);
            JsonArray origins = new JsonArray();
            mod.getOrigin().getPaths().stream().map(path -> path.toAbsolutePath().normalize())
                    .map(Object::toString).sorted().forEach(origins::add);
            row.add("origins", origins);
            row.addProperty("content_sha256_status", "unavailable");
            mods.add(row);
            JsonObject artifact = new JsonObject();
            artifact.addProperty("id", id);
            artifact.addProperty("version", version);
            artifact.add("origins", origins.deepCopy());
            artifact.add("sha256", com.google.gson.JsonNull.INSTANCE);
            artifact.addProperty("sha256_status", "unavailable_bounded_off_thread_hasher_not_installed");
            artifacts.add(artifact);
        }
        runtime.add("loaded_mods", mods);
        runtime.add("client_artifacts", artifacts);
        root.add("runtime", runtime);
        root.addProperty("resource_status", "unknown_effective_origins_and_manifests");
        root.addProperty("shader_status", "unknown_no_supported_shader_state_adapter");
        root.addProperty("gl_status", "partial_until_capture_thread_probe");
        root.add("resource_packs", new JsonArray());
        JsonObject certification = new JsonObject();
        certification.addProperty("status", "unavailable");
        JsonArray reasons = new JsonArray();
        reasons.add("loaded_client_artifact_hashes_unavailable");
        reasons.add("effective_resource_manifests_and_reload_epoch_unavailable");
        reasons.add("shader_program_identity_unavailable");
        reasons.add("framebuffer_readback_epoch_unbound");
        certification.add("unavailable_reasons", reasons);
        root.add("certification", certification);
        root.add("frames", new JsonArray());
        return root;
    }

    private static String modVersion(FabricLoader loader, String id) {
        Optional<ModContainer> container = loader.getModContainer(id);
        return container.map(mod -> mod.getMetadata().getVersion().getFriendlyString())
                .orElse("unavailable");
    }

    /** Preserve ResourcePackManager's selected precedence; do not sort IDs. */
    public static JsonArray activeProfiles(MinecraftClient client) {
        JsonArray values = new JsonArray();
        try {
            Collection<ResourcePackProfile> profiles = client.getResourcePackManager().getEnabledProfiles();
            for (ResourcePackProfile profile : profiles) {
                if (values.size() >= MAX_RESOURCE_PROFILES) {
                    throw new IllegalStateException("resource profile count exceeds provenance cap");
                }
                JsonObject row = new JsonObject();
                row.addProperty("id", profile.getId());
                row.addProperty("name", profile.getDisplayName().getString());
                row.addProperty("content_manifest_status", "unavailable_effective_origin_adapter");
                values.add(row);
            }
        } catch (RuntimeException | LinkageError failure) {
            JsonObject unavailable = new JsonObject();
            unavailable.addProperty("status", "unavailable");
            unavailable.addProperty("reason", "ordered_resource_profile_adapter_failed");
            values.add(unavailable);
        }
        return values;
    }

    /** Snapshot the values available on the client thread before recorder submission. */
    public static JsonObject captureSnapshot(MinecraftClient client, String frameId,
                                             JsonArray orderedPacks) {
        JsonObject snapshot = new JsonObject();
        snapshot.addProperty("frame_id", frameId);
        snapshot.add("ordered_resource_packs", orderedPacks.deepCopy());
        snapshot.addProperty("resource_epoch_status", "unbound");
        snapshot.addProperty("capture_epoch_status", "capture_epoch_unbound");
        JsonObject gl = new JsonObject();
        try {
            gl.addProperty("vendor", nullable(GL11.glGetString(GL11.GL_VENDOR)));
            gl.addProperty("renderer", nullable(GL11.glGetString(GL11.GL_RENDERER)));
            gl.addProperty("version", nullable(GL11.glGetString(GL11.GL_VERSION)));
            gl.addProperty("glsl_version", nullable(GL11.glGetString(GL20.GL_SHADING_LANGUAGE_VERSION)));
            gl.addProperty("framebuffer_width", client.getFramebuffer().textureWidth);
            gl.addProperty("framebuffer_height", client.getFramebuffer().textureHeight);
            gl.addProperty("status", "observed_at_recorder_submission_not_readback_bound");
        } catch (RuntimeException | LinkageError failure) {
            gl.addProperty("status", "unavailable_gl_context_or_adapter");
        }
        snapshot.add("gl", gl);
        snapshot.addProperty("shader_status", "unknown_no_supported_shader_state_adapter");
        return snapshot;
    }

    private static String nullable(String value) {
        return value == null ? "unavailable" : value;
    }

    public static String digest(JsonObject value) {
        return sha256(canonical(value).getBytes(StandardCharsets.UTF_8));
    }

    public static String sha256(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (Exception impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    public static JsonObject seal(JsonObject value) {
        if (value.size() > MAX_CANONICAL_OBJECT_KEYS) {
            throw new IllegalArgumentException("runtime/render provenance object exceeds key cap");
        }
        JsonObject copy = new JsonObject();
        for (var entry : value.entrySet()) {
            if (!entry.getKey().equals("seal_sha256")) copy.add(entry.getKey(), entry.getValue());
        }
        copy.addProperty("seal_sha256", digest(copy));
        return copy;
    }

    public static String canonical(JsonObject value) {
        measureCanonical(value, 0, new long[] {0});
        StringBuilder builder = new StringBuilder();
        appendCanonical(builder, value);
        String encoded = builder.toString();
        if (encoded.getBytes(StandardCharsets.UTF_8).length > MAX_CANONICAL_BYTES) {
            throw new IllegalArgumentException("runtime/render provenance exceeds byte cap");
        }
        return encoded;
    }

    private static void appendCanonical(StringBuilder output, JsonElement value) {
        if (value == null || value.isJsonNull()) {
            output.append("null");
            return;
        }
        if (value.isJsonPrimitive()) {
            var primitive = value.getAsJsonPrimitive();
            if (primitive.isString()) {
                appendString(output, primitive.getAsString());
            } else if (primitive.isNumber()) {
                String number = primitive.getAsString();
                if (!number.matches("-?(0|[1-9][0-9]*)")) {
                    throw new IllegalArgumentException(
                            "canonical JSON numbers must be integers");
                }
                try {
                    number = Long.toString(Long.parseLong(number));
                } catch (NumberFormatException exception) {
                    throw new IllegalArgumentException(
                            "canonical JSON integer exceeds signed 64-bit range", exception);
                }
                output.append(number);
            } else {
                output.append(primitive.getAsString());
            }
            return;
        }
        if (value.isJsonArray()) {
            output.append('[');
            boolean first = true;
            for (JsonElement item : value.getAsJsonArray()) {
                if (!first) output.append(',');
                appendCanonical(output, item);
                first = false;
            }
            output.append(']');
            return;
        }
        output.append('{');
        boolean first = true;
        List<String> keys = new ArrayList<>(value.getAsJsonObject().keySet());
        keys.sort(RuntimeRenderProvenance::compareCodePoints);
        for (String key : keys) {
            if (!first) output.append(',');
            appendString(output, key);
            output.append(':');
            appendCanonical(output, value.getAsJsonObject().get(key));
            first = false;
        }
        output.append('}');
    }

    private static void measureCanonical(JsonElement value, int depth, long[] total) {
        if (depth > MAX_CANONICAL_DEPTH) {
            throw new IllegalArgumentException("canonical JSON nesting exceeds depth cap");
        }
        if (value == null || value.isJsonNull()) {
            addMeasured(total, 4);
        } else if (value.isJsonPrimitive()) {
            var primitive = value.getAsJsonPrimitive();
            if (primitive.isString()) {
                measureString(primitive.getAsString(), total);
            } else if (primitive.isNumber()) {
                String valueText = primitive.getAsString();
                if (!valueText.matches("-?(0|[1-9][0-9]*)")) {
                    throw new IllegalArgumentException("canonical JSON numbers must be integers");
                }
                try {
                    Long.parseLong(valueText);
                } catch (NumberFormatException exception) {
                    throw new IllegalArgumentException(
                            "canonical JSON integer exceeds signed 64-bit range", exception);
                }
                addMeasured(total, Long.toString(Long.parseLong(valueText)).length());
            } else {
                addMeasured(total, primitive.getAsString().length());
            }
        } else if (value.isJsonArray()) {
            var array = value.getAsJsonArray();
            if (array.size() > MAX_CANONICAL_BYTES) {
                throw new IllegalArgumentException("canonical JSON exceeds byte cap");
            }
            addMeasured(total, 2L + Math.max(0, array.size() - 1));
            for (JsonElement item : array) measureCanonical(item, depth + 1, total);
        } else {
            var object = value.getAsJsonObject();
            if (object.size() > MAX_CANONICAL_OBJECT_KEYS) {
                throw new IllegalArgumentException("canonical JSON object exceeds key cap");
            }
            addMeasured(total, 2L + Math.max(0, object.size() - 1));
            for (var entry : object.entrySet()) {
                measureString(entry.getKey(), total);
                addMeasured(total, 1);
                measureCanonical(entry.getValue(), depth + 1, total);
            }
        }
    }

    private static void measureString(String value, long[] total) {
        addMeasured(total, 2);
        for (int index = 0; index < value.length(); index++) {
            char current = value.charAt(index);
            if (current == '"' || current == '\\' || current == '\b' || current == '\f' ||
                    current == '\n' || current == '\r' || current == '\t') {
                addMeasured(total, 2);
            } else if (current < 0x20 || current == 0x2028 || current == 0x2029) {
                addMeasured(total, 6);
            } else if (Character.isHighSurrogate(current)) {
                if (index + 1 >= value.length() || !Character.isLowSurrogate(value.charAt(index + 1))) {
                    throw new IllegalArgumentException("unpaired high surrogate");
                }
                index++;
                addMeasured(total, 12);
            } else if (Character.isLowSurrogate(current)) {
                throw new IllegalArgumentException("unpaired low surrogate");
            } else if (current < 0x80) {
                addMeasured(total, 1);
            } else if (current < 0x800) {
                addMeasured(total, 2);
            } else {
                addMeasured(total, 3);
            }
        }
    }

    private static void addMeasured(long[] total, long amount) {
        total[0] += amount;
        if (total[0] > MAX_CANONICAL_BYTES) {
            throw new IllegalArgumentException("runtime/render provenance exceeds byte cap");
        }
    }

    private static int compareCodePoints(String left, String right) {
        int leftIndex = 0;
        int rightIndex = 0;
        while (leftIndex < left.length() && rightIndex < right.length()) {
            int leftPoint = left.codePointAt(leftIndex);
            int rightPoint = right.codePointAt(rightIndex);
            int compared = Integer.compare(leftPoint, rightPoint);
            if (compared != 0) return compared;
            leftIndex += Character.charCount(leftPoint);
            rightIndex += Character.charCount(rightPoint);
        }
        if (leftIndex == left.length() && rightIndex == right.length()) return 0;
        return leftIndex == left.length() ? -1 : 1;
    }

    private static void appendString(StringBuilder output, String value) {
        output.append('"');
        for (int index = 0; index < value.length(); index++) {
            char current = value.charAt(index);
            switch (current) {
                case '"' -> output.append("\\\"");
                case '\\' -> output.append("\\\\");
                case '\b' -> output.append("\\b");
                case '\f' -> output.append("\\f");
                case '\n' -> output.append("\\n");
                case '\r' -> output.append("\\r");
                case '\t' -> output.append("\\t");
                default -> {
                    if (current < 0x20 || current == 0x2028 || current == 0x2029) {
                        appendUnicodeEscape(output, current);
                    } else if (Character.isHighSurrogate(current)) {
                        if (index + 1 >= value.length() ||
                                !Character.isLowSurrogate(value.charAt(index + 1))) {
                            throw new IllegalArgumentException("unpaired high surrogate");
                        }
                        char low = value.charAt(++index);
                        appendUnicodeEscape(output, current);
                        appendUnicodeEscape(output, low);
                    } else if (Character.isLowSurrogate(current)) {
                        throw new IllegalArgumentException("unpaired low surrogate");
                    } else {
                        output.append(current);
                    }
                }
            }
        }
        output.append('"');
    }

    private static void appendUnicodeEscape(StringBuilder output, char value) {
        output.append("\\u");
        String hex = Integer.toHexString(value);
        output.append("0".repeat(4 - hex.length())).append(hex);
    }
}
