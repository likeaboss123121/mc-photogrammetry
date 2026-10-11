package com.mcphotogrammetry.capture;

import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class RuntimeRenderProvenanceTest {
    @Test
    void sharedPythonJavaGoldenBytesAndDigestMatch() throws Exception {
        byte[] golden = Files.readAllBytes(Path.of(
                "src/test/resources/runtime_render_provenance_canonical.golden"));
        String text = new String(golden, StandardCharsets.UTF_8);
        JsonObject value = JsonParser.parseString(text).getAsJsonObject();
        assertEquals(text, RuntimeRenderProvenance.canonical(value));
        assertEquals("63e6f86bf176989980e2f19a9dffb4b08cf5f67a996c7fb0f74d89e992abc2f8",
                RuntimeRenderProvenance.sha256(golden));
    }

    @Test
    void canonicalNumbersMustBeSignedLongIntegers() {
        JsonObject fractional = new JsonObject();
        fractional.addProperty("value", 2.5);
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> RuntimeRenderProvenance.canonical(fractional));

        JsonObject oversized = new JsonObject();
        oversized.add("value", new com.google.gson.JsonPrimitive(
                new java.math.BigInteger("9223372036854775808")));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> RuntimeRenderProvenance.canonical(oversized));

        JsonObject undersized = new JsonObject();
        undersized.add("value", new com.google.gson.JsonPrimitive(
                new java.math.BigInteger("-9223372036854775809")));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> RuntimeRenderProvenance.canonical(undersized));

        JsonObject bounds = new JsonObject();
        bounds.addProperty("max", Long.MAX_VALUE);
        bounds.addProperty("min", Long.MIN_VALUE);
        assertEquals("{\"max\":9223372036854775807,\"min\":-9223372036854775808}",
                RuntimeRenderProvenance.canonical(bounds));
        assertEquals("{\"value\":0}", RuntimeRenderProvenance.canonical(
                JsonParser.parseString("{\"value\":-0}").getAsJsonObject()));
    }

    @Test
    void canonicalOutputCapIsCheckedBeforeSerialization() {
        JsonObject oversized = new JsonObject();
        oversized.addProperty("value", "x".repeat(1_048_577));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> RuntimeRenderProvenance.canonical(oversized));
    }

    @Test
    void canonicalUnicodeKeysUseCodePointOrdering() {
        JsonObject value = new JsonObject();
        value.addProperty("😀", 2);
        value.addProperty("\ue000", 1);
        assertEquals("{\"\ue000\":1,\"\\ud83d\\ude00\":2}",
                RuntimeRenderProvenance.canonical(value));
    }

    @Test
    void canonicalEncodingSortsNestedKeysButPreservesArrayOrderAndUnicode() {
        JsonObject root = new JsonObject();
        root.addProperty("z", 1);
        JsonObject nested = new JsonObject();
        nested.addProperty("b", "snowman \u2603");
        nested.add("a", JsonNull.INSTANCE);
        root.add("a", nested);
        var array = new com.google.gson.JsonArray();
        array.add(2);
        array.add(1);
        root.add("arr", array);

        assertEquals("{\"a\":{\"a\":null,\"b\":\"snowman ☃\"},\"arr\":[2,1],\"z\":1}",
                RuntimeRenderProvenance.canonical(root));
        JsonObject sealOne = RuntimeRenderProvenance.seal(root);
        JsonObject reordered = new JsonObject();
        reordered.add("arr", array.deepCopy());
        reordered.add("a", nested.deepCopy());
        reordered.addProperty("z", 1);
        JsonObject sealTwo = RuntimeRenderProvenance.seal(reordered);
        assertEquals(sealOne.get("seal_sha256"), sealTwo.get("seal_sha256"));
        reordered.getAsJsonArray("arr").set(0, new com.google.gson.JsonPrimitive(1));
        assertNotEquals(sealOne.get("seal_sha256"),
                RuntimeRenderProvenance.seal(reordered).get("seal_sha256"));
    }
}
