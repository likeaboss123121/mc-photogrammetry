package com.mcphotogrammetry.capture;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PoseStabilizerTest {

    private static final class FakePlayer implements PoseStabilizer.Target {
        private double x = 191.109;
        private double y = 85.558;
        private double z = 368.620;
        private float yaw = 137.0F;
        private float pitch = -16.0F;
        private double velocityX = 0.125;
        private double velocityY = -14.554;
        private double velocityZ = -0.5;
        private float fallDistance = 14.554F;
        private boolean noGravity;
        private boolean invulnerable;

        @Override
        public void setNoGravity(boolean value) {
            noGravity = value;
        }

        @Override
        public void setVelocity(double vx, double vy, double vz) {
            velocityX = vx;
            velocityY = vy;
            velocityZ = vz;
        }

        @Override
        public void setFallDistance(float value) {
            fallDistance = value;
        }

        @Override
        public void setInvulnerable(boolean value) {
            invulnerable = value;
        }
    }

    @Test
    void repeatedHoldCancelsMotionAndDamageWithoutChangingTheRequestedPose() {
        FakePlayer player = new FakePlayer();
        double[] pose = {player.x, player.y, player.z, player.yaw, player.pitch};

        PoseStabilizer.hold(player);
        PoseStabilizer.hold(player);

        assertTrue(player.noGravity);
        assertTrue(player.invulnerable);
        assertEquals(0.0, player.velocityX, 0.0);
        assertEquals(0.0, player.velocityY, 0.0);
        assertEquals(0.0, player.velocityZ, 0.0);
        assertEquals(0.0F, player.fallDistance, 0.0F);
        assertEquals(pose[0], player.x, 0.0);
        assertEquals(pose[1], player.y, 0.0);
        assertEquals(pose[2], player.z, 0.0);
        assertEquals((float) pose[3], player.yaw, 0.0F);
        assertEquals((float) pose[4], player.pitch, 0.0F);
    }

    @Test
    void sessionOnlyPinsTheValidatedPlayerAndStopsAtRunTermination() {
        PoseStabilizer.Session session = new PoseStabilizer.Session();
        UUID stagedPlayer = UUID.fromString("47a6a81a-fd99-4a7d-914a-b46ad4dd3151");
        UUID unrelatedPlayer = UUID.fromString("8b215810-0244-4637-a405-d41984c959f3");

        assertFalse(session.matches(stagedPlayer));
        assertThrows(IllegalArgumentException.class, () -> session.start(null));
        session.start(stagedPlayer);
        assertTrue(session.matches(stagedPlayer));
        assertFalse(session.matches(unrelatedPlayer));
        session.stop();
        assertFalse(session.matches(stagedPlayer));
    }

    @Test
    void clientAndIntegratedServerTickHooksAreEnabledByTheClientMixinConfig() {
        try (InputStream resource = getClass().getResourceAsStream(
                "/mc-photo-capture.mixins.json")) {
            assertNotNull(resource, "the packaged mixin configuration must be present");
            JsonObject config = JsonParser.parseReader(new InputStreamReader(resource,
                    StandardCharsets.UTF_8)).getAsJsonObject();
            var clientMixins = config.getAsJsonArray("client");
            assertTrue(clientMixins.asList().stream()
                    .anyMatch(name -> name.getAsString().equals("MinecraftClientTickMixin")));
            assertTrue(clientMixins.asList().stream()
                    .anyMatch(name -> name.getAsString().equals("ServerPlayerTickMixin")),
                    "the integrated-server player tick hook must be packaged for singleplayer");
        } catch (java.io.IOException error) {
            throw new AssertionError("could not read the packaged mixin configuration", error);
        }
    }
}
