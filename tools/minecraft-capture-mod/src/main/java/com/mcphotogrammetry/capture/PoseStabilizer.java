package com.mcphotogrammetry.capture;

import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Physics and damage lock for the one validated staged-world player being used as
 * a capture camera. The target deliberately exposes no position or rotation setter:
 * holding physics must never change the requested feet/eye pose.
 */
public final class PoseStabilizer {

    /** Minimal adapter so the pose-lock contract is testable without Minecraft. */
    public interface Target {
        void setNoGravity(boolean noGravity);

        void setVelocity(double x, double y, double z);

        void setFallDistance(float fallDistance);

        void setInvulnerable(boolean invulnerable);
    }

    /** Cross-thread identity gate. Activated only after staged-world validation. */
    public static final class Session {
        private final AtomicReference<UUID> playerId = new AtomicReference<>();

        public void start(UUID validatedPlayerId) {
            if (validatedPlayerId == null) {
                throw new IllegalArgumentException("a validated staged player UUID is required");
            }
            playerId.set(validatedPlayerId);
        }

        public boolean matches(UUID candidate) {
            return candidate != null && candidate.equals(playerId.get());
        }

        public void stop() {
            playerId.set(null);
        }
    }

    private PoseStabilizer() {
    }

    /** Reapply harmless, zero-motion physics before the next entity tick. */
    public static void hold(Target target) {
        if (target == null) {
            throw new IllegalArgumentException("a player target is required");
        }
        target.setNoGravity(true);
        target.setVelocity(0.0, 0.0, 0.0);
        target.setFallDistance(0.0F);
        target.setInvulnerable(true);
    }

}
