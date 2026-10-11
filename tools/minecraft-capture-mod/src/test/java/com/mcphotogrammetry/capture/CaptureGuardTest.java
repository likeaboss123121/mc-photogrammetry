package com.mcphotogrammetry.capture;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pure rules for the DS-20 capture guard.
 *
 * <p>The controller supplies the live measurement; these tests pin the only rule
 * that decides whether a screenshot may be taken and which screens are recoverable.
 * Window focus is telemetry only (DS-20C): capture runs on a hidden, unfocused
 * Windows Desktop 2, so it must never appear in the readiness rule.
 */
class CaptureGuardTest {

    @Test
    void captureIsAllowedWithNoScreenUnpausedAndAConvergedPoseRegardlessOfFocus() {
        assertTrue(CaptureGuard.noScreen(false, true, true, "ok").ready());
        assertTrue(CaptureGuard.noScreen(false, false, true, "ok").ready(),
                "an unfocused window is the normal state on the hidden capture "
                        + "desktop and must not block a clean, unpaused, converged frame");

        assertFalse(CaptureGuard.noScreen(true, true, true, "ok").ready(),
                "a paused client must never be captured");
        assertFalse(CaptureGuard.noScreen(true, false, true, "ok").ready(),
                "a paused client must never be captured even when the window is unfocused");
        assertFalse(CaptureGuard.noScreen(false, true, false, "yaw delta 1.350").ready(),
                "a drifted pose must never be captured");
        assertFalse(CaptureGuard.noScreen(false, false, false, "yaw delta 1.350").ready(),
                "an unfocused, drifted pose must still be refused");
        assertFalse(CaptureGuard.ordinaryGameMenu(true, true, true, "ok").ready(),
                "a capture must never be taken behind a screen");
        assertFalse(CaptureGuard.ordinaryGameMenu(true, false, true, "ok").ready(),
                "a capture must never be taken behind a screen, focused or not");
    }

    @Test
    void onlyTheOrdinaryPauseMenuIsClassifiedAsRecoverable() {
        CaptureGuard menu = CaptureGuard.ordinaryGameMenu(true, false, true, "ok");
        assertEquals(CaptureGuard.ScreenKind.GAME_MENU, menu.screen());
        assertTrue(menu.screenOpen());

        CaptureGuard death = CaptureGuard.unexpectedScreen(
                "net.minecraft.client.gui.screen.DeathScreen", false, true, true, "ok");
        assertEquals(CaptureGuard.ScreenKind.UNEXPECTED, death.screen());
        assertTrue(death.screenOpen());
        assertFalse(death.ready());

        CaptureGuard none = CaptureGuard.noScreen(false, true, true, "ok");
        assertEquals(CaptureGuard.ScreenKind.NONE, none.screen());
        assertFalse(none.screenOpen());
    }

    @Test
    void theDescriptionCarriesEveryFlagForTheEvidenceRecord() {
        String described = CaptureGuard.unexpectedScreen(
                "net.minecraft.client.gui.screen.DeathScreen", true, false, false,
                "feet off by 0.900").describe();
        assertTrue(described.contains("DeathScreen"));
        assertTrue(described.contains("paused=true"));
        assertTrue(described.contains("focused=false"));
        assertTrue(described.contains("poseConverged=false"));
        assertTrue(described.contains("feet off by 0.900"));
    }
}
