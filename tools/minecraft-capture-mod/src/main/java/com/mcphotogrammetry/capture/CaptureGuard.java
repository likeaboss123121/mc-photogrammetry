package com.mcphotogrammetry.capture;

/**
 * Minecraft-free snapshot of the client state immediately before a screenshot.
 *
 * <p>The DS-20 smoke run produced four "captured" frames while frames 01-03
 * actually showed the ordinary in-game pause menu: the log recorded
 * {@code Saving and pausing game...} right after frame 01 applied its client
 * settings, and frame 00 was captured with a yaw error of 1.35 degrees that the
 * single pre-settle convergence check never re-measured. Both failures are
 * invisible to a pose check taken minutes-earlier in the sequence, so the capture
 * path now takes one more explicit measurement right before the screenshot.
 *
 * <p>This record is the measurement. The controller builds it from the live client
 * ({@code currentScreen}, {@code isPaused()}, {@code isWindowFocused()} and the
 * same feet/eye/yaw/pitch comparison the earlier stages use) and {@link
 * CaptureDriver} decides what may happen next from the flags alone, which keeps the
 * decision unit-testable on a plain JVM.
 *
 * <p>Capture is allowed only when there is no screen, the client is not paused, and
 * the pose is re-converged. Window focus is recorded as read-only telemetry but is
 * <em>never</em> an acceptance requirement: interactive capture runs on a hidden,
 * unfocused Windows Desktop 2 where {@code focused=false} is the normal state, and
 * DS-21 stages {@code pauseOnLostFocus:false} so losing focus does not pause the
 * client. An unexpected screen is never closed and never captured behind; only
 * {@link ScreenKind#GAME_MENU} - the ordinary in-game pause menu - is eligible for
 * automatic recovery.
 */
public record CaptureGuard(ScreenKind screen, String screenType, boolean paused,
                           boolean focused, boolean poseConverged, String poseDetail) {

    /** What kind of screen, if any, was open when the guard was evaluated. */
    public enum ScreenKind {
        /** No screen is open: the ordinary in-game world view is on screen. */
        NONE,
        /** The ordinary in-game pause menu ({@code GameMenuScreen}); recoverable. */
        GAME_MENU,
        /** Anything else; never closed, never captured behind. */
        UNEXPECTED
    }

    public CaptureGuard {
        if (screen == null) {
            screen = ScreenKind.NONE;
        }
        if (poseDetail == null) {
            poseDetail = "";
        }
    }

    public static CaptureGuard noScreen(boolean paused, boolean focused,
                                        boolean poseConverged, String poseDetail) {
        return new CaptureGuard(ScreenKind.NONE, null, paused, focused, poseConverged,
                poseDetail);
    }

    public static CaptureGuard ordinaryGameMenu(boolean paused, boolean focused,
                                                boolean poseConverged, String poseDetail) {
        return new CaptureGuard(ScreenKind.GAME_MENU,
                "net.minecraft.client.gui.screen.GameMenuScreen", paused, focused,
                poseConverged, poseDetail);
    }

    public static CaptureGuard unexpectedScreen(String screenType, boolean paused,
                                                boolean focused, boolean poseConverged,
                                                String poseDetail) {
        return new CaptureGuard(ScreenKind.UNEXPECTED, screenType, paused, focused,
                poseConverged, poseDetail);
    }

    public boolean screenOpen() {
        return screen != ScreenKind.NONE;
    }

    /**
     * True only for a state in which a screenshot is allowed to be taken.
     *
     * <p>Focus is deliberately absent from this rule. It is recorded for evidence
     * only: the hidden capture desktop is normally unfocused, so a clean, unpaused,
     * pose-converged frame is valid and capturable whether or not the window has
     * focus. Requiring focus here would fail every real capture and would have the
     * controller steal focus it does not need.
     */
    public boolean ready() {
        return screen == ScreenKind.NONE && !paused && poseConverged;
    }

    /** One-line description used in failure and evidence text. */
    public String describe() {
        return "screen=" + (screenType == null ? "none" : screenType)
                + ", paused=" + paused
                + ", focused=" + focused
                + ", poseConverged=" + poseConverged
                + (poseDetail.isEmpty() ? "" : ", pose=[" + poseDetail + "]");
    }
}
