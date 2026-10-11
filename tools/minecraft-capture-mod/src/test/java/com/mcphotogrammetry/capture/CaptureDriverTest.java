package com.mcphotogrammetry.capture;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The third, fourth, and DS-20 launch-rejection driver tests.
 *
 * <p>It is deliberately Minecraft-free and asserts the {@link CaptureDriver.Effect}s
 * the driver emits and the stages it actually entered, not just an enum successor.
 * The defects it pins are the ones the reviews and the real smoke launch found:
 *
 * <ul>
 *   <li>the controller called convergence while the stage machine was still at
 *       {@code WAIT_POSITION}, so a valid capture threw;</li>
 *   <li>a completed SETTINGS advanced the helper to SETTLE without scheduling
 *       capture, so a job could walk to DONE with no screenshot;</li>
 *   <li>the controller stopped setting FOV, gamma, HUD, and window size at all;</li>
 *   <li>the client-settings effect must run once, after convergence and before the
 *       server settings task and the settle window;</li>
 *   <li>DS-20: the pose was measured only once, before the resize and settle, so a
 *       first-frame yaw drift reached the screenshot;</li>
 *   <li>DS-20: a pause menu that opened during settle was captured instead of
 *       closed, and an unexpected screen was never distinguished from it;</li>
 *   <li>DS-20: capture must not be invoked until the live guard - no screen, not
 *       paused, pose re-converged - has passed, with bounded retries;</li>
 *   <li>DS-20C: focus is telemetry only and is not part of readiness, so an
 *       unfocused but clean, unpaused, converged frame is captured with no
 *       focus-stealing effect and no retry; only real contamination (screen, pause,
 *       pose drift) may trigger recovery.</li>
 * </ul>
 */
class CaptureDriverTest {

    /** A guard that permits capture: no screen, unpaused, pose converged. */
    private static CaptureGuard ready() {
        // Unfocused on purpose: that is the normal state on the hidden capture
        // desktop, and readiness must not depend on focus (DS-20C).
        return CaptureGuard.noScreen(false, false, true, "feet off by 0.001");
    }

    /**
     * The stage loop the controller runs, with every asynchronous wait applied in
     * order. The returned log is exactly the {@code (step, effect)} sequence the
     * controller must have observed.
     */
    private static final class Harness {
        private final CaptureDriver driver;
        private final List<String> log = new ArrayList<>();
        private int positionSchedules;

        Harness() {
            this(3, CaptureDriver.DEFAULT_MAX_GUARD_ATTEMPTS);
        }

        Harness(int maxPositionAttempts, int maxGuardAttempts) {
            driver = new CaptureDriver(maxPositionAttempts, maxGuardAttempts);
            record("start", driver.start());
        }

        /** Apply the effects one step produced, exactly as the controller would. */
        private void record(String source, List<CaptureDriver.Effect> effects) {
            for (CaptureDriver.Effect effect : effects) {
                log.add(source + ":" + effect);
                if (effect == CaptureDriver.Effect.SCHEDULE_POSITION) {
                    positionSchedules++;
                }
            }
        }

        void position(boolean converged) {
            record("position", driver.onPositionApplied(converged, "feet off by 0.900", "f"));
        }

        void clientSettings() {
            record("client-settings", driver.onClientSettingsApplied());
        }

        void settings() {
            record("settings", driver.onSettingsApplied());
        }

        void settleOutcome() {
            record("settle", driver.onSettleElapsed());
        }

        void guard(CaptureGuard guard) {
            record("guard", driver.onGuardEvaluated(guard));
        }

        void captureInvoked() {
            record("capture", driver.onCaptureCompleted());
        }

        void screenshot() {
            record("file", driver.onScreenshotReady());
        }

        void fail() {
            record("fail", driver.fail());
        }

        List<String> log() {
            return List.copyOf(log);
        }

        List<CaptureDriver.Stage> stages() {
            return driver.stages();
        }

        int positionSchedules() {
            return positionSchedules;
        }

        long countOf(String suffix) {
            return log.stream().filter(entry -> entry.endsWith(suffix)).count();
        }

        /** Drive a clean job up to and including the settle window. */
        void toSettle() {
            position(true);
            clientSettings();
            position(true);
            settings();
        }

        /** Drive a clean job up to the capture-guard evaluation. */
        void toGuard() {
            toSettle();
            settleOutcome();
        }
    }

    private static long countStage(List<CaptureDriver.Stage> stages,
                                   CaptureDriver.Stage stage) {
        return stages.stream().filter(candidate -> candidate == stage).count();
    }

    @Test
    void emitsEveryRequiredEffectExactlyOnceInTheRequiredOrder() {
        Harness harness = new Harness();
        harness.position(true);
        harness.clientSettings();
        harness.position(true);
        harness.settings();
        harness.settleOutcome();
        harness.guard(ready());
        harness.captureInvoked();
        harness.screenshot();

        assertEquals(List.of(
                "start:SCHEDULE_POSITION",
                "position:APPLY_CLIENT_SETTINGS",
                "client-settings:SCHEDULE_POSITION",
                "position:SCHEDULE_SETTINGS",
                "settings:SETTLE",
                "settle:EVALUATE_GUARD",
                "guard:INVOKE_CAPTURE",
                "capture:AWAIT_FILE",
                "file:DONE"), harness.log());

        // The stages the job actually entered, in order: a missing, reordered, or
        // duplicated step is visible here as data, not only as an enum value.
        assertEquals(List.of(
                CaptureDriver.Stage.POSITION_SCHEDULED,
                CaptureDriver.Stage.CONVERGE,
                CaptureDriver.Stage.CLIENT_SETTINGS,
                CaptureDriver.Stage.REASSERT_SCHEDULED,
                CaptureDriver.Stage.REASSERT_CONVERGE,
                CaptureDriver.Stage.SETTINGS_SCHEDULED,
                CaptureDriver.Stage.SETTLE,
                CaptureDriver.Stage.GUARD,
                CaptureDriver.Stage.CAPTURE_INVOKED,
                CaptureDriver.Stage.AWAIT_FILE,
                CaptureDriver.Stage.DONE), harness.stages());
        assertEquals(harness.stages().size(),
                harness.stages().stream().distinct().count(),
                "no stage may be entered twice");
        assertEquals(2, harness.positionSchedules(),
                "POSITION is scheduled once for the pose and once to re-assert it after "
                        + "the settings/resize");
        assertFalse(harness.stages().contains(CaptureDriver.Stage.FAILED));
    }

    /**
     * The fourth launch rejection: the controller recorded and restored client
     * settings but never applied them. This pins where they run - once, after
     * convergence, before the post-settings pose re-assertion, the server settings
     * task, and the settle window.
     */
    @Test
    void clientSettingsAreAppliedExactlyOnceAfterConvergenceAndBeforeServerSettings() {
        Harness harness = new Harness();
        harness.position(true);
        harness.clientSettings();
        harness.position(true);
        harness.settings();
        harness.settleOutcome();
        harness.guard(ready());

        List<String> log = harness.log();
        assertEquals(1, harness.countOf("APPLY_CLIENT_SETTINGS"),
                "the client-settings effect must be emitted exactly once");
        assertTrue(log.indexOf("position:APPLY_CLIENT_SETTINGS")
                        < log.indexOf("client-settings:SCHEDULE_POSITION"),
                "the settings (including the resize) must be applied before the pose is "
                        + "re-asserted");
        assertTrue(log.indexOf("client-settings:SCHEDULE_POSITION")
                        < log.indexOf("position:SCHEDULE_SETTINGS"),
                "the re-assertion must precede the server settings task");
        assertTrue(log.indexOf("position:SCHEDULE_SETTINGS") < log.indexOf("settings:SETTLE"),
                "client settings must be applied before the settle window");
        assertTrue(log.indexOf("settings:SETTLE") < log.indexOf("guard:INVOKE_CAPTURE"),
                "the settle window and guard must still precede capture");

        List<CaptureDriver.Stage> stages = harness.stages();
        assertTrue(stages.indexOf(CaptureDriver.Stage.CONVERGE)
                < stages.indexOf(CaptureDriver.Stage.CLIENT_SETTINGS));
        assertTrue(stages.indexOf(CaptureDriver.Stage.CLIENT_SETTINGS)
                < stages.indexOf(CaptureDriver.Stage.REASSERT_SCHEDULED));
        assertTrue(stages.indexOf(CaptureDriver.Stage.REASSERT_CONVERGE)
                < stages.indexOf(CaptureDriver.Stage.SETTINGS_SCHEDULED));
        assertTrue(stages.indexOf(CaptureDriver.Stage.SETTINGS_SCHEDULED)
                < stages.indexOf(CaptureDriver.Stage.SETTLE));

        assertThrows(IllegalStateException.class, harness::clientSettings,
                "a job must not apply its client settings twice");
    }

    @Test
    void aDivergenceRetriesPositionAndNeverSchedulesSettingsOrCapture() {
        Harness harness = new Harness();
        harness.position(false);

        assertEquals(List.of("start:SCHEDULE_POSITION", "position:SCHEDULE_POSITION"),
                harness.log(),
                "a divergence may only re-schedule POSITION");
        assertEquals(List.of(
                CaptureDriver.Stage.POSITION_SCHEDULED,
                CaptureDriver.Stage.POSITION_SCHEDULED), harness.stages());
        assertFalse(harness.stages().contains(CaptureDriver.Stage.CLIENT_SETTINGS));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.SETTINGS_SCHEDULED));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.SETTLE));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.CAPTURE_INVOKED));

        // The retry can still converge, and then the normal order resumes.
        harness.position(true);
        harness.clientSettings();
        harness.position(true);
        harness.settings();
        harness.settleOutcome();
        assertEquals(List.of(
                "start:SCHEDULE_POSITION",
                "position:SCHEDULE_POSITION",
                "position:APPLY_CLIENT_SETTINGS",
                "client-settings:SCHEDULE_POSITION",
                "position:SCHEDULE_SETTINGS",
                "settings:SETTLE",
                "settle:EVALUATE_GUARD"), harness.log());
        assertFalse(harness.log().contains("settle:INVOKE_CAPTURE"),
                "the guard, not the settle window, decides capture");
        assertEquals(3, harness.positionSchedules());
    }

    @Test
    void aTerminalConvergenceFailureFailsTheJobWithoutEverCapturing() {
        Harness harness = new Harness();
        harness.position(false); // attempt 2
        harness.position(false); // attempt 3
        harness.position(false); // attempt 3 is the last: terminal

        assertEquals("position:FAIL_JOB", harness.log().get(harness.log().size() - 1));
        assertTrue(harness.stages().contains(CaptureDriver.Stage.FAILED));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.CLIENT_SETTINGS));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.SETTINGS_SCHEDULED));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.SETTLE));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.CAPTURE_INVOKED));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.AWAIT_FILE));
        assertEquals(3, harness.positionSchedules(),
                "a terminal failure stops at the bounded attempt count");
        assertEquals(1, countStage(harness.stages(), CaptureDriver.Stage.FAILED),
                "the failure is recorded exactly once");
        assertTrue(harness.driver.failureDetail().contains("did not converge"));
    }

    /**
     * DS-20, first-frame drift. The smoke run captured frame 00 with a yaw error of
     * 1.35 degrees: the pose was measured before the resize and never again. Here the
     * pre-settle convergence check passes, then the live guard measures a drift, and
     * no screenshot may be invoked until the pose has re-converged and a fresh settle
     * window has elapsed.
     */
    @Test
    void aFirstFrameDriftIsCaughtByTheGuardAndNeedsAReconvergenceBeforeCapture() {
        Harness harness = new Harness();
        harness.toGuard();

        harness.guard(CaptureGuard.noScreen(false, true, false, "yaw delta 1.350"));
        assertEquals("guard:SCHEDULE_POSITION",
                harness.log().get(harness.log().size() - 1),
                "a drifted pose may only re-schedule POSITION");
        assertEquals(0, harness.countOf("INVOKE_CAPTURE"),
                "no screenshot may be invoked before the pose re-converges");
        assertEquals(1, countStage(harness.stages(), CaptureDriver.Stage.GUARD),
                "the drift is caught once, at the guard");

        // The re-assertion converges; because the guard demanded recovery, the job
        // must not repeat the server settings task, but it must take a fresh settle.
        harness.position(true);
        assertEquals("position:SETTLE",
                harness.log().get(harness.log().size() - 1),
                "a recovery re-enters a fresh settle window");
        assertEquals(1, harness.countOf("SCHEDULE_SETTINGS"),
                "the world settings already describe the scene; only the pose was re-asserted");
        assertEquals(2, countStage(harness.stages(), CaptureDriver.Stage.SETTLE),
                "the contaminated settle window is replaced by a fresh one");
        assertEquals(1, harness.driver.recoveries());

        harness.settleOutcome();
        harness.guard(ready());
        assertEquals("guard:INVOKE_CAPTURE",
                harness.log().get(harness.log().size() - 1));
        assertEquals(1, harness.countOf("INVOKE_CAPTURE"),
                "capture is invoked exactly once, after the guard passes");
    }

    /**
     * DS-20, a pause during settle. The log recorded "Saving and pausing game..."
     * right after frame 01 applied its settings, and frames 01-03 showed the pause
     * menu. The guard must close only that ordinary menu, then require a fresh settle
     * and pose re-assertion before it will allow the screenshot.
     */
    @Test
    void aPauseDuringSettleClosesOnlyTheOrdinaryMenuAndRequiresAFreshSettle() {
        Harness harness = new Harness();
        harness.toGuard();

        harness.guard(CaptureGuard.ordinaryGameMenu(true, true, true, "within tolerance"));
        List<String> log = harness.log();
        assertEquals(List.of("guard:CLOSE_MENU", "guard:SCHEDULE_POSITION"),
                log.subList(log.size() - 2, log.size()),
                "the ordinary menu is closed and the pose re-asserted");
        assertEquals(0, harness.countOf("INVOKE_CAPTURE"),
                "nothing may be captured while the client is paused behind a screen");

        harness.position(true);
        assertEquals(1, harness.countOf("SCHEDULE_SETTINGS"),
                "a recovered job does not re-apply the world settings");
        harness.settleOutcome();
        harness.guard(ready());

        assertEquals(2, countStage(harness.stages(), CaptureDriver.Stage.SETTLE),
                "the settle that ran while the menu was open is void");
        assertEquals(2, countStage(harness.stages(), CaptureDriver.Stage.GUARD));
        assertEquals(1, harness.countOf("INVOKE_CAPTURE"));
        assertEquals(1, harness.countOf("SCHEDULE_SETTINGS"),
                "the server settings task runs exactly once per job");
    }

    /** DS-20: only the ordinary pause menu is recoverable; anything else is refused. */
    @Test
    void anUnexpectedScreenIsRefusedAndNeverCapturedBehind() {
        Harness harness = new Harness();
        harness.toGuard();

        harness.guard(CaptureGuard.unexpectedScreen(
                "net.minecraft.client.gui.screen.DeathScreen", false, true, true, "ok"));

        assertEquals("guard:FAIL_JOB", harness.log().get(harness.log().size() - 1));
        assertTrue(harness.stages().contains(CaptureDriver.Stage.FAILED));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.CAPTURE_INVOKED));
        assertEquals(0, harness.countOf("INVOKE_CAPTURE"));
        assertEquals(0, harness.countOf("CLOSE_MENU"),
                "an unexpected screen is never closed");
        assertTrue(harness.driver.failureDetail().contains("DeathScreen"),
                "the failure names the screen it refused");
        assertThrows(IllegalStateException.class, harness::screenshot);
    }

    /**
     * DS-20C: the binding background constraint. Interactive capture runs on a
     * hidden, unfocused Windows Desktop 2, so {@code focused=false} is the normal
     * state. Focus is telemetry only: an unfocused but clean, unpaused,
     * pose-converged guard must capture on the first evaluation, with no
     * focus-stealing effect and no recovery cycle.
     */
    @Test
    void anUnfocusedCleanUnpausedConvergedFrameIsCapturedWithoutFocusRetry() {
        Harness harness = new Harness();
        harness.toGuard();

        harness.guard(CaptureGuard.noScreen(false, false, true, "within tolerance"));

        assertEquals("guard:INVOKE_CAPTURE",
                harness.log().get(harness.log().size() - 1),
                "an unfocused but clean, unpaused, converged frame is captured");
        assertEquals(1, harness.countOf("INVOKE_CAPTURE"));
        assertEquals(0, harness.countOf("FOCUS_WINDOW"),
                "no focus-stealing effect may be emitted for a merely unfocused window");
        assertEquals(0, harness.driver.recoveries(),
                "an unfocused window is not contamination and must not open a recovery");
        assertEquals(1, harness.driver.guardAttempts(),
                "the clean unfocused guard passes on its first evaluation");
        assertEquals(2, harness.positionSchedules(),
                "only the pose and its post-settings re-assertion are scheduled; focus "
                        + "adds no extra teleport");
        assertFalse(harness.stages().contains(CaptureDriver.Stage.FAILED));
    }

    /**
     * DS-20: a guard that never becomes ready fails after a bounded number of tries.
     * DS-20C: the not-ready state used here is a real one (no screen, unpaused, pose
     * drifted) so the recovery is a pose re-assertion, not a focus attempt.
     */
    @Test
    void aGuardThatNeverBecomesReadyFailsAfterTheBoundedAttempts() {
        Harness harness = new Harness(3, 2);
        harness.toGuard();

        harness.guard(CaptureGuard.noScreen(false, false, false, "yaw delta 1.350")); // drifted
        List<String> log = harness.log();
        assertEquals(List.of("guard:SCHEDULE_POSITION"),
                log.subList(log.size() - 1, log.size()),
                "a drifted pose re-asserts the pose; an unfocused window adds nothing");
        assertEquals(0, harness.countOf("INVOKE_CAPTURE"));

        harness.position(true); // fresh settle
        harness.settleOutcome();
        harness.guard(CaptureGuard.noScreen(false, false, false, "yaw delta 1.350"));
        // attempt 2 == max

        assertEquals("guard:FAIL_JOB", harness.log().get(harness.log().size() - 1));
        assertTrue(harness.driver.failureDetail().contains("2 evaluations"));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.CAPTURE_INVOKED));
        assertEquals(0, harness.countOf("INVOKE_CAPTURE"));
        assertEquals(2, harness.driver.guardAttempts());
    }

    /** DS-20: paused with no screen to close cannot be recovered, so it fails at once. */
    @Test
    void aPausedClientWithNoRecoverableScreenIsRefusedImmediately() {
        Harness harness = new Harness();
        harness.toGuard();

        harness.guard(CaptureGuard.noScreen(true, true, true, "within tolerance"));

        assertEquals("guard:FAIL_JOB", harness.log().get(harness.log().size() - 1));
        assertTrue(harness.driver.failureDetail().contains("paused"));
        assertEquals(0, harness.countOf("INVOKE_CAPTURE"));
    }

    /**
     * DS-20: a pose that drifts after the client settings must be re-converged, and a
     * job that never re-converges must never reach the settle window or capture.
     */
    @Test
    void aPostSettingsDivergenceRetriesAndNeverSettlesOrCaptures() {
        Harness harness = new Harness(2, 3);
        harness.position(true);
        harness.clientSettings();      // schedules the re-assertion
        harness.position(false);       // re-assertion 1 fails: retry
        harness.position(false);       // re-assertion 2 fails: retry
        harness.position(false);       // re-assertions exhausted: terminal

        assertEquals("position:FAIL_JOB", harness.log().get(harness.log().size() - 1));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.SETTINGS_SCHEDULED));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.SETTLE));
        assertFalse(harness.stages().contains(CaptureDriver.Stage.CAPTURE_INVOKED));
        assertEquals(0, harness.countOf("INVOKE_CAPTURE"));
        assertTrue(harness.driver.failureDetail().contains("re-converge"));
    }

    @Test
    void aTerminalFailureAfterCaptureWasInvokedCannotBeFollowedByCapture() {
        Harness harness = new Harness();
        harness.toGuard();
        harness.guard(ready());
        harness.captureInvoked();      // ...and the file is awaited
        harness.fail();                // ...but the screenshot never completes

        assertEquals("fail:FAIL_JOB", harness.log().get(harness.log().size() - 1));
        assertThrows(IllegalStateException.class, harness::screenshot,
                "a failed job must not be able to report a screenshot");
        assertThrows(IllegalStateException.class, harness::settleOutcome,
                "a failed job must not be able to invoke capture again");
    }

    @Test
    void stagesCannotBeReachedOutOfOrderOrTwice() {
        // Client settings, re-assertion outcomes, server settings, settle, guard and
        // capture cannot be reported before a position was applied.
        Harness early = new Harness();
        assertThrows(IllegalStateException.class, early::clientSettings);
        assertThrows(IllegalStateException.class, early::settings);
        assertThrows(IllegalStateException.class, early::settleOutcome);
        assertThrows(IllegalStateException.class, () -> early.guard(ready()));
        assertThrows(IllegalStateException.class, early::captureInvoked);
        assertThrows(IllegalStateException.class, early::screenshot);

        // The re-assertion cannot be skipped, the server settings task cannot run
        // before the re-assertion converged, and neither can the settle window.
        Harness unsettled = new Harness();
        unsettled.position(true);
        assertThrows(IllegalStateException.class, unsettled::settings);
        assertThrows(IllegalStateException.class, unsettled::settleOutcome);
        assertThrows(IllegalStateException.class, () -> unsettled.guard(ready()));
        unsettled.clientSettings();
        assertThrows(IllegalStateException.class, unsettled::settings);
        assertThrows(IllegalStateException.class, unsettled::settleOutcome);
        assertThrows(IllegalStateException.class, () -> unsettled.guard(ready()));
        unsettled.position(true);
        unsettled.settings();

        // Capture cannot complete before it was invoked, settle cannot expire twice,
        // and the guard cannot be evaluated twice on one settle window.
        assertThrows(IllegalStateException.class, unsettled::captureInvoked);
        unsettled.settleOutcome();
        assertThrows(IllegalStateException.class, unsettled::settleOutcome);
        unsettled.guard(ready());
        assertThrows(IllegalStateException.class, () -> unsettled.guard(ready()));
        unsettled.captureInvoked();
        unsettled.screenshot();
        assertThrows(IllegalStateException.class, unsettled::screenshot);
    }

    @Test
    void onlyOnePositionAttemptIsAllowedWhenTheDriverIsConfiguredWithOne() {
        CaptureDriver driver = new CaptureDriver(1);
        assertEquals(List.of(CaptureDriver.Effect.SCHEDULE_POSITION), driver.start());
        assertEquals(List.of(CaptureDriver.Effect.FAIL_JOB),
                driver.onPositionApplied(false, "off by 1.0", "f"));
        assertTrue(driver.failed());
        assertThrows(IllegalArgumentException.class, () -> new CaptureDriver(0));
        assertThrows(IllegalArgumentException.class, () -> new CaptureDriver(1, 0));
    }
}
