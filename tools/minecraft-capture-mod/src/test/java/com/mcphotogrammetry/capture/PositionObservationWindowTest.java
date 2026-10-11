package com.mcphotogrammetry.capture;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PositionObservationWindowTest {

    @Test
    void delayedClientConvergenceSucceedsWithTheOriginalTeleport() {
        PositionObservationWindow window = new PositionObservationWindow(5);
        CaptureDriver driver = new CaptureDriver(3);

        assertEquals(List.of(CaptureDriver.Effect.SCHEDULE_POSITION), driver.start());
        assertEquals(1, driver.positionAttempts());
        assertEquals(PositionObservationWindow.TaskResult.OBSERVING,
                window.serverTaskCompleted(null));

        assertEquals(PositionObservationWindow.Observation.WAITING, window.observe(false));
        assertEquals(PositionObservationWindow.Observation.WAITING, window.observe(false));
        assertEquals(1, driver.positionAttempts(),
                "transient client/server disagreement must not spend a teleport attempt");
        assertEquals(PositionObservationWindow.Observation.CONVERGED, window.observe(true));
        assertEquals(List.of(CaptureDriver.Effect.APPLY_CLIENT_SETTINGS),
                driver.onPositionApplied(true, "converged on third client tick", "frame-0"));
        assertEquals(1, driver.positionAttempts());
        assertFalse(window.isObserving());
    }

    @Test
    void retryBudgetAdvancesOnlyAfterEachFullObservationTimeout() {
        int timeoutTicks = PositionObservationWindow.DEFAULT_TIMEOUT_TICKS;
        PositionObservationWindow window = new PositionObservationWindow(timeoutTicks);
        CaptureDriver driver = new CaptureDriver(3);
        int schedules = 0;
        assertEquals(List.of(CaptureDriver.Effect.SCHEDULE_POSITION), driver.start());
        schedules++;

        for (int attempt = 1; attempt <= 3; attempt++) {
            assertEquals(PositionObservationWindow.TaskResult.OBSERVING,
                    window.serverTaskCompleted(null));
            for (int tick = 1; tick < timeoutTicks; tick++) {
                assertEquals(PositionObservationWindow.Observation.WAITING,
                        window.observe(false));
                assertEquals(attempt, driver.positionAttempts());
                assertEquals(attempt, schedules,
                        "no replacement teleport is scheduled before the timeout");
            }
            assertEquals(PositionObservationWindow.Observation.TIMED_OUT,
                    window.observe(false));
            List<CaptureDriver.Effect> effects = driver.onPositionApplied(
                    false, "pose did not arrive", "frame-0");
            if (attempt < 3) {
                assertEquals(List.of(CaptureDriver.Effect.SCHEDULE_POSITION), effects);
                schedules++;
                assertEquals(attempt + 1, driver.positionAttempts());
            } else {
                assertEquals(List.of(CaptureDriver.Effect.FAIL_JOB), effects);
                assertTrue(driver.failed());
            }
        }
        assertEquals(3, schedules);
        assertEquals(3, driver.positionAttempts());
        assertFalse(driver.stages().contains(CaptureDriver.Stage.SETTLE));
        assertFalse(driver.stages().contains(CaptureDriver.Stage.CAPTURE_INVOKED));
    }

    @Test
    void serverTaskErrorFailsImmediatelyWithoutStartingObservation() {
        PositionObservationWindow window = new PositionObservationWindow(5);
        CaptureDriver driver = new CaptureDriver(3);
        driver.start();

        assertEquals(PositionObservationWindow.TaskResult.FAILED,
                window.serverTaskCompleted("no server player matched UUID"));
        assertFalse(window.isObserving());
        assertEquals(0, window.observedTicks());
        assertEquals(List.of(CaptureDriver.Effect.FAIL_JOB), driver.fail());
        assertEquals(1, driver.positionAttempts());
        assertThrows(IllegalStateException.class, () -> window.observe(false));
    }

    @Test
    void reassertAndLaterFrameKeepTheNormalSingleTickConvergencePath() {
        PositionObservationWindow window = new PositionObservationWindow(5);
        CaptureDriver first = new CaptureDriver(3);
        first.start();
        window.serverTaskCompleted(null);
        assertEquals(PositionObservationWindow.Observation.CONVERGED, window.observe(true));
        assertEquals(List.of(CaptureDriver.Effect.APPLY_CLIENT_SETTINGS),
                first.onPositionApplied(true, "frame 0", "frame-0"));

        assertEquals(List.of(CaptureDriver.Effect.SCHEDULE_POSITION),
                first.onClientSettingsApplied());
        window.serverTaskCompleted(null);
        assertEquals(PositionObservationWindow.Observation.CONVERGED, window.observe(true));
        assertEquals(List.of(CaptureDriver.Effect.SCHEDULE_SETTINGS),
                first.onPositionApplied(true, "reasserted", "frame-0"));
        assertEquals(1, first.positionAttempts());
        assertEquals(0, first.reassertAttempts());
        assertEquals(List.of(CaptureDriver.Effect.SETTLE), first.onSettingsApplied());
        assertEquals(List.of(CaptureDriver.Effect.EVALUATE_GUARD), first.onSettleElapsed());
        assertEquals(List.of(CaptureDriver.Effect.INVOKE_CAPTURE),
                first.onGuardEvaluated(CaptureGuard.noScreen(false, false, true, "ok")));
        assertEquals(List.of(CaptureDriver.Effect.AWAIT_FILE), first.onCaptureCompleted());
        assertEquals(List.of(CaptureDriver.Effect.DONE), first.onScreenshotReady());

        // The controller reuses this observer after advance(); its cancellation and
        // new POSITION completion must give the later frame a clean first observation.
        window.cancel();
        CaptureDriver later = new CaptureDriver(3);
        assertEquals(List.of(CaptureDriver.Effect.SCHEDULE_POSITION), later.start());
        assertEquals(PositionObservationWindow.TaskResult.OBSERVING,
                window.serverTaskCompleted(null));
        assertEquals(PositionObservationWindow.Observation.CONVERGED,
                window.observe(true));
        assertEquals(List.of(CaptureDriver.Effect.APPLY_CLIENT_SETTINGS),
                later.onPositionApplied(true, "frame 1", "frame-1"));
        assertEquals(1, later.positionAttempts());
        assertEquals(1, window.observedTicks());
    }

    @Test
    void cancellationStopsAnUnfinishedObservationAtPlanTermination() {
        PositionObservationWindow window = new PositionObservationWindow(5);
        assertEquals(PositionObservationWindow.TaskResult.OBSERVING,
                window.serverTaskCompleted(null));
        assertEquals(PositionObservationWindow.Observation.WAITING, window.observe(false));
        window.cancel();
        assertFalse(window.isObserving());
        assertEquals(0, window.observedTicks());
        assertThrows(IllegalStateException.class, () -> window.observe(false));

        PoseStabilizer.Session session = new PoseStabilizer.Session();
        UUID player = UUID.fromString("47a6a81a-fd99-4a7d-914a-b46ad4dd3151");
        session.start(player);
        session.stop();
        assertFalse(session.matches(player),
                "plan termination must still stop the staged-player pose lock");
    }
}
