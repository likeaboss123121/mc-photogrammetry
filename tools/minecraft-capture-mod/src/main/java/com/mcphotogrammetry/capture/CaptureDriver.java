package com.mcphotogrammetry.capture;

import java.util.ArrayList;
import java.util.List;

/**
 * One explicit, Minecraft-free driver for the per-job capture sequence.
 *
 * <p>The third launch rejection found the controller running two state machines at
 * once: a {@code Phase} enum and a {@code JobFlow} helper that could disagree. A
 * completed POSITION called convergence while the helper was still at
 * {@code WAIT_POSITION} (an illegal transition that threw on every valid capture),
 * and a completed SETTINGS advanced the helper to SETTLE without ever moving the
 * controller phase, so the helper walked through CAPTURE and WAIT_FILE while the
 * controller never invoked {@code capture()}.
 *
 * <p>This class replaces both with a single authoritative machine. The controller
 * holds no stage state of its own: each step returns the {@link Effect}s that step
 * emitted, the controller applies them, and the next outcome is fed back in. The
 * required sequence is now
 *
 * <pre>
 * SCHEDULE_POSITION -&gt; CONVERGE -&gt; APPLY_CLIENT_SETTINGS
 *     -&gt; REASSERT POSITION -&gt; REASSERT CONVERGE -&gt; SCHEDULE_SETTINGS -&gt; SETTLE
 *     -&gt; EVALUATE_GUARD -&gt; INVOKE_CAPTURE -&gt; AWAIT_FILE -&gt; DONE
 * </pre>
 *
 * <p>The fourth launch rejection found that the controller no longer set FOV,
 * gamma, HUD, or window size at all: only record and restore remained. The client
 * settings are therefore an explicit stage of this machine, applied on the client
 * thread exactly once after the pose converged and before any server settings
 * task or settle window can run.
 *
 * <p>The DS-20 smoke launch found the remaining hole: the pose was measured once,
 * before the resize and the settle window, and never again. Frame 00 was captured
 * with a 1.35-degree yaw error, and frames 01-03 were captured with the ordinary
 * pause menu on screen after the window resize drew focus away. The machine now
 * re-asserts and re-measures feet/yaw/pitch after the client settings (which
 * include the window resize and any menu it caused) and once more immediately
 * before the screenshot, and it refuses to capture unless a fresh
 * {@link CaptureGuard} says there is no screen, the client is not paused, and the
 * pose re-converged. Window focus is recorded by the guard as telemetry only: the
 * interactive run happens on a hidden, unfocused Windows Desktop 2 where
 * {@code focused=false} is normal, so an unfocused window never triggers a retry,
 * a focus-stealing call, or a failure. A guard that needs recovery emits exactly
 * the recovery effects it needs and requires a fresh settle window before it is
 * consulted again; an unexpected screen is terminal.
 *
 * <p>Every transition is appended to {@link #stages()} in the order it actually
 * happened, and every outcome method refuses to run unless the machine is at the
 * stage that outcome belongs to. A missing, reordered, or duplicated step therefore
 * fails loudly instead of silently skipping a capture. A divergence during CONVERGE
 * emits another {@code SCHEDULE_POSITION} (still no settings and no capture); a
 * terminal convergence failure emits {@code FAIL_JOB} and can never reach SETTLE,
 * CAPTURE, or AWAIT_FILE.
 */
public final class CaptureDriver {

    /** Default number of guard evaluations before a never-ready capture is terminal. */
    public static final int DEFAULT_MAX_GUARD_ATTEMPTS = 3;

    /** The stages a job's machine actually entered, in order. */
    public enum Stage {
        POSITION_SCHEDULED, CONVERGE, CLIENT_SETTINGS, REASSERT_SCHEDULED,
        REASSERT_CONVERGE, SETTINGS_SCHEDULED, SETTLE, GUARD, CAPTURE_INVOKED,
        AWAIT_FILE, DONE, FAILED
    }

    /**
     * One thing the controller must do or observe next.
     *
     * <p>{@code SCHEDULE_POSITION} and {@code SCHEDULE_SETTINGS} are asynchronous:
     * the controller submits the server task and calls back when it completes.
     * {@code APPLY_CLIENT_SETTINGS} and {@code CLOSE_MENU} are synchronous and
     * client-thread only. {@code EVALUATE_GUARD} asks the controller to measure the
     * live screen/pause/pose state (plus focus as read-only telemetry) and report it
     * back. {@code INVOKE_CAPTURE} is synchronous: the controller calls
     * {@link #onCaptureCompleted()} as soon as it has started the screenshot writer.
     */
    public enum Effect {
        SCHEDULE_POSITION, APPLY_CLIENT_SETTINGS, CLOSE_MENU,
        SCHEDULE_SETTINGS, SETTLE, EVALUATE_GUARD, INVOKE_CAPTURE, AWAIT_FILE, DONE,
        FAIL_JOB
    }

    private final int maxPositionAttempts;
    private final int maxGuardAttempts;
    private final List<Stage> stages = new ArrayList<>();

    private boolean started;
    private int positionAttempts;
    private int reassertAttempts;
    private int guardAttempts;
    private int recoveries;
    private boolean recovering;
    private boolean settleElapsed;
    private boolean terminal;
    private String failureDetail;

    /**
     * @param maxPositionAttempts how many teleports may be scheduled before a
     *                            divergence is terminal; at least one
     */
    public CaptureDriver(int maxPositionAttempts) {
        this(maxPositionAttempts, DEFAULT_MAX_GUARD_ATTEMPTS);
    }

    /**
     * @param maxPositionAttempts how many teleports may be scheduled before a
     *                            divergence is terminal; at least one
     * @param maxGuardAttempts    how many capture-guard evaluations may fail to
     *                            become ready before the job is terminal; at least one
     */
    public CaptureDriver(int maxPositionAttempts, int maxGuardAttempts) {
        if (maxPositionAttempts < 1) {
            throw new IllegalArgumentException("at least one position attempt is required");
        }
        if (maxGuardAttempts < 1) {
            throw new IllegalArgumentException("at least one guard attempt is required");
        }
        this.maxPositionAttempts = maxPositionAttempts;
        this.maxGuardAttempts = maxGuardAttempts;
    }

    /** Begin a job: schedule its first POSITION and nothing else. */
    public List<Effect> start() {
        if (started) {
            throw new IllegalStateException("the job driver was already started");
        }
        started = true;
        positionAttempts = 1;
        stages.add(Stage.POSITION_SCHEDULED);
        return List.of(Effect.SCHEDULE_POSITION);
    }

    /** The stages this job actually entered, in order; used by the driver test. */
    public List<Stage> stages() {
        return List.copyOf(stages);
    }

    /** The stage the machine is waiting at right now, or null before {@link #start()}. */
    public Stage currentStage() {
        return stages.isEmpty() ? null : stages.get(stages.size() - 1);
    }

    /** True once the job has failed terminally; no capture may follow. */
    public boolean failed() {
        return terminal;
    }

    /** The human-readable why of a terminal failure, or null when none happened. */
    public String failureDetail() {
        return failureDetail;
    }

    public int positionAttempts() {
        return positionAttempts;
    }

    public int reassertAttempts() {
        return reassertAttempts;
    }

    public int guardAttempts() {
        return guardAttempts;
    }

    public int recoveries() {
        return recoveries;
    }

    /**
     * A teleport outcome. The method is stage-aware on purpose: the controller
     * submits the same server task for the initial pose and for the post-settings
     * re-assertion, so a single entry point removes the chance that the two paths
     * disagree about which stage they are reporting.
     *
     * @param converged whether the recorded client pose is within tolerance
     * @param detail    the measured divergence, recorded on a terminal failure
     * @param frameId   the job being checked, for the retry log
     */
    public List<Effect> onPositionApplied(boolean converged, String detail, String frameId) {
        Stage current = currentStage();
        if (current == Stage.POSITION_SCHEDULED) {
            return onInitialPosition(converged, detail);
        }
        if (current == Stage.REASSERT_SCHEDULED) {
            return onReassertedPosition(converged, detail);
        }
        throw new IllegalStateException("a position outcome arrived at stage " + current
                + " for " + frameId);
    }

    private List<Effect> onInitialPosition(boolean converged, String detail) {
        require(Stage.POSITION_SCHEDULED, "a position outcome");
        if (!converged) {
            if (positionAttempts >= maxPositionAttempts) {
                return terminate("client did not converge on the requested pose after "
                        + positionAttempts + " teleports: " + detail);
            }
            // A divergence retries POSITION and nothing else: client settings, server
            // settings, and capture stay unreachable from here.
            positionAttempts++;
            stages.add(Stage.POSITION_SCHEDULED);
            return List.of(Effect.SCHEDULE_POSITION);
        }
        stages.add(Stage.CONVERGE);
        stages.add(Stage.CLIENT_SETTINGS);
        return List.of(Effect.APPLY_CLIENT_SETTINGS);
    }

    /**
     * The requested client options (FOV, gamma, HUD, window size) were applied on
     * the client thread. This is the only way to reach the server settings task, so
     * the client settings can never be skipped or run after the world was mutated.
     *
     * <p>The window resize can move the player and can open the ordinary pause menu,
     * so the pose is re-asserted before anything else happens. The resize may also
     * leave the window unfocused; that is normal on the hidden capture desktop and is
     * deliberately ignored here (no focus-stealing call is emitted).
     */
    public List<Effect> onClientSettingsApplied() {
        require(Stage.CLIENT_SETTINGS, "a client-settings outcome");
        stages.add(Stage.REASSERT_SCHEDULED);
        return List.of(Effect.SCHEDULE_POSITION);
    }

    private List<Effect> onReassertedPosition(boolean converged, String detail) {
        require(Stage.REASSERT_SCHEDULED, "a position re-assertion outcome");
        if (!converged) {
            if (reassertAttempts >= maxPositionAttempts) {
                return terminate("client did not re-converge on the requested pose after "
                        + (reassertAttempts + 1) + " re-assertions: " + detail);
            }
            reassertAttempts++;
            stages.add(Stage.REASSERT_SCHEDULED);
            return List.of(Effect.SCHEDULE_POSITION);
        }
        stages.add(Stage.REASSERT_CONVERGE);
        if (recovering) {
            // A recovered job (menu closed, pose re-asserted) does not need the world
            // settings applied again, but it does need a fresh settle window: the old
            // one is void, and the screenshot must not be taken on it.
            recovering = false;
            return enterSettle();
        }
        stages.add(Stage.SETTINGS_SCHEDULED);
        return List.of(Effect.SCHEDULE_SETTINGS);
    }

    /**
     * Enter the settle window. The returned {@code SETTLE} effect means "wait the
     * job's declared settle ticks, then call {@link #onSettleElapsed()}"; settings
     * and capture cannot run in between.
     */
    public List<Effect> onSettingsApplied() {
        require(Stage.SETTINGS_SCHEDULED, "a settings outcome");
        return enterSettle();
    }

    private List<Effect> enterSettle() {
        settleElapsed = false;
        stages.add(Stage.SETTLE);
        return List.of(Effect.SETTLE);
    }

    /**
     * The declared settle ticks have passed. The job does not capture yet: the
     * controller must first measure the live capture guard
     * ({@code screen}/{@code paused}/pose, plus {@code focused} as read-only
     * telemetry) and report it back.
     */
    public List<Effect> onSettleElapsed() {
        require(Stage.SETTLE, "a settle expiry");
        if (settleElapsed) {
            throw new IllegalStateException("settle expired more than once");
        }
        settleElapsed = true;
        stages.add(Stage.GUARD);
        return List.of(Effect.EVALUATE_GUARD);
    }

    /**
     * The live capture guard was measured immediately before the screenshot.
     *
     * <p>The only way to reach capture. A ready guard emits {@code INVOKE_CAPTURE};
     * an unexpected screen is terminal; anything else emits the recovery effects it
     * needs and requires a fresh position re-assertion plus settle window before
     * the guard is consulted again. A guard that never becomes ready is bounded by
     * {@code maxGuardAttempts}.
     *
     * <p>Focus is not consulted here. A window that is simply unfocused (the normal
     * state on a hidden capture desktop) produces a ready guard whenever the screen
     * is clear, the client is unpaused, and the pose converged; it never triggers a
     * retry, a focus-stealing effect, or a failure on its own.
     */
    public List<Effect> onGuardEvaluated(CaptureGuard guard) {
        require(Stage.GUARD, "a capture-guard outcome");
        if (guard == null) {
            throw new IllegalArgumentException("a capture-guard outcome is required");
        }
        guardAttempts++;
        if (guard.ready()) {
            stages.add(Stage.CAPTURE_INVOKED);
            return List.of(Effect.INVOKE_CAPTURE);
        }
        if (guard.screen() == CaptureGuard.ScreenKind.UNEXPECTED) {
            return terminate("refusing to capture behind the unexpected screen '"
                    + guard.screenType() + "': " + guard.describe());
        }
        if (guard.paused() && guard.screen() == CaptureGuard.ScreenKind.NONE) {
            // There is no screen to close, so nothing the controller can do will
            // clear the pause: fail rather than capture a paused world.
            return terminate("refusing to capture while the client is paused with no "
                    + "recoverable screen: " + guard.describe());
        }
        if (guardAttempts >= maxGuardAttempts) {
            return terminate("the capture guard did not become ready after "
                    + guardAttempts + " evaluations: " + guard.describe());
        }
        List<Effect> effects = new ArrayList<>();
        if (guard.screen() == CaptureGuard.ScreenKind.GAME_MENU) {
            effects.add(Effect.CLOSE_MENU);
        }
        // Focus is telemetry only. The hidden capture desktop is normally unfocused,
        // so an unfocused window is not a recoverable contamination and must not add
        // a focus-stealing effect or a retry of its own.
        // The pose is always re-asserted, and the recovery always demands a fresh
        // settle window, so a frame can never be captured on the settle window that
        // was running while the menu was open or the pose was drifting.
        recovering = true;
        recoveries++;
        stages.add(Stage.REASSERT_SCHEDULED);
        effects.add(Effect.SCHEDULE_POSITION);
        return List.copyOf(effects);
    }

    /** The screenshot writer was started; the driver now waits for its file. */
    public List<Effect> onCaptureCompleted() {
        require(Stage.CAPTURE_INVOKED, "a capture completion");
        stages.add(Stage.AWAIT_FILE);
        return List.of(Effect.AWAIT_FILE);
    }

    /** The completed, non-empty screenshot was accepted. */
    public List<Effect> onScreenshotReady() {
        require(Stage.AWAIT_FILE, "a screenshot");
        stages.add(Stage.DONE);
        return List.of(Effect.DONE);
    }

    /**
     * A terminal failure that did not come from the machine's own decisions (a
     * missing server, a screenshot timeout, an unreadable file). Capture can never
     * be invoked afterwards.
     */
    public List<Effect> fail() {
        if (terminal) {
            return List.of(Effect.FAIL_JOB);
        }
        return terminate("the job failed outside the driver's stages");
    }

    private List<Effect> terminate(String detail) {
        terminal = true;
        failureDetail = detail;
        stages.add(Stage.FAILED);
        return List.of(Effect.FAIL_JOB);
    }

    /** Refuse an outcome that does not belong to the stage the machine is at. */
    private void require(Stage expected, String label) {
        if (terminal) {
            throw new IllegalStateException(label + " arrived after the job failed");
        }
        if (stages.isEmpty() || stages.get(stages.size() - 1) != expected) {
            throw new IllegalStateException(label + " arrived at stage "
                    + (stages.isEmpty() ? "none" : stages.get(stages.size() - 1))
                    + " instead of " + expected);
        }
    }
}
