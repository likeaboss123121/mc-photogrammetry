package com.mcphotogrammetry.capture;

import java.util.ArrayList;
import java.util.List;

/**
 * The Minecraft-free per-job stage machine.
 *
 * <p>The launch rejection found that the controller could run a job without ever
 * scheduling its teleport: {@code startApply()} moved straight to the general
 * apply loop, which saw no pending action and no recorded world settings and
 * scheduled <em>settings</em> first. This class makes the required order explicit
 * and testable without Minecraft:
 *
 * <pre>POSITION -&gt; CONVERGE -&gt; SETTINGS -&gt; SETTLE -&gt; CAPTURE -&gt; WAIT_FILE -&gt; DONE</pre>
 *
 * <p>{@link #next()} refuses to skip or repeat a stage, and {@link #converged()}
 * refuses to leave the convergence stage unless the client actually reached the
 * requested pose. A job whose convergence fails therefore never reaches SETTLE or
 * CAPTURE.
 */
public final class JobFlow {

    /** Every stage a job passes through, in the only permitted order. */
    public enum Stage {
        WAIT_POSITION, CONVERGE, WAIT_SETTINGS, SETTLE, CAPTURE, WAIT_FILE, DONE
    }

    /** The server-thread task a stage submits, if any. */
    public enum Action {
        POSITION, SETTINGS
    }

    private Stage stage = Stage.WAIT_POSITION;
    private final List<Stage> visited = new ArrayList<>();

    public JobFlow() {
        visited.add(stage);
    }

    public Stage stage() {
        return stage;
    }

    /** The stages this job has occupied, in order; used by the transition test. */
    public List<Stage> visited() {
        return List.copyOf(visited);
    }

    /** The server task the current stage must submit, or null when it submits none. */
    public Action requiredAction() {
        return actionFor(stage);
    }

    public static Action actionFor(Stage stage) {
        return switch (stage) {
            case WAIT_POSITION -> Action.POSITION;
            case WAIT_SETTINGS -> Action.SETTINGS;
            default -> null;
        };
    }

    /** True when the current stage is waiting on the client to reach the pose. */
    public boolean awaitingConvergence() {
        return stage == Stage.CONVERGE;
    }

    /**
     * Advance one stage. Called after the stage's work has actually happened, so a
     * missing or reordered step shows up as an illegal transition instead of a
     * silently skipped capture.
     */
    public Stage next() {
        Stage following = following(stage);
        if (following == null) {
            throw new IllegalStateException("the job already finished at " + stage);
        }
        stage = following;
        visited.add(stage);
        return stage;
    }

    public static Stage following(Stage stage) {
        return switch (stage) {
            case WAIT_POSITION -> Stage.CONVERGE;
            case CONVERGE -> Stage.WAIT_SETTINGS;
            case WAIT_SETTINGS -> Stage.SETTLE;
            case SETTLE -> Stage.CAPTURE;
            case CAPTURE -> Stage.WAIT_FILE;
            case WAIT_FILE -> Stage.DONE;
            case DONE -> null;
        };
    }

    /**
     * Leave the convergence wait, but only when the client really converged.
     *
     * @param converged whether the recorded client pose is within tolerance
     * @return true when the job may continue; false means the caller must fail the
     *         job, and the stage stays put so neither SETTLE nor CAPTURE can run
     */
    public boolean confirmConvergence(boolean converged) {
        if (stage != Stage.CONVERGE) {
            throw new IllegalStateException("convergence was reported from stage " + stage);
        }
        if (!converged) {
            return false;
        }
        next();
        return true;
    }
}
