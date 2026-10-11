package com.mcphotogrammetry.capture;

/**
 * Bounds client-side observation after an asynchronous server POSITION task.
 *
 * <p>A completed server task proves that the teleport was applied on the server;
 * it does not prove that the client has consumed the resulting network update.
 * The controller asks this window about the client pose once per client tick and
 * reports a failed attempt only after the observation budget expires. The default
 * budget is one nominal second at Minecraft's 20 client ticks per second. This is
 * tick-driven, not a sleep, so a converged pose succeeds on the first tick it is
 * observed.
 */
public final class PositionObservationWindow {

    public static final int DEFAULT_TIMEOUT_TICKS = 20;

    public enum TaskResult {
        OBSERVING, FAILED
    }

    public enum Observation {
        WAITING, CONVERGED, TIMED_OUT
    }

    private final int timeoutTicks;
    private boolean observing;
    private int observedTicks;

    public PositionObservationWindow() {
        this(DEFAULT_TIMEOUT_TICKS);
    }

    public PositionObservationWindow(int timeoutTicks) {
        if (timeoutTicks < 1) {
            throw new IllegalArgumentException("at least one observation tick is required");
        }
        this.timeoutTicks = timeoutTicks;
    }

    /** Start observing after a successful server task, or report its error immediately. */
    public TaskResult serverTaskCompleted(String error) {
        cancel();
        if (error != null) {
            return TaskResult.FAILED;
        }
        observing = true;
        return TaskResult.OBSERVING;
    }

    /** Report one client-tick pose measurement. */
    public Observation observe(boolean converged) {
        if (!observing) {
            throw new IllegalStateException("no completed POSITION task is being observed");
        }
        observedTicks++;
        if (converged) {
            observing = false;
            return Observation.CONVERGED;
        }
        if (observedTicks >= timeoutTicks) {
            observing = false;
            return Observation.TIMED_OUT;
        }
        return Observation.WAITING;
    }

    public boolean isObserving() {
        return observing;
    }

    public int observedTicks() {
        return observedTicks;
    }

    /** Cancel pending observation when the job or plan terminates. */
    public void cancel() {
        observing = false;
        observedTicks = 0;
    }
}
