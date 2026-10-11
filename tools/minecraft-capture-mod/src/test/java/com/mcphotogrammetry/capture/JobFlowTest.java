package com.mcphotogrammetry.capture;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The launch-rejection transition test.
 *
 * <p>The first submission could run a job without ever scheduling its teleport:
 * the apply loop saw no pending action and went straight to SETTINGS. This test
 * pins the only permitted order - POSITION, convergence, SETTINGS, settle, capture,
 * file wait, done - and proves that a stage cannot be skipped, repeated, or reached
 * out of order, and that a failed convergence never reaches settle or capture.
 */
class JobFlowTest {

    @Test
    void runsEveryStageExactlyOnceInTheRequiredOrder() {
        JobFlow flow = new JobFlow();
        assertEquals(JobFlow.Stage.WAIT_POSITION, flow.stage());
        assertEquals(JobFlow.Action.POSITION, flow.requiredAction(),
                "the first stage of every job must schedule the position task");

        // POSITION task done -> CONVERGE.
        flow.next();
        assertEquals(JobFlow.Stage.CONVERGE, flow.stage());
        assertTrue(flow.awaitingConvergence());
        assertNull(flow.requiredAction(), "convergence waits on the client, not the server");
        assertTrue(flow.confirmConvergence(true));

        // SETTINGS.
        assertEquals(JobFlow.Stage.WAIT_SETTINGS, flow.stage());
        assertEquals(JobFlow.Action.SETTINGS, flow.requiredAction());
        flow.next();

        assertEquals(JobFlow.Stage.SETTLE, flow.stage());
        flow.next();
        assertEquals(JobFlow.Stage.CAPTURE, flow.stage());
        flow.next();
        assertEquals(JobFlow.Stage.WAIT_FILE, flow.stage());
        flow.next();
        assertEquals(JobFlow.Stage.DONE, flow.stage());
        assertThrows(IllegalStateException.class, flow::next,
                "a finished job must not advance again");

        List<JobFlow.Stage> visited = flow.visited();
        assertEquals(List.of(JobFlow.Stage.WAIT_POSITION, JobFlow.Stage.CONVERGE,
                JobFlow.Stage.WAIT_SETTINGS, JobFlow.Stage.SETTLE, JobFlow.Stage.CAPTURE,
                JobFlow.Stage.WAIT_FILE, JobFlow.Stage.DONE), visited);
        assertEquals(visited.size(), visited.stream().distinct().count(),
                "no stage may be visited twice");
    }

    @Test
    void theRequiredActionSequenceIsPositionThenSettingsExactlyOnceEach() {
        List<JobFlow.Action> scheduled = new java.util.ArrayList<>();
        JobFlow flow = new JobFlow();
        JobFlow.Action first = flow.requiredAction();
        assertNotNull(first);
        scheduled.add(first);
        flow.next();
        assertTrue(flow.confirmConvergence(true));
        JobFlow.Action second = flow.requiredAction();
        assertNotNull(second);
        scheduled.add(second);
        flow.next();
        assertNull(flow.requiredAction(), "settle and capture submit no server task");

        assertEquals(List.of(JobFlow.Action.POSITION, JobFlow.Action.SETTINGS), scheduled,
                "the launch defect was a missing POSITION and/or a reordered SETTINGS");
    }

    @Test
    void failedConvergenceCannotReachSettleOrCapture() {
        JobFlow flow = new JobFlow();
        flow.next();
        assertEquals(JobFlow.Stage.CONVERGE, flow.stage());
        assertFalse(flow.confirmConvergence(false),
                "a client that never reached the pose must not continue");
        assertEquals(JobFlow.Stage.CONVERGE, flow.stage(),
                "the flow stays at the convergence stage so neither settle nor capture runs");
        assertTrue(flow.awaitingConvergence(),
                "the job is still waiting on the client and submits no next task");
        assertNull(flow.requiredAction(),
                "no SERVER task belongs to a failed convergence; the job must fail");
        assertFalse(flow.visited().contains(JobFlow.Stage.SETTLE));
        assertFalse(flow.visited().contains(JobFlow.Stage.CAPTURE));

        // The only way onward is a convergence that actually succeeded.
        assertTrue(flow.confirmConvergence(true));
        assertEquals(JobFlow.Stage.WAIT_SETTINGS, flow.stage());
        assertFalse(flow.visited().contains(JobFlow.Stage.SETTLE));
        assertFalse(flow.visited().contains(JobFlow.Stage.CAPTURE));
    }

    @Test
    void convergenceCannotBeReportedFromTheWrongStage() {
        assertThrows(IllegalStateException.class, () -> new JobFlow().confirmConvergence(true));
        JobFlow settled = new JobFlow();
        settled.next();
        settled.confirmConvergence(true);
        settled.next();
        assertEquals(JobFlow.Stage.SETTLE, settled.stage());
        assertThrows(IllegalStateException.class, () -> settled.confirmConvergence(true));
        settled.next();
        assertEquals(JobFlow.Stage.CAPTURE, settled.stage());
        assertThrows(IllegalStateException.class, () -> settled.confirmConvergence(true));
    }

    @Test
    void everyStageMapsToItsOwnDocumentedSuccessor() {
        assertEquals(JobFlow.Stage.CONVERGE, JobFlow.following(JobFlow.Stage.WAIT_POSITION));
        assertEquals(JobFlow.Stage.WAIT_SETTINGS, JobFlow.following(JobFlow.Stage.CONVERGE));
        assertEquals(JobFlow.Stage.SETTLE, JobFlow.following(JobFlow.Stage.WAIT_SETTINGS));
        assertEquals(JobFlow.Stage.CAPTURE, JobFlow.following(JobFlow.Stage.SETTLE));
        assertEquals(JobFlow.Stage.WAIT_FILE, JobFlow.following(JobFlow.Stage.CAPTURE));
        assertEquals(JobFlow.Stage.DONE, JobFlow.following(JobFlow.Stage.WAIT_FILE));
        assertNull(JobFlow.following(JobFlow.Stage.DONE));
        assertNull(JobFlow.actionFor(JobFlow.Stage.CONVERGE));
        assertNull(JobFlow.actionFor(JobFlow.Stage.SETTLE));
        assertNull(JobFlow.actionFor(JobFlow.Stage.CAPTURE));
        assertNull(JobFlow.actionFor(JobFlow.Stage.WAIT_FILE));
        assertNull(JobFlow.actionFor(JobFlow.Stage.DONE));
    }
}
