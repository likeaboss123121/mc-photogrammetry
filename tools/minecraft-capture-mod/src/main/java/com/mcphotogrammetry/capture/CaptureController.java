package com.mcphotogrammetry.capture;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.GameMenuScreen;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.util.ScreenshotRecorder;
import net.minecraft.entity.Entity;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.Vec3d;
import net.minecraft.util.WorldSavePath;
import net.minecraft.world.GameRules;
import net.minecraft.world.TeleportTarget;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.UUID;

/**
 * Drives one armed capture plan on the client tick.
 *
 * <p>The mod is inert until {@code config/mc-photo-capture/armed.json} names a
 * plan whose file digest and schema validate. Once armed it only acts inside an
 * integrated singleplayer world whose level name matches the plan and whose own
 * save root carries the plan's staging token. It applies one bounded job at a time
 * in the explicit order {@code POSITION -> client observation -> CONVERGE -> client settings -> re-assert
 * POSITION -> CONVERGE -> SETTINGS -> SETTLE -> capture guard -> capture}, records
 * what it actually observed (including failures), exports the server world's block
 * states, restores the client options it changed, and atomically disarms
 * {@code armed.json} so the next launch is inert again.
 *
 * <p>Requested client options (FOV, gamma, HUD, window size) are applied on the
 * client thread by the {@code APPLY_CLIENT_SETTINGS} effect, exactly once, after
 * the pose converged and before any server settings task or settle window runs.
 * The result records the request and the actual client/world state separately, and
 * the requested capture time is applied to a staged copy whose daylight cycle is
 * frozen first, so the recorded time/weather describe the screenshot rather than a
 * pre-settle sample.
 *
 * <p>The sequence is owned by one Minecraft-free machine, {@link CaptureDriver};
 * this class holds no job-stage state of its own. It submits the server task an
 * effect asks for, waits for completion, observes client pose convergence for a
 * bounded tick window, and feeds one outcome back, so the controller and stage
 * machine cannot disagree about whether a teleport succeeded.
 *
 * <p>Every integrated-server mutation (player teleport, time, weather) and the
 * ground-truth export run as bounded tasks submitted to the server thread; the
 * client tick only schedules them and polls a completion flag. No {@code ServerWorld}
 * is touched from the client thread. A plan's {@code camera.position} is the
 * player's <em>feet</em>, so the server player is teleported to exactly those feet
 * and the recorded eye is {@code feet + eye_height}. After the staged world's
 * marker is validated, a UUID-scoped pose lock clears motion and fall state,
 * disables gravity, and prevents damage on both the client and integrated-server
 * player ticks until the plan terminates. The lock is deliberately not restored:
 * capture runs against an isolated staged-world copy in a disposable worker, and
 * the source save is never modified.
 *
 * <p>It never sends a command, never reads F3, and never touches a multiplayer
 * server: the singleplayer check happens before any player state is modified.
 */
public final class CaptureController {

    private static final Logger LOGGER = LoggerFactory.getLogger("mc-photo-capture");
    private static final int FILE_WAIT_TICKS = 400;
    private static final int SERVER_WAIT_TICKS = 1200;
    private static final int GROUND_TRUTH_WAIT_TICKS = 600;
    private static final int MAX_POSITION_ATTEMPTS = 3;
    private static final long SESSION_TICK_LIMIT = 60_000L;
    private static final double MOVE_TOLERANCE = CapturePlumbing.DEFAULT_POSITION_TOLERANCE;
    private static final double HEADING_TOLERANCE = PlanAudit.HEADING_TOLERANCE_DEGREES;
    private static final String OUTPUT_DIRECTORY = "mc-photo-capture/out";

    private enum Phase {
        IDLE, START_APPLY, APPLY, WAIT_FILE, FINISH, DONE
    }

    /**
     * One in-flight server task.
     *
     * <p>Identity lives in this object, not in shared mutable fields: the task
     * marks <em>its own</em> ticket complete, so a late callback can never be read
     * as the success of the next task. That is the task-state race the second
     * review flagged.
     */
    private static final class Ticket {
        private final JobFlow.Action action;
        private final String label;
        private volatile boolean completed;
        private volatile String error;

        Ticket(JobFlow.Action action, String label) {
            this.action = action;
            this.label = label;
        }
    }

    private final List<JsonObject> results = new ArrayList<>();
    private final Set<String> recordedFrames = new LinkedHashSet<>();
    private final Set<String> groundTruthErrors = new LinkedHashSet<>();
    private final PoseStabilizer.Session poseSession = new PoseStabilizer.Session();
    private final PositionObservationWindow positionObservation =
            new PositionObservationWindow();

    private Path configDir;
    private Path armedPath;
    private PlanAudit.AuditedPlan plan;
    private Phase phase = Phase.DONE;
    private int jobIndex;
    private int waitTicks;
    private int fileWaitTicks;
    private int serverWaitTicks;
    private int groundTruthWaitTicks;
    private long sessionTicks;
    private CaptureDriver jobDriver;
    private String lastConvergenceDetail = "";
    /** The last live capture-guard measurement, recorded into the frame evidence. */
    private CaptureGuard lastGuard;
    /** The last pose measurement, so capture evidence carries the numeric deltas. */
    private MoveCheck lastPoseCheck;
    /** Recovery counters, recorded per frame so a repaired run is visible. */
    private int menuCloses;
    private Ticket pendingTask;
    private File screenshotFile;
    /** Written by the recorder's callback (off the client tick), so volatile. */
    private volatile boolean screenshotCallbackDone;
    private volatile String screenshotCallbackError;
    private String screenshotDigest;
    private JsonObject provenanceExpectation;
    private JsonObject runtimeProvenance;
    private JsonObject pendingCaptureSnapshot;
    private String groundTruthJson;
    private Ticket groundTruthTask;
    private boolean runFailed;
    private boolean finished;
    private volatile boolean groundTruthExported;
    /** Actual server-side time/weather, sampled on the server thread after applying. */
    private volatile ServerActuals serverActuals;
    private Integer originalFov;
    private Double originalGamma;
    private boolean originalHudHidden;
    private int originalWindowWidth = -1;
    private int originalWindowHeight = -1;

    /** Read and validate the arming file. Absent file means completely inert. */
    public void load() {
        configDir = FabricLoader.getInstance().getConfigDir().resolve(PlanAudit.PLAN_SUBDIRECTORY);
        armedPath = configDir.resolve("armed.json");
        if (!Files.isRegularFile(armedPath)) {
            LOGGER.info("mc-photo-capture: no armed.json, staying inert");
            phase = Phase.DONE;
            return;
        }
        try {
            JsonObject armed = JsonParser
                    .parseString(Files.readString(armedPath, StandardCharsets.UTF_8))
                    .getAsJsonObject();
            String relative = requireText(armed, "plan_path");
            String expectedDigest = requireText(armed, "plan_sha256");
            Path gameDir = FabricLoader.getInstance().getGameDir();
            Path resolved = PlanAudit.resolveContained(gameDir, relative);
            byte[] planBytes = Files.readAllBytes(resolved);
            PlanAudit.verifyDigest(planBytes, expectedDigest);
            plan = PlanAudit.parse(PlanAudit.utf8(planBytes));
            if (armed.has("runtime_render_provenance")) {
                provenanceExpectation = armed.getAsJsonObject("runtime_render_provenance").deepCopy();
                if (provenanceExpectation.get("plan_id").getAsString().equals(plan.planId())
                        && provenanceExpectation.get("plan_sha256").getAsString().equals(expectedDigest)
                        && provenanceExpectation.get("staging_token").getAsString().equals(plan.stagingToken())) {
                    runtimeProvenance = RuntimeRenderProvenance.create(
                            FabricLoader.getInstance(), provenanceExpectation);
                } else {
                    throw new IllegalArgumentException("runtime/render provenance expectation does not bind the armed plan");
                }
            }
            // The plan's staging token is already validated as a real 64-character
            // digest by PlanAudit, so no game-directory side file is consulted here.
            // The world the client actually opens is proven later, in begin(), by
            // the marker inside that world's own save root.
            phase = Phase.IDLE;
            LOGGER.info("mc-photo-capture: armed for plan {} ({} jobs), digest verified; "
                            + "the opened world's in-save staging marker is checked on entry",
                    plan.planId(), plan.jobs().size());
        } catch (Exception error) {
            reject("armed plan was rejected: " + error.getMessage());
        }
    }

    public void tick(MinecraftClient client) {
        if (plan == null || phase == Phase.DONE) {
            return;
        }
        maintainClientPlayer(client);
        sessionTicks++;
        if (sessionTicks > SESSION_TICK_LIMIT) {
            abort(client, "session tick limit exceeded before the plan finished");
            return;
        }
        switch (phase) {
            case IDLE -> begin(client);
            case START_APPLY -> startApply(client);
            case APPLY -> apply(client);
            case WAIT_FILE -> awaitFile(client);
            case FINISH -> finish(client);
            default -> {
            }
        }
    }

    private void begin(MinecraftClient client) {
        if (client.player == null || client.world == null || client.getServer() == null) {
            if (++serverWaitTicks > SERVER_WAIT_TICKS) {
                abort(client, "no integrated world was entered before the wait limit");
            }
            return;
        }
        if (!client.isInSingleplayer()) {
            abort(client, "refusing to run outside an integrated singleplayer world");
            return;
        }
        String levelName = client.getServer().getSaveProperties().getLevelName();
        if (!plan.worldIdentity().equals(levelName)) {
            abort(client, "world identity mismatch: plan expects '" + plan.worldIdentity()
                    + "' but the open world is '" + levelName + "'");
            return;
        }
        // Level name alone is not identity, and neither is the save *folder* name:
        // the integrated server resolves the real open save root, including a world
        // whose folder differs from its LevelName. The stager placed the marker
        // inside the copied world itself, so reading it proves this exact world was
        // prepared for this plan. The live tree is never rehashed: Minecraft creates
        // and updates files after a copy, which would fail a valid older copy and
        // stall the client on a multi-GB save.
        Path worldRoot = client.getServer().getSavePath(WorldSavePath.ROOT);
        try {
            StagingMarker.readAndValidateWorld(worldRoot, plan.stagingToken(),
                    plan.worldIdentity());
        } catch (StagingMarker.StagingMismatch error) {
            abort(client, "staged-copy identity mismatch for the open world " + worldRoot
                    + ": " + error.getMessage());
            return;
        }
        if (plan.contextExportOnly()) {
            // Context export has no camera jobs. Keep the player, options, and
            // screenshot recorder untouched; the already-validated staged world
            // is enough for the bounded server-thread block export.
            jobIndex = 0;
            phase = Phase.FINISH;
            LOGGER.info("mc-photo-capture: validated context-export-only plan {} for {}",
                    plan.planId(), plan.sceneId());
            return;
        }
        recordOriginals(client);
        poseSession.start(client.player.getUuid());
        maintainClientPlayer(client);
        jobIndex = 0;
        phase = Phase.START_APPLY;
    }

    /** Called by the client tick hook before vanilla can apply local gravity. */
    public void maintainClientPlayer(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player != null && poseSession.matches(player.getUuid())) {
            maintainEntity(player);
        }
    }

    /** Called by the server-player tick mixin on the integrated server thread. */
    public void maintainServerPlayer(ServerPlayerEntity player) {
        if (player != null && poseSession.matches(player.getUuid())) {
            maintainEntity(player);
        }
    }

    private static void maintainEntity(Entity entity) {
        PoseStabilizer.hold(new PoseStabilizer.Target() {
            @Override
            public void setNoGravity(boolean noGravity) {
                entity.setNoGravity(noGravity);
            }

            @Override
            public void setVelocity(double x, double y, double z) {
                entity.setVelocity(x, y, z);
            }

            @Override
            public void setFallDistance(float fallDistance) {
                entity.fallDistance = fallDistance;
            }

            @Override
            public void setInvulnerable(boolean invulnerable) {
                entity.setInvulnerable(invulnerable);
            }
        });
    }

    /**
     * Arm the per-job state, then start the driver.
     *
     * <p>Resetting every per-job field here - including the driver itself - is what
     * guarantees an initial POSITION is always scheduled: a fresh
     * {@link CaptureDriver} begins at {@code POSITION_SCHEDULED} and emits exactly
     * one {@code SCHEDULE_POSITION}. There is no second state machine to disagree
     * with it; the controller only applies the effects the driver returns.
     */
    private void startApply(MinecraftClient client) {
        PlanAudit.CaptureJob job = plan.jobs().get(jobIndex);
        jobDriver = new CaptureDriver(MAX_POSITION_ATTEMPTS,
                CaptureDriver.DEFAULT_MAX_GUARD_ATTEMPTS);
        waitTicks = job.settleTicks();
        fileWaitTicks = 0;
        serverWaitTicks = 0;
        pendingTask = null;
        positionObservation.cancel();
        screenshotFile = null;
        screenshotDigest = null;
        screenshotCallbackDone = false;
        screenshotCallbackError = null;
        serverActuals = null;
        lastConvergenceDetail = "";
        lastGuard = null;
        lastPoseCheck = null;
        menuCloses = 0;
        phase = Phase.APPLY;
        applyEffects(client, "start", jobDriver.start());
    }

    /**
     * Drive the current stage: submit its server task, wait for it, observe any
     * client-side pose convergence window, then hand the outcome to the driver.
     *
     * <p>The only way into SETTLE is a POSITION whose client pose converged, and the only way into
     * capture is a SETTLE whose countdown expired, so a job that never reached the
     * requested pose can never be settled or captured.
     */
    private void apply(MinecraftClient client) {
        PlanAudit.CaptureJob job = plan.jobs().get(jobIndex);
        if (jobDriver == null) {
            failJob(job, "the job driver was missing when the job was applied");
            return;
        }
        if (positionObservation.isObserving()) {
            MoveCheck check = checkPosition(client, job);
            recordPose(check);
            PositionObservationWindow.Observation observation =
                    positionObservation.observe(check.ok());
            if (observation == PositionObservationWindow.Observation.WAITING) {
                return;
            }
            boolean converged = observation == PositionObservationWindow.Observation.CONVERGED;
            LOGGER.info("mc-photo-capture: client pose observation for {} -> {} after "
                            + "{} of {} ticks ({})",
                    job.frameId(), converged ? "CONVERGED" : "TIMED_OUT",
                    positionObservation.observedTicks(),
                    PositionObservationWindow.DEFAULT_TIMEOUT_TICKS, check.detail());
            applyEffects(client, "position", jobDriver.onPositionApplied(
                    converged, lastConvergenceDetail, job.frameId()));
            return;
        }
        if (pendingTask != null) {
            if (!pendingTask.completed) {
                if (++serverWaitTicks > SERVER_WAIT_TICKS) {
                    failJob(job, "the server did not finish " + pendingTask.label
                            + " within " + SERVER_WAIT_TICKS + " ticks");
                }
                return;
            }
            Ticket finished = pendingTask;
            pendingTask = null;
            serverWaitTicks = 0;
            if (finished.action == JobFlow.Action.POSITION) {
                if (positionObservation.serverTaskCompleted(finished.error)
                        == PositionObservationWindow.TaskResult.FAILED) {
                    failJob(job, finished.label + " failed on the server thread: "
                            + finished.error);
                } else {
                    LOGGER.info("mc-photo-capture: POSITION task completed for {}; "
                                    + "observing client convergence for up to {} ticks",
                            job.frameId(), PositionObservationWindow.DEFAULT_TIMEOUT_TICKS);
                }
            } else {
                if (finished.error != null) {
                    failJob(job, finished.label + " failed on the server thread: "
                            + finished.error);
                    return;
                }
                applyEffects(client, "settings", jobDriver.onSettingsApplied());
            }
            return;
        }
        if (isSettling(client, job)) {
            return;
        }
        failJob(job, "the job driver was awaiting an outcome that never arrived");
    }

    /** How a FAIL_JOB effect should read, given the step that produced it. */
    private String failureReason(String source) {
        if (jobDriver != null && jobDriver.failureDetail() != null) {
            return jobDriver.failureDetail();
        }
        return source.equals("position")
                ? "client did not converge on the requested pose after "
                        + MAX_POSITION_ATTEMPTS + " teleports: " + lastConvergenceDetail
                : "the job failed during " + source;
    }

    /**
     * True while the job is waiting out its declared settle ticks.
     *
     * <p>The countdown is owned by the driver stage, not by a one-shot phase: a
     * recovered job (pause menu closed, pose re-asserted) enters SETTLE a second
     * time and gets a full fresh window, so the screenshot can never be taken on the
     * settle window that was running while the menu was open.
     */
    private boolean isSettling(MinecraftClient client, PlanAudit.CaptureJob job) {
        if (jobDriver == null
                || jobDriver.currentStage() != CaptureDriver.Stage.SETTLE) {
            return false;
        }
        if (--waitTicks > 0) {
            return true;
        }
        applyEffects(client, "settle", jobDriver.onSettleElapsed());
        return true;
    }

    /** Apply the effects one driver step emitted; each effect is handled exactly once. */
    private void applyEffects(MinecraftClient client, String source,
                              List<CaptureDriver.Effect> effects) {
        PlanAudit.CaptureJob job = plan.jobs().get(jobIndex);
        MinecraftServer server = client.getServer();
        for (CaptureDriver.Effect effect : effects) {
            switch (effect) {
                case SCHEDULE_POSITION, SCHEDULE_SETTINGS -> {
                    if (server == null) {
                        failJob(job, "the integrated server disappeared before "
                                + effect + " could be scheduled");
                        return;
                    }
                    scheduleStage(client, server,
                            effect == CaptureDriver.Effect.SCHEDULE_POSITION
                                    ? JobFlow.Action.POSITION : JobFlow.Action.SETTINGS, job);
                }
                case APPLY_CLIENT_SETTINGS -> {
                    String error = applyClientSettings(client, job);
                    if (error != null) {
                        failJob(job, error);
                        return;
                    }
                    applyEffects(client, "client-settings", jobDriver.onClientSettingsApplied());
                }
                case SETTLE -> {
                    // Nothing to submit: the controller owns the settle countdown.
                    // Resetting it here is what makes a post-recovery settle a fresh
                    // window rather than the remains of the contaminated one.
                    waitTicks = job.settleTicks();
                    LOGGER.info("mc-photo-capture: {} -> SETTLE for {} ({} ticks)", source,
                            job.frameId(), waitTicks);
                }
                case EVALUATE_GUARD -> {
                    CaptureGuard guard = measureGuard(client, job);
                    lastGuard = guard;
                    LOGGER.info("mc-photo-capture: capture guard for {} -> {}", job.frameId(),
                            guard.describe());
                    applyEffects(client, "guard", jobDriver.onGuardEvaluated(guard));
                }
                case CLOSE_MENU -> {
                    if (!closeOrdinaryMenu(client, job)) {
                        return;
                    }
                }
                case INVOKE_CAPTURE -> capture(client);
                case FAIL_JOB -> {
                    failJob(job, failureReason(source));
                    return;
                }
                default -> {
                    // AWAIT_FILE and DONE are polling states: the file wait and the
                    // final record already drive them, so no action is due here.
                }
            }
        }
    }

    /** Record a terminal job failure and move to the next job exactly once. */
    private void failJob(PlanAudit.CaptureJob job, String reason) {
        if (jobDriver != null && !jobDriver.failed()) {
            jobDriver.fail();
        }
        recordFailure(job, reason);
        advance();
    }


    /**
     * Submit one bounded server-thread task for the current stage.
     *
     * <p>The client player's UUID is resolved on the client thread before the task
     * is submitted, so the server task targets exactly that player instead of
     * whatever happens to be first in the player list.
     */
    private void scheduleStage(MinecraftClient client, MinecraftServer server,
                               JobFlow.Action action, PlanAudit.CaptureJob job) {
        Ticket ticket = new Ticket(action, action.name());
        pendingTask = ticket;
        serverWaitTicks = 0;
        UUID playerId = client.player == null ? null : client.player.getUuid();
        switch (action) {
            case POSITION -> {
                maintainClientPlayer(client);
                scheduleTask(server, ticket, () -> teleportPlayer(server, playerId, job));
            }
            case SETTINGS -> scheduleTask(server, ticket, () -> {
                applyWorldSettings(server, job);
                readWorldSettings(server, this);
            });
        }
    }

    private void scheduleTask(MinecraftServer server, Ticket ticket, Runnable task) {
        server.execute(() -> {
            try {
                task.run();
            } catch (RuntimeException error) {
                ticket.error = String.valueOf(error);
            } finally {
                ticket.completed = true;
            }
        });
    }

    /**
     * Runs on the client thread: apply the plan's requested client options once.
     *
     * <p>Nothing else in the controller sets these values, so this is the only place
     * a capture can acquire its requested FOV, gamma, HUD state, or window size. It
     * is driven by the driver's {@code APPLY_CLIENT_SETTINGS} effect, which the
     * machine emits exactly once per job, after the pose converged and before the
     * server settings task: the window resize and any option change land before the
     * settle window, so they are in effect for the screenshot.
     *
     * @return null on success, or a failure reason recorded against the frame
     */
    private static String applyClientSettings(MinecraftClient client, PlanAudit.CaptureJob job) {
        try {
            // Minecraft's FOV option is an integer; the plan's value is rounded once,
            // and the finalizer compares the same rounded integer.
            client.options.getFov().setValue((int) Math.round(job.fovDegrees()));
            client.options.getGamma().setValue(job.settings().gamma());
            client.options.hudHidden = job.settings().hudHidden();
            client.getWindow().setWindowedSize(job.width(), job.height());
        } catch (RuntimeException error) {
            return "could not apply the requested client settings: " + error;
        }
        LOGGER.info("mc-photo-capture: applied client settings for {} (fov {}, gamma {}, "
                        + "hud hidden {}, window {}x{})",
                job.frameId(), (int) Math.round(job.fovDegrees()), job.settings().gamma(),
                job.settings().hudHidden(), job.width(), job.height());
        return null;
    }

    /**
     * Runs on the server thread: put the identified player's feet where the plan asks.
     *
     * <p>Targeting by UUID matters because the integrated server may hold more than
     * one player entity; {@code getPlayerList().get(0)} is not guaranteed to be the
     * client the screenshots come from.
     */
    private static void teleportPlayer(MinecraftServer server, UUID playerId,
                                       PlanAudit.CaptureJob job) {
        if (playerId == null) {
            throw new IllegalStateException("no client player was available to teleport");
        }
        ServerPlayerEntity player = server.getPlayerManager().getPlayer(playerId);
        if (player == null) {
            throw new IllegalStateException("no server player matched " + playerId);
        }
        ServerWorld overworld = server.getOverworld();
        if (overworld == null) {
            throw new IllegalStateException("the integrated server has no overworld");
        }
        // Absolute position/yaw/pitch on the integrated server: the plan's
        // camera.position is the player's feet, so no eye-height adjustment is made
        // here and none may be added without redefining the ground-truth contract.
        TeleportTarget target = new TeleportTarget(overworld,
                new Vec3d(job.x(), job.y(), job.z()), Vec3d.ZERO,
                (float) job.yawDegrees(), (float) job.pitchDegrees(),
                Set.of(), TeleportTarget.NO_OP);
        maintainEntity(player);
        player.teleportTo(target);
        // teleportTo preserves the requested feet/yaw/pitch; re-zero motion after
        // its internal transition so inherited velocity/fall state cannot survive.
        maintainEntity(player);
    }

    /**
     * Runs on the server thread: apply the requested time and weather.
     *
     * <p>The staged copy's daylight cycle is disabled before the capture time is
     * set. Otherwise {@code timeOfDay} advances once per tick during the settle
     * window, so the value sampled when the settings were applied would no longer
     * describe the screenshot. Freezing the cycle is the smaller robust fix: the
     * time stays exactly at the requested value through settle and capture, and the
     * finalizer can require an exact match. This mutates only the throwaway staged
     * copy the plan is armed against, never a source world; the plan is consumed
     * afterwards, so the rule is deliberately left in place.
     */
    private static void applyWorldSettings(MinecraftServer server, PlanAudit.CaptureJob job) {
        ServerWorld overworld = server.getOverworld();
        if (overworld == null) {
            throw new IllegalStateException("the integrated server has no overworld");
        }
        overworld.getGameRules().get(GameRules.DO_DAYLIGHT_CYCLE).set(false, server);
        overworld.setTimeOfDay(job.settings().timeOfDay());
        applyWeather(overworld, job.settings().weather());
    }

    /** Runs on the server thread: sample what the world now actually reports. */
    private static void readWorldSettings(MinecraftServer server, CaptureController controller) {
        ServerWorld overworld = server.getOverworld();
        if (overworld == null) {
            throw new IllegalStateException("the integrated server has no overworld");
        }
        controller.serverActuals = new ServerActuals(overworld.getTimeOfDay(),
                overworld.isRaining(), overworld.isThundering(),
                !overworld.getGameRules().getBoolean(GameRules.DO_DAYLIGHT_CYCLE));
    }

    private record ServerActuals(long timeOfDay, boolean raining, boolean thundering,
                                 boolean daylightCycleFrozen) {

        /** The weather label the plan declared: thunder, rain, or clear. */
        String weather() {
            if (thundering) {
                return "thunder";
            }
            return raining ? "rain" : "clear";
        }
    }

    /**
     * Runs on the server thread: set the requested weather and make it visible now.
     *
     * <p>{@code setWeather} only writes the world properties; the rain and thunder
     * gradients that {@code isRaining()}/{@code isThundering()} read ramp at 0.01 per
     * tick, so a sample taken right after the call would still report the previous
     * weather. Setting the gradients directly makes the sampled actual describe the
     * captured frame instead of a value that is still catching up.
     */
    private static void applyWeather(ServerWorld world, String weather) {
        switch (weather) {
            case "rain" -> {
                world.setWeather(0, 6000, true, false);
                world.setRainGradient(1.0F);
                world.setThunderGradient(0.0F);
            }
            case "thunder" -> {
                world.setWeather(0, 6000, true, true);
                world.setRainGradient(1.0F);
                world.setThunderGradient(1.0F);
            }
            default -> {
                world.setWeather(6000, 0, false, false);
                world.setRainGradient(0.0F);
                world.setThunderGradient(0.0F);
            }
        }
    }

    private static double[] clientFeet(ClientPlayerEntity player) {
        return new double[] {player.getX(), player.getY(), player.getZ()};
    }

    private static double[] clientEye(ClientPlayerEntity player) {
        return new double[] {player.getX(), player.getEyeY(), player.getZ()};
    }

    private static MoveCheck checkPosition(MinecraftClient client, PlanAudit.CaptureJob job) {
        ClientPlayerEntity player = client.player;
        if (player == null) {
            return new MoveCheck(false, "no client player is loaded", Double.NaN, Double.NaN,
                    Double.NaN, Double.NaN);
        }
        double[] requestedFeet = {job.x(), job.y(), job.z()};
        double[] requestedEye = CapturePlumbing.eyeFromFeet(requestedFeet, job.eyeHeight());
        double[] actualFeet = clientFeet(player);
        double[] actualEye = clientEye(player);
        double yawDelta = PlanAudit.angleDeltaDegrees(player.getYaw(), job.yawDegrees());
        double pitchDelta = Math.abs(player.getPitch() - job.pitchDegrees());
        double feetError = CapturePlumbing.distance(actualFeet, requestedFeet);
        double eyeError = CapturePlumbing.distance(actualEye, requestedEye);
        boolean within = CapturePlumbing.withinTolerance(actualFeet, actualEye,
                requestedFeet, requestedEye, MOVE_TOLERANCE)
                && PlanAudit.headingWithinTolerance(player.getYaw(), job.yawDegrees(),
                        HEADING_TOLERANCE)
                && pitchDelta <= HEADING_TOLERANCE;
        String detail = String.format(java.util.Locale.ROOT,
                "feet %.3f/%.3f/%.3f -> %.3f/%.3f/%.3f (%.3f), eye %.3f/%.3f/%.3f (%.3f), "
                        + "yaw delta %.3f, pitch delta %.3f",
                requestedFeet[0], requestedFeet[1], requestedFeet[2],
                actualFeet[0], actualFeet[1], actualFeet[2],
                feetError,
                actualEye[0], actualEye[1], actualEye[2],
                eyeError,
                yawDelta, pitchDelta);
        return new MoveCheck(within, detail, feetError, eyeError, yawDelta, pitchDelta);
    }

    private record MoveCheck(boolean ok, String detail, double feetError, double eyeError,
                             double yawError, double pitchError) {
    }

    /**
     * Measure the live capture guard immediately before a screenshot.
     *
     * <p>This is the DS-20 fix: the earlier convergence check runs before the window
     * resize and the settle window, so it cannot see a resize-induced yaw drift or a
     * pause menu that opened during settle. The measurement is repeated here, at the
     * last moment where it can still stop the capture.
     *
     * <p>Window focus is read for evidence only. Interactive capture runs on a
     * hidden, unfocused Windows Desktop 2, so {@code focused=false} is normal and
     * must not stop a clean, unpaused, pose-converged frame; {@link
     * CaptureGuard#ready()} intentionally ignores it.
     */
    private CaptureGuard measureGuard(MinecraftClient client, PlanAudit.CaptureJob job) {
        MoveCheck pose = checkPosition(client, job);
        recordPose(pose);
        Screen screen = client.currentScreen;
        boolean focused = isWindowFocused(client);
        boolean paused = client.isPaused();
        if (screen == null) {
            return CaptureGuard.noScreen(paused, focused, pose.ok(), pose.detail());
        }
        String screenType = screen.getClass().getName();
        if (screen.getClass() == GameMenuScreen.class) {
            return CaptureGuard.ordinaryGameMenu(paused, focused, pose.ok(), pose.detail());
        }
        return CaptureGuard.unexpectedScreen(screenType, paused, focused, pose.ok(),
                pose.detail());
    }

    private void recordPose(MoveCheck pose) {
        lastPoseCheck = pose;
        lastConvergenceDetail = pose.detail();
    }

    /**
     * Close only the ordinary in-game pause menu; never any other screen.
     *
     * <p>The driver only asks for this after classifying the screen as
     * {@code GAME_MENU}, but the screen can change between the measurement and this
     * call, so the check is repeated here. Anything else is refused rather than
     * dismissed, because closing a screen the capture did not expect could discard
     * user state (a death screen, a world-load screen, a confirmation).
     *
     * @return false when the job was failed and no later effect may run
     */
    private boolean closeOrdinaryMenu(MinecraftClient client, PlanAudit.CaptureJob job) {
        Screen screen = client.currentScreen;
        if (screen == null) {
            return true;
        }
        if (screen.getClass() != GameMenuScreen.class) {
            failJob(job, "refusing to close the unexpected screen "
                    + screen.getClass().getName() + " before capture");
            return false;
        }
        try {
            client.setScreen(null);
        } catch (RuntimeException error) {
            failJob(job, "could not close the ordinary pause menu before capture: " + error);
            return false;
        }
        menuCloses++;
        LOGGER.info("mc-photo-capture: closed the ordinary pause menu for {}; a fresh "
                + "settle and pose re-assertion are required", job.frameId());
        // No focus call: the window is normally unfocused on the hidden capture
        // desktop, focus is telemetry, and the guard never requires it.
        return true;
    }

    /**
     * Read-only focus probe for the frame evidence; never a capture gate.
     *
     * <p>A missing GLFW/context reports "not focused" rather than throwing, because
     * an unfocused window is a normal state that must not disturb the capture.
     */
    private static boolean isWindowFocused(MinecraftClient client) {
        try {
            return client.isWindowFocused();
        } catch (RuntimeException | LinkageError error) {
            LOGGER.warn("mc-photo-capture: could not read the window focus state", error);
            return false;
        }
    }

    private void capture(MinecraftClient client) {
        PlanAudit.CaptureJob job = plan.jobs().get(jobIndex);
        try {
            if (runtimeProvenance != null) {
                pendingCaptureSnapshot = RuntimeRenderProvenance.captureSnapshot(
                        client, job.frameId(), RuntimeRenderProvenance.activeProfiles(client));
            }
            File target = new File(client.runDirectory, "screenshots/" + job.filename());
            if (CapturePlumbing.removeStaleScreenshot(target.toPath())) {
                LOGGER.info("mc-photo-capture: removed stale screenshot {}", target);
            }
            screenshotFile = target;
            screenshotDigest = null;
            screenshotCallbackDone = false;
            screenshotCallbackError = null;
            fileWaitTicks = 0;
            ScreenshotRecorder.saveScreenshot(client.runDirectory, job.filename(),
                    client.getFramebuffer(),
                    message -> {
                        // The recorder reports a Text (usually null on success).
                        screenshotCallbackError = message == null ? null : message.getString();
                        screenshotCallbackDone = true;
                    });
            // The writer is asynchronous, but starting it is a synchronous driver
            // step: tell the driver now so the next tick waits for the file instead
            // of invoking capture again.
            phase = Phase.WAIT_FILE;
            jobDriver.onCaptureCompleted();
        } catch (IOException | RuntimeException error) {
            failJob(job, "screenshot call failed: " + error);
        }
    }

    private void awaitFile(MinecraftClient client) {
        PlanAudit.CaptureJob job = plan.jobs().get(jobIndex);
        File file = screenshotFile;
        if (CapturePlumbing.isScreenshotAccepted(file == null ? null : file.toPath(),
                screenshotCallbackDone)) {
            // Hash the finished file before reporting success; the finalizer
            // verifies this digest against the source and the copied screenshot.
            try {
                screenshotDigest = StagingMarker.fileSha256(file.toPath());
            } catch (IOException error) {
                failJob(job, "could not hash the finished screenshot " + file + ": " + error);
                return;
            }
            applyEffects(client, "screenshot", jobDriver.onScreenshotReady());
            recordSuccess(client, job, file);
            advance();
            return;
        }
        if (++fileWaitTicks > FILE_WAIT_TICKS) {
            String reason = "screenshot was not completed within " + FILE_WAIT_TICKS + " ticks";
            if (!screenshotCallbackDone) {
                reason += " (the recorder callback never signalled)";
            }
            if (screenshotCallbackError != null) {
                reason += ": " + screenshotCallbackError;
            }
            failJob(job, reason);
        }
    }

    private void advance() {
        jobIndex++;
        pendingTask = null;
        positionObservation.cancel();
        serverWaitTicks = 0;
        serverActuals = null;
        jobDriver = null;
        phase = jobIndex >= plan.jobs().size() ? Phase.FINISH : Phase.START_APPLY;
    }

    /**
     * Export ground truth, restore the client, and disarm.
     *
     * <p>Ordering is deliberate. The frame results and the run status are written
     * before the export is attempted, so if the export or the process fails, the
     * durable record already says {@code ground_truth_exported: false} and the
     * finalizer refuses the scene. Every terminal path - success, export failure,
     * abort - reaches {@link #terminate}, which restores the client and disarms.
     */
    private void finish(MinecraftClient client) {
        if (finished) {
            return;
        }
        MinecraftServer server = client.getServer();
        if (groundTruthJson == null) {
            if (server == null) {
                failGroundTruth(client, "the integrated server is gone; ground truth "
                        + "was never exported");
                terminate(client);
                return;
            }
            if (groundTruthTask == null) {
                // The plan is complete and cannot be re-run after disarm, so the
                // remaining budget starts now, at submission.
                groundTruthWaitTicks = 0;
                Ticket ticket = new Ticket(JobFlow.Action.SETTINGS, "GROUND_TRUTH");
                groundTruthTask = ticket;
                scheduleTask(server, ticket,
                        () -> groundTruthJson = pretty(GroundTruthWriter.export(server, plan)));
                return;
            }
            if (!groundTruthTask.completed) {
                if (++groundTruthWaitTicks <= GROUND_TRUTH_WAIT_TICKS) {
                    return;
                }
                failGroundTruth(client, "ground truth export did not finish within "
                        + GROUND_TRUTH_WAIT_TICKS + " ticks");
                terminate(client);
                return;
            }
            if (groundTruthTask.error != null) {
                failGroundTruth(client, "ground truth export failed on the server thread: "
                        + groundTruthTask.error);
                terminate(client);
                return;
            }
            if (groundTruthJson == null) {
                failGroundTruth(client, "ground truth export produced no payload");
                terminate(client);
                return;
            }
        }
        groundTruthExported = writeGroundTruthPayload(client, groundTruthJson, false);
        if (!groundTruthExported) {
            runFailed = true;
            groundTruthErrors.add("could not write exported ground_truth.json payload");
        }
        writeResults(client);
        terminate(client);
    }

    /** Restore the client options and disarm; safe to call once per run. */
    private void terminate(MinecraftClient client) {
        if (finished) {
            return;
        }
        finished = true;
        positionObservation.cancel();
        poseSession.stop();
        restoreOriginals(client);
        disarm();
        LOGGER.info("mc-photo-capture: plan {} finished with {} frame results ({}); "
                        + "ground truth {}",
                plan.planId(), results.size(), runFailed ? "FAILED" : "ok",
                groundTruthExported ? "exported" : "missing");
        phase = Phase.DONE;
    }

    /**
     * Leave a durable, unambiguous failure state.
     *
     * <p>The finalizer refuses a scene without a real ground-truth export, so a
     * failed export must not look like a successful run: the frame results and run
     * status are written first with {@code ground_truth_exported: false}, and a
     * sentinel payload replaces any partial export. The scene cannot be reported as
     * a success even if the failure result itself is the last thing written.
     */
    private void failGroundTruth(MinecraftClient client, String reason) {
        runFailed = true;
        groundTruthErrors.add(reason);
        groundTruthJson = null;
        groundTruthExported = false;
        LOGGER.error("mc-photo-capture: ground truth failure: {}", reason);
        writeResults(client);
        JsonArray errors = new JsonArray();
        for (String error : groundTruthErrors) {
            errors.add(error);
        }
        JsonObject sentinel = new JsonObject();
        sentinel.addProperty("schema_version", PlanAudit.SUPPORTED_SCHEMA_VERSION);
        sentinel.addProperty("mod_contract_version", PlanAudit.SUPPORTED_MOD_CONTRACT_VERSION);
        sentinel.addProperty("ground_truth_exported", false);
        sentinel.add("errors", errors);
        writeGroundTruthPayload(client, pretty(sentinel), true);
        writeResults(client);
    }

    private void abort(MinecraftClient client, String reason) {
        if (finished) {
            return;
        }
        runFailed = true;
        LOGGER.error("mc-photo-capture: aborting plan {}: {}",
                plan == null ? "?" : plan.planId(), reason);
        for (PlanAudit.CaptureJob job : plan.jobs()) {
            if (!recordedFrames.contains(job.frameId())) {
                recordFailure(job, reason);
            }
        }
        writeResults(client);
        failGroundTruth(client, "the plan aborted before a ground-truth export: " + reason);
        terminate(client);
    }

    private void recordSuccess(MinecraftClient client, PlanAudit.CaptureJob job, File file) {
        JsonObject actual = new JsonObject();
        ClientPlayerEntity player = client.player;
        if (player != null) {
            actual.add("feet_position", triple(clientFeet(player)));
            actual.add("eye_position", triple(clientEye(player)));
            actual.addProperty("yaw_degrees", player.getYaw());
            actual.addProperty("pitch_degrees", player.getPitch());
        }
        actual.addProperty("fov_degrees", client.options.getFov().getValue());
        actual.addProperty("gamma", client.options.getGamma().getValue());
        actual.addProperty("hud_hidden", client.options.hudHidden);
        actual.addProperty("window_width", client.getWindow().getWidth());
        actual.addProperty("window_height", client.getWindow().getHeight());
        actual.addProperty("framebuffer_width", client.getFramebuffer().textureWidth);
        actual.addProperty("framebuffer_height", client.getFramebuffer().textureHeight);
        ServerActuals world = serverActuals;
        if (world != null) {
            actual.addProperty("time_of_day", world.timeOfDay());
            actual.addProperty("raining", world.raining());
            actual.addProperty("thundering", world.thundering());
            actual.addProperty("weather", world.weather());
            actual.addProperty("daylight_cycle_frozen", world.daylightCycleFrozen());
        }
        actual.addProperty("screenshot_bytes", file.length());
        actual.addProperty("screenshot_sha256", screenshotDigest);
        actual.addProperty("staging_token", plan.stagingToken());

        // The capture guard measurement is recorded verbatim, so a contaminated frame
        // can be identified from the result alone: the screen that was open (if any),
        // whether the client was paused, whether the window was focused, and the exact
        // pose error that was accepted immediately before the screenshot.
        CaptureGuard guard = lastGuard;
        if (guard != null) {
            actual.addProperty("screen_open", guard.screenOpen());
            if (guard.screenType() == null) {
                actual.add("screen_type", JsonNull.INSTANCE);
            } else {
                actual.addProperty("screen_type", guard.screenType());
            }
            actual.addProperty("ordinary_game_menu",
                    guard.screen() == CaptureGuard.ScreenKind.GAME_MENU);
            actual.addProperty("paused", guard.paused());
            actual.addProperty("focused", guard.focused());
            actual.addProperty("pose_converged", guard.poseConverged());
            actual.addProperty("pose_guard_detail", guard.poseDetail());
        }
        MoveCheck pose = lastPoseCheck;
        if (pose != null) {
            actual.addProperty("pose_error_feet", pose.feetError());
            actual.addProperty("pose_error_eye", pose.eyeError());
            actual.addProperty("pose_error_yaw_degrees", pose.yawError());
            actual.addProperty("pose_error_pitch_degrees", pose.pitchError());
        }
        if (jobDriver != null) {
            actual.addProperty("pose_attempts", jobDriver.positionAttempts());
            actual.addProperty("pose_reassertions", jobDriver.reassertAttempts());
            actual.addProperty("guard_attempts", jobDriver.guardAttempts());
            actual.addProperty("recovery_cycles", jobDriver.recoveries());
        }
        actual.addProperty("menu_closes", menuCloses);

        // The request and the observation are recorded side by side, for every
        // setting, so the finalizer can refuse a capture whose actual state did not
        // honour the plan instead of trusting the request as if it were active.
        JsonObject requested = new JsonObject();
        requested.addProperty("fov_degrees", (int) Math.round(job.fovDegrees()));
        requested.addProperty("gamma", job.settings().gamma());
        requested.addProperty("hud_hidden", job.settings().hudHidden());
        requested.addProperty("window_width", job.width());
        requested.addProperty("window_height", job.height());
        requested.addProperty("framebuffer_width", job.width());
        requested.addProperty("framebuffer_height", job.height());
        requested.addProperty("time_of_day", job.settings().timeOfDay());
        requested.addProperty("weather", job.settings().weather());
        // A request is never a setting: switching resource/shader packs at runtime
        // is not implemented, so both stay explicitly false until it is.
        requested.addProperty("resource_pack", job.settings().resourcePack());
        requested.addProperty("shader_pack", job.settings().shaderPack());
        actual.add("requested", requested);
        JsonArray orderedPacks = RuntimeRenderProvenance.activeProfiles(client);
        JsonArray packIds = new JsonArray();
        for (var pack : orderedPacks) {
            packIds.add(pack.isJsonObject() && pack.getAsJsonObject().has("id")
                    ? pack.getAsJsonObject().get("id") : JsonNull.INSTANCE);
        }
        actual.add("active_resource_packs", packIds);
        actual.addProperty("resource_pack_applied", false);
        actual.addProperty("shader_pack_applied", false);
        if (runtimeProvenance != null) {
            runtimeProvenance.add("resource_packs", orderedPacks.deepCopy());
            // The submission snapshot is immutable evidence of what was sampled
            // before the asynchronous recorder call. Completion-time observations
            // are separate and cannot rewrite that snapshot or its digest.
            JsonObject snapshot = pendingCaptureSnapshot == null
                    ? new JsonObject() : pendingCaptureSnapshot.deepCopy();
            var savedPacks = snapshot.get("ordered_resource_packs");
            boolean resourceOrderStable = savedPacks != null
                    && orderedPacks.toString().equals(savedPacks.toString());
            String snapshotDigest = RuntimeRenderProvenance.digest(snapshot);
            JsonObject frame = new JsonObject();
            frame.addProperty("frame_id", job.frameId());
            frame.addProperty("filename", job.filename());
            frame.addProperty("screenshot_sha256", screenshotDigest);
            frame.addProperty("screenshot_bytes", file.length());
            frame.addProperty("snapshot_sha256", snapshotDigest);
            frame.addProperty("capture_epoch_status", snapshot.has("capture_epoch_status")
                    ? snapshot.get("capture_epoch_status").getAsString() : "capture_epoch_unbound");
            frame.addProperty("completion_resource_order_stable", resourceOrderStable);
            frame.addProperty("completion_resource_order_status", resourceOrderStable
                    ? "same_profile_order_observed_after_encode"
                    : "profile_order_changed_or_unavailable_after_encode");
            frame.add("snapshot", snapshot);
            runtimeProvenance.getAsJsonArray("frames").add(frame);
            JsonObject sealed = RuntimeRenderProvenance.seal(runtimeProvenance);
            String serialized = RuntimeRenderProvenance.canonical(sealed);
            Path sidecar = outputDirectory(client).resolve("runtime-render-provenance.json");
            try {
                CapturePlumbing.writeJsonAtomic(sidecar, serialized);
            } catch (IOException error) {
                failJob(job, "could not persist runtime/render provenance: " + error.getMessage());
                return;
            }
            String sidecarDigest = RuntimeRenderProvenance.sha256(
                    serialized.getBytes(StandardCharsets.UTF_8));
            JsonObject binding = new JsonObject();
            binding.addProperty("filename", "runtime-render-provenance.json");
            binding.addProperty("attempt_nonce", provenanceExpectation.get("attempt_nonce").getAsString());
            binding.addProperty("snapshot_sha256", snapshotDigest);
            binding.addProperty("capture_epoch_status", snapshot.has("capture_epoch_status")
                    ? snapshot.get("capture_epoch_status").getAsString() : "capture_epoch_unbound");
            actual.add("runtime_render_provenance", binding);
        }
        pushResult(job, true, "screenshots/" + job.filename(), actual, List.of());
        pendingCaptureSnapshot = null;
    }

    private static JsonArray triple(double[] values) {
        JsonArray array = new JsonArray();
        for (double value : values) {
            array.add(value);
        }
        return array;
    }

    private static JsonArray intTriple(int[] values) {
        JsonArray array = new JsonArray();
        for (int value : values) {
            array.add(value);
        }
        return array;
    }

    private void recordFailure(PlanAudit.CaptureJob job, String reason) {
        runFailed = true;
        pushResult(job, false, null, new JsonObject(), List.of(reason));
    }

    private void pushResult(PlanAudit.CaptureJob job, boolean captured, String screenshot,
                            JsonObject actual, List<String> errors) {
        if (!recordedFrames.add(job.frameId())) {
            return;
        }
        JsonObject result = new JsonObject();
        result.addProperty("schema_version", PlanAudit.SUPPORTED_SCHEMA_VERSION);
        result.addProperty("frame_id", job.frameId());
        result.addProperty("filename", job.filename());
        result.addProperty("captured", captured);
        if (screenshot == null) {
            result.add("screenshot", JsonNull.INSTANCE);
        } else {
            result.addProperty("screenshot", screenshot);
        }
        result.addProperty("settled_ticks", job.settleTicks());
        result.add("actual", actual);
        JsonArray errorList = new JsonArray();
        for (String error : errors) {
            errorList.add(error);
        }
        result.add("errors", errorList);
        results.add(result);
    }

    private Path outputDirectory(MinecraftClient client) {
        return client.runDirectory.toPath().resolve(OUTPUT_DIRECTORY).resolve(plan.planId());
    }

    private void writeResults(MinecraftClient client) {
        try {
            Path output = outputDirectory(client);
            Files.createDirectories(output);
            JsonArray array = new JsonArray();
            for (JsonObject result : results) {
                array.add(result);
            }
            // results.json stays strictly a list of frame results so a reader that
            // maps every entry to a planned job cannot trip over a status record.
            CapturePlumbing.writeJsonAtomic(output.resolve("results.json"), pretty(array));
            writeRunStatus(client);
        } catch (IOException error) {
            LOGGER.error("mc-photo-capture: could not write results.json", error);
        }
    }

    /**
     * A durable, separate record of whether ground truth was exported.
     *
     * <p>{@code ground_truth_exported} is true only when an export really succeeded:
     * a failure sentinel clears {@code groundTruthJson}, so the flag can never be
     * set by a partial or empty payload.
     */
    private void writeRunStatus(MinecraftClient client) {
        boolean allFramesCaptured = !results.isEmpty() && results.size() == plan.jobs().size();
        for (JsonObject result : results) {
            if (!result.has("captured") || !result.get("captured").getAsBoolean()) {
                allFramesCaptured = false;
                break;
            }
        }
        JsonObject status = new JsonObject();
        status.addProperty("schema_version", PlanAudit.SUPPORTED_SCHEMA_VERSION);
        status.addProperty("mod_contract_version", PlanAudit.SUPPORTED_MOD_CONTRACT_VERSION);
        status.addProperty("plan_id", plan.planId());
        status.addProperty("staging_token", plan.stagingToken());
        status.addProperty("run_failed", runFailed);
        status.addProperty("all_frames_captured", allFramesCaptured);
        status.addProperty("ground_truth_exported", groundTruthExported);
        status.addProperty("context_export_only", plan.contextExportOnly());
        status.addProperty("runtime_render_provenance_required", provenanceExpectation != null);
        if (provenanceExpectation != null) {
            JsonObject provenanceRef = new JsonObject();
            provenanceRef.addProperty("filename", "runtime-render-provenance.json");
            provenanceRef.add("attempt_nonce", provenanceExpectation.get("attempt_nonce"));
            provenanceRef.add("plan_sha256", provenanceExpectation.get("plan_sha256"));
            provenanceRef.add("staging_token", provenanceExpectation.get("staging_token"));
            provenanceRef.add("expectation_sha256", provenanceExpectation.get("expectation_sha256"));
            provenanceRef.add("capture_mod_source_sha256", provenanceExpectation.get("capture_mod_source_sha256"));
            provenanceRef.add("capture_mod_staged_sha256", provenanceExpectation.get("capture_mod_staged_sha256"));
            provenanceRef.addProperty("mode", provenanceExpectation.get("mode").getAsString());
            Path sidecar = outputDirectory(client).resolve("runtime-render-provenance.json");
            if (Files.isRegularFile(sidecar)) {
                try {
                    if (Files.size(sidecar) > 1_048_576L) {
                        throw new IOException("runtime/render sidecar exceeds byte cap");
                    }
                    byte[] bytes;
                    try (var input = Files.newInputStream(sidecar)) {
                        bytes = input.readNBytes(1_048_577);
                    }
                    if (bytes.length > 1_048_576) {
                        throw new IOException("runtime/render sidecar exceeds byte cap");
                    }
                    JsonObject saved = JsonParser.parseString(
                            new String(bytes, StandardCharsets.UTF_8)).getAsJsonObject();
                    provenanceRef.addProperty("sha256", RuntimeRenderProvenance.sha256(bytes));
                    provenanceRef.addProperty("bytes", bytes.length);
                    provenanceRef.add("seal_sha256", saved.get("seal_sha256"));
                } catch (Exception error) {
                    provenanceRef.addProperty("read_status", "unavailable");
                }
            } else {
                provenanceRef.addProperty("read_status", "missing");
            }
            provenanceRef.addProperty("certifiable", false);
            provenanceRef.addProperty("status", "proxy_only_capture_epoch_or_runtime_binding_unavailable");
            provenanceRef.addProperty("certification_status", "unavailable");
            provenanceRef.addProperty("certification_refusal",
                    "required shader/resource/program/readback epoch adapters are unavailable");
            status.add("runtime_render_provenance", provenanceRef);
        }
        if (plan.contextExportOnly()) {
            JsonObject context = new JsonObject();
            context.addProperty("scene_id", plan.sceneId());
            context.addProperty("world_identity", plan.worldIdentity());
            JsonObject region = new JsonObject();
            region.add("min", intTriple(plan.minimum()));
            region.add("max", intTriple(plan.maximum()));
            context.add("region", region);
            context.addProperty("camera_pose_applied", false);
            context.addProperty("result_count", results.size());
            status.add("context_export", context);
        }
        JsonArray errors = new JsonArray();
        for (String error : groundTruthErrors) {
            errors.add(error);
        }
        status.add("errors", errors);
        try {
            CapturePlumbing.writeJsonAtomic(outputDirectory(client).resolve("run_status.json"),
                    pretty(status));
        } catch (IOException error) {
            LOGGER.error("mc-photo-capture: could not write run_status.json", error);
        }
    }

    /** Write the sentinel or the exported ground truth atomically. */
    private boolean writeGroundTruthPayload(MinecraftClient client, String payload,
                                            boolean failureSentinel) {
        if (payload == null) {
            groundTruthExported = false;
            return false;
        }
        try {
            CapturePlumbing.writeJsonAtomic(outputDirectory(client).resolve("ground_truth.json"),
                    payload);
            groundTruthExported = CapturePlumbing.isGroundTruthExported(true, failureSentinel);
            return true;
        } catch (IOException error) {
            groundTruthExported = CapturePlumbing.isGroundTruthExported(false, failureSentinel);
            LOGGER.error("mc-photo-capture: could not write ground_truth.json", error);
            return false;
        }
    }

    private void recordOriginals(MinecraftClient client) {
        originalFov = client.options.getFov().getValue();
        originalGamma = client.options.getGamma().getValue();
        originalHudHidden = client.options.hudHidden;
        originalWindowWidth = client.getWindow().getWidth();
        originalWindowHeight = client.getWindow().getHeight();
    }

    private void restoreOriginals(MinecraftClient client) {
        if (originalFov == null) {
            return;
        }
        try {
            client.options.getFov().setValue(originalFov);
            client.options.getGamma().setValue(originalGamma);
            client.options.hudHidden = originalHudHidden;
            if (originalWindowWidth > 0 && originalWindowHeight > 0) {
                client.getWindow().setWindowedSize(originalWindowWidth, originalWindowHeight);
            }
            client.options.write();
        } catch (RuntimeException error) {
            LOGGER.error("mc-photo-capture: could not restore client options", error);
        }
    }

    private void reject(String reason) {
        LOGGER.error("mc-photo-capture: {}", reason);
        try {
            Files.createDirectories(configDir);
            JsonObject error = new JsonObject();
            error.addProperty("schema_version", PlanAudit.SUPPORTED_SCHEMA_VERSION);
            error.addProperty("error", reason);
            error.addProperty("armed_path", String.valueOf(armedPath));
            CapturePlumbing.writeJsonAtomic(configDir.resolve("last-error.json"), pretty(error));
        } catch (IOException writeError) {
            LOGGER.error("mc-photo-capture: could not record the rejection", writeError);
        }
        disarm("rejected");
        plan = null;
        phase = Phase.DONE;
    }

    private void disarm() {
        disarm("consumed");
    }

    /** Same-directory atomic rename, then best-effort delete; never a plain unlink. */
    private void disarm(String reason) {
        try {
            CapturePlumbing.disarmAtomic(armedPath, reason);
        } catch (IOException error) {
            LOGGER.error("mc-photo-capture: could not atomically disarm armed.json", error);
        }
    }

    private static String requireText(JsonObject object, String field) throws PlanAudit.PlanRejected {
        if (!object.has(field) || object.get(field).isJsonNull()
                || object.get(field).getAsString().isBlank()) {
            throw new PlanAudit.PlanRejected("armed.json is missing " + field);
        }
        return object.get(field).getAsString();
    }

    private static String pretty(Object payload) {
        return new GsonBuilder().setPrettyPrinting().create().toJson(payload) + System.lineSeparator();
    }
}
