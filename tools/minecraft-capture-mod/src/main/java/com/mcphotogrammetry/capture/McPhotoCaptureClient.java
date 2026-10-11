package com.mcphotogrammetry.capture;

import net.fabricmc.api.ClientModInitializer;
import net.minecraft.client.MinecraftClient;
import net.minecraft.server.network.ServerPlayerEntity;

/**
 * Entry point of the capture mod. It only loads and validates the arming file;
 * every capture action happens from the client tick hook.
 */
public final class McPhotoCaptureClient implements ClientModInitializer {

    public static final String MOD_ID = "mc_photo_capture";

    private static CaptureController controller;

    @Override
    public void onInitializeClient() {
        controller = new CaptureController();
        controller.load();
    }

    public static void onClientTick(MinecraftClient client) {
        CaptureController current = controller;
        if (current != null) {
            current.tick(client);
        }
    }

    public static void onBeforeClientTick(MinecraftClient client) {
        CaptureController current = controller;
        if (current != null) {
            current.maintainClientPlayer(client);
        }
    }

    public static void onServerPlayerTick(ServerPlayerEntity player) {
        CaptureController current = controller;
        if (current != null) {
            current.maintainServerPlayer(player);
        }
    }
}
