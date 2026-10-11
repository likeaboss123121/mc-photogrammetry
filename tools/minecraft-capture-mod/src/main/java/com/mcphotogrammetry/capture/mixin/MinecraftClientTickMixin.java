package com.mcphotogrammetry.capture.mixin;

import com.mcphotogrammetry.capture.McPhotoCaptureClient;

import net.minecraft.client.MinecraftClient;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Runs the capture controller once per client tick, after vanilla has processed
 * the tick. This is the only hook the mod installs, so no Fabric API module is
 * required at runtime.
 */
@Mixin(MinecraftClient.class)
public abstract class MinecraftClientTickMixin {

    @Inject(method = "tick", at = @At("HEAD"))
    private void mcPhotoCapture$beforeClientTick(CallbackInfo info) {
        McPhotoCaptureClient.onBeforeClientTick((MinecraftClient) (Object) this);
    }

    @Inject(method = "tick", at = @At("TAIL"))
    private void mcPhotoCapture$afterClientTick(CallbackInfo info) {
        McPhotoCaptureClient.onClientTick((MinecraftClient) (Object) this);
    }
}
