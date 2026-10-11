package com.mcphotogrammetry.capture.mixin;

import com.mcphotogrammetry.capture.McPhotoCaptureClient;

import net.minecraft.server.network.ServerPlayerEntity;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Keeps the validated staged capture player motionless before every server tick. */
@Mixin(ServerPlayerEntity.class)
public abstract class ServerPlayerTickMixin {

    @Inject(method = "tick", at = @At("HEAD"))
    private void mcPhotoCapture$beforeServerPlayerTick(CallbackInfo info) {
        McPhotoCaptureClient.onServerPlayerTick((ServerPlayerEntity) (Object) this);
    }
}
