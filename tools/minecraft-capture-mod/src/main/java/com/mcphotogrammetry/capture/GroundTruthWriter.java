package com.mcphotogrammetry.capture;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import net.minecraft.block.BlockState;
import net.minecraft.registry.Registries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.state.property.Property;
import net.minecraft.util.math.BlockPos;

import java.util.Map;

/**
 * Exports the exact {@code BlockState} id and properties for a plan's region from
 * the integrated server's world.
 *
 * <p>This is ground truth for evaluation only. It is written next to the capture
 * results in the staging game directory and is never an input to normal
 * reconstruction; the Python finalizer copies it into the dataset with an
 * explicit advanced-only label.
 */
final class GroundTruthWriter {

    private GroundTruthWriter() {
    }

    static JsonObject export(MinecraftServer server, PlanAudit.AuditedPlan plan) {
        ServerWorld world = server.getOverworld();
        int[] minimum = plan.minimum();
        int[] maximum = plan.maximum();
        JsonArray blocks = new JsonArray();
        for (int y = minimum[1]; y <= maximum[1]; y++) {
            for (int z = minimum[2]; z <= maximum[2]; z++) {
                for (int x = minimum[0]; x <= maximum[0]; x++) {
                    BlockState state = world.getBlockState(BlockPos.ORIGIN.add(x, y, z));
                    JsonObject entry = new JsonObject();
                    JsonArray position = new JsonArray();
                    position.add(x);
                    position.add(y);
                    position.add(z);
                    entry.add("pos", position);
                    entry.addProperty("name", Registries.BLOCK.getId(state.getBlock()).toString());
                    JsonObject properties = new JsonObject();
                    for (Map.Entry<Property<?>, Comparable<?>> property
                            : state.getEntries().entrySet()) {
                        properties.addProperty(property.getKey().getName(),
                                serialize(property.getKey(), property.getValue()));
                    }
                    entry.add("properties", properties);
                    blocks.add(entry);
                }
            }
        }

        JsonObject region = new JsonObject();
        region.add("min", triple(minimum));
        region.add("max", triple(maximum));

        JsonObject root = new JsonObject();
        root.addProperty("schema_version", PlanAudit.SUPPORTED_SCHEMA_VERSION);
        root.addProperty("mod_contract_version", PlanAudit.SUPPORTED_MOD_CONTRACT_VERSION);
        root.addProperty("scene_id", plan.sceneId());
        root.addProperty("world_identity", plan.worldIdentity());
        root.add("region", region);
        root.add("blocks", blocks);
        return root;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static String serialize(Property property, Comparable value) {
        // Property#name(Comparable) is exactly how the vanilla command serializer
        // renders a value ("north", "top", "true"), so downstream parsers see the
        // same strings a resource pack would.
        return property.name(value);
    }

    private static JsonArray triple(int[] values) {
        JsonArray array = new JsonArray();
        for (int value : values) {
            array.add(value);
        }
        return array;
    }
}
