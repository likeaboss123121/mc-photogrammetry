from __future__ import annotations

from pathlib import Path


class LitematicValidationError(ValueError):
    """Raised when a litematic does not have the expected structure."""


def validate_litematic(path: str | Path) -> dict[str, object]:
    """Load and validate the structural invariants needed by this exporter."""
    try:
        import nbtlib
    except ImportError as error:
        raise LitematicValidationError("litematic validation requires nbtlib") from error

    source = Path(path)
    try:
        document = nbtlib.load(source, gzipped=True)
    except (OSError, ValueError) as error:
        raise LitematicValidationError(f"could not read litematic {source}: {error}") from error
    root = document
    for key in ("Version", "SubVersion", "MinecraftDataVersion", "Metadata", "Regions"):
        if key not in root:
            raise LitematicValidationError(f"missing root field: {key}")
    regions = root["Regions"]
    if len(regions) != 1:
        raise LitematicValidationError("this validator expects exactly one region")
    region = next(iter(regions.values()))
    for key in ("Position", "Size", "BlockStatePalette", "BlockStates"):
        if key not in region:
            raise LitematicValidationError(f"missing region field: {key}")

    size = region["Size"]
    dimensions = tuple(int(size[axis]) for axis in ("x", "y", "z"))
    if any(dimension <= 0 for dimension in dimensions):
        raise LitematicValidationError("region dimensions must be positive")
    position = region["Position"]
    if tuple(int(position[axis]) for axis in ("x", "y", "z")) != (0, 0, 0):
        raise LitematicValidationError("single-region exports must use a zero local position")
    palette = region["BlockStatePalette"]
    if not palette:
        raise LitematicValidationError("block-state palette cannot be empty")
    if "Name" not in palette[0] or str(palette[0]["Name"]) != "minecraft:air":
        raise LitematicValidationError("palette index zero must be minecraft:air")

    block_count = dimensions[0] * dimensions[1] * dimensions[2]
    bits = max(2, (len(palette) - 1).bit_length())
    required_longs = (block_count * bits + 63) // 64
    if len(region["BlockStates"]) < required_longs:
        raise LitematicValidationError("block-state array is shorter than the region requires")

    return {
        "position": tuple(int(position[axis]) for axis in ("x", "y", "z")),
        "dimensions": dimensions,
        "palette_size": len(palette),
        "block_count": block_count,
        "bits_per_entry": bits,
    }
