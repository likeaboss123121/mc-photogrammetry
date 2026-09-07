from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GuiMaskConfig:
    """Normalized regions for the vanilla HUD at an arbitrary resolution."""

    hotbar_height: float = 0.12
    crosshair_radius: float = 0.025
    chat_width: float = 0.34
    chat_height: float = 0.22
    mask_status_bars: bool = True
    mask_debug_area: bool = False
    held_item_width: float = 0.42
    held_item_height: float = 0.45
    """First-person held item, drawn over the lower-right corner of the world."""


def create_gui_mask(height: int, width: int, config: GuiMaskConfig | None = None) -> np.ndarray:
    """Return True for pixels excluded from vision processing."""
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    settings = config or GuiMaskConfig()
    mask = np.zeros((height, width), dtype=bool)

    hotbar_top = max(0, int(height * (1 - settings.hotbar_height)))
    mask[hotbar_top:, :] = True

    center_x, center_y = width // 2, height // 2
    radius = max(1, int(min(height, width) * settings.crosshair_radius))
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    mask[y0:y1, x0:x1] = True

    chat_top = max(0, int(height * (1 - settings.hotbar_height - settings.chat_height)))
    chat_right = min(width, int(width * settings.chat_width))
    mask[chat_top:hotbar_top, :chat_right] = True

    if settings.held_item_width > 0 and settings.held_item_height > 0:
        item_top = max(0, int(height * (1 - settings.held_item_height)))
        item_left = max(0, int(width * (1 - settings.held_item_width)))
        mask[item_top:, item_left:] = True

    if settings.mask_status_bars:
        mask[: max(1, int(height * 0.1)), : int(width * 0.55)] = True

    if settings.mask_debug_area:
        mask[: max(1, int(height * 0.08)), :] = True

    return mask


def apply_manual_mask(mask: np.ndarray, *, x0: int, y0: int, x1: int, y1: int, value: bool) -> np.ndarray:
    """Return a copy with a user rectangle added to or removed from the mask."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    left, right = sorted((max(0, x0), min(mask.shape[1], x1)))
    top, bottom = sorted((max(0, y0), min(mask.shape[0], y1)))
    updated = mask.copy()
    updated[top:bottom, left:right] = value
    return updated
