"""Input discovery and frame manifests."""

from .media import MediaKind, MediaManifest, SourceFrame, discover_media
from .image import ImageDecodeError, describe_image, load_image
from .frames import FrameCandidate, hamming_distance, perceptual_hash, quality_score, select_keyframes
from .video import VideoDecodeError, iter_video_frames

__all__ = [
	"ImageDecodeError",
	"FrameCandidate",
	"MediaKind",
	"MediaManifest",
	"SourceFrame",
	"describe_image",
	"discover_media",
	"load_image",
	"VideoDecodeError",
	"hamming_distance",
	"perceptual_hash",
	"quality_score",
	"select_keyframes",
	"iter_video_frames",
]
