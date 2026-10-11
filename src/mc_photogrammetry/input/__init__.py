"""Input discovery and frame manifests."""

from .media import MediaKind, MediaManifest, SourceFrame, discover_media, read_source_bytes
from .image import ImageDecodeError, describe_image, load_image, load_image_bytes
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
	"load_image_bytes",
	"read_source_bytes",
	"VideoDecodeError",
	"hamming_distance",
	"perceptual_hash",
	"quality_score",
	"select_keyframes",
	"iter_video_frames",
]
