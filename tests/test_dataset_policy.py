from dataset.quality import verify_license
from dataset.downloader import extension_for_url
from dataset.video_processor import VideoProcessor


def test_unknown_and_noncommercial_licenses_are_not_training_eligible():
    assert not verify_license("unknown").training_eligible
    assert not verify_license("CC-BY-NC").training_eligible
    assert verify_license("CC-BY 4.0").training_eligible


def test_media_extension_is_constrained_to_known_formats():
    assert extension_for_url("https://example.invalid/file.mp4", "video") == ".mp4"
    assert extension_for_url("https://example.invalid/no-extension", "image") == ".jpg"


def test_video_clip_duration_policy_is_explicit():
    assert VideoProcessor.SUPPORTED_CLIP_DURATIONS == [2, 4, 6, 8, 12]
