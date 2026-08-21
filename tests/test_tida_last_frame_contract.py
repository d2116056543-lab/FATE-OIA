import numpy as np

from fate_oia.datasets.tida_clip_manifest import compare_last_frames


def test_identical_last_frame_passes_all_metrics():
    image = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
    metrics = compare_last_frames(image, image.copy())
    assert metrics["ssim"] == 1.0
    assert metrics["normalized_mae"] == 0.0
    assert metrics["phash_distance"] == 0
    assert metrics["pass"]
