from PIL import Image
import numpy as np

from fate_oia.transforms_video import SynchronizedVideoTransform


def test_all_frames_share_flip_and_geometry():
    frames = [Image.fromarray(np.full((24, 40, 3), i, dtype=np.uint8)) for i in range(15)]
    transform = SynchronizedVideoTransform(target_hw=(32, 48), context_hw=(16, 24), flip_probability=1.0)
    out = transform(frames, training=True, random_value=0.0)
    assert out["target_image"].shape == (3, 32, 48)
    assert out["context_images"].shape == (14, 3, 16, 24)
    assert out["meta"]["flipped"] is True
    assert len({tuple(m["normalized_geometry"]) for m in out["meta"]["frames"]}) == 1
