import torch
from torch import nn

from fate_oia.models.tida_object_tracker import TIDAFrozenPointTracker
from fate_oia.engine.extract_tida_object_tracks import select_seed_tracks


class _DummyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, video, *, grid_size, grid_query_frame, backward_tracking):
        self.calls += 1
        batch, frames, _, height, width = video.shape
        assert batch == 1
        assert video.min() >= 0 and video.max() <= 255
        points = grid_size * grid_size
        tracks = torch.zeros(1, frames, points, 2, device=video.device)
        tracks[..., 0] = width - 1
        tracks[..., 1] = height - 1
        visibility = torch.ones(1, frames, points, dtype=torch.bool, device=video.device)
        return tracks, visibility


def test_tracker_handles_batch_by_safe_per_sample_calls_and_normalizes_coordinates():
    predictor = _DummyPredictor()
    tracker = TIDAFrozenPointTracker(predictor, grid_size=3)
    normalized = torch.zeros(2, 5, 3, 12, 20)

    output = tracker(normalized)

    assert predictor.calls == 2
    assert output["object_tracks_xy"].shape == (2, 5, 9, 2)
    assert output["object_tracks_visibility"].shape == (2, 5, 9)
    assert torch.allclose(output["object_tracks_xy"], torch.ones(2, 5, 9, 2))
    assert not any(parameter.requires_grad for parameter in tracker.parameters())


def test_seed_track_selection_reuses_only_requested_names():
    payload = {
        "file_names": ["a.jpg", "b.jpg", "c.jpg"],
        "tracks_xy": torch.arange(18).reshape(3, 3, 1, 2).float(),
        "visibility": torch.ones(3, 3, 1, dtype=torch.bool),
    }
    names, tracks, visibility = select_seed_tracks(payload, {"b.jpg", "missing.jpg"})
    assert names == ["b.jpg"]
    assert tracks.shape == (1, 3, 1, 2)
    assert visibility.shape == (1, 3, 1)
    assert tracks[0, 0, 0, 0] == payload["tracks_xy"][1, 0, 0, 0]
