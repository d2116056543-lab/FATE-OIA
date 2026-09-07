import torch

from fate_oia.models.coev_video_decoder import CoEVVideoDecoder


def test_full_patch_reread_has_layer_space_time_identity_and_masks_invalid_history():
    torch.manual_seed(4);model=CoEVVideoDecoder(dim=12,temporal_layers=3).eval()
    frame_q=torch.randn(1,15,25,12);times=torch.linspace(-5,0,15).unsqueeze(0)
    valid=torch.ones(1,15,dtype=torch.bool);valid[:,0]=False
    history=[torch.randn(1,14,4,12) for _ in range(3)];target=[torch.randn(1,6,12) for _ in range(3)]
    out=model.read_history(frame_q,times,valid,history,target,(2,2),(2,3))
    changed=[x.clone() for x in history]
    for x in changed: x[:,0].add_(1000)
    out_changed=model.read_history(frame_q,times,valid,changed,target,(2,2),(2,3))
    assert out["full_history_kv_length"]==14*4*3+6*3
    assert torch.allclose(out["q_video"],out_changed["q_video"],atol=2e-5)
    shifted=model.read_history(frame_q,times*.5,valid,history,target,(2,2),(2,3))["q_video"]
    assert not torch.allclose(out["q_video"],shifted)
    assert len(model.layer_proj)==3 and model.layer_id.shape==(3,12)


def test_nine_frame_decoder_uses_all_eight_history_frames_and_target():
    decoder=CoEVVideoDecoder(dim=12,temporal_layers=1)
    frame_q=torch.randn(1,9,25,12);times=torch.linspace(-5,0,9).unsqueeze(0)
    valid=torch.ones(1,9,dtype=torch.bool)
    history=[torch.randn(1,8,4,12) for _ in range(3)];target=[torch.randn(1,6,12) for _ in range(3)]
    out=decoder.read_history(frame_q,times,valid,history,target,(2,2),(2,3))
    assert out["q_video"].shape==(1,25,12)
    assert out["full_history_kv_length"]==8*4*3+6*3


def test_action_spatial_support_is_canonical_and_left_right_mirror():
    model=CoEVVideoDecoder(dim=12);support=model.class_spatial_support((16,28),torch.device("cpu"),torch.float32)
    assert torch.equal(support[2].flip(-1),support[3])
    assert support[0,-1,14]==1 and support[0,0,14]==0


def test_vectorized_spatial_frames_equal_explicit_per_frame_calls():
    torch.manual_seed(17)
    model = CoEVVideoDecoder(dim=12, temporal_layers=1).eval()
    b, t, n, d = 2, 14, 6, 12
    fields = [torch.randn(b, t, n, d) for _ in range(3)]
    explicit = torch.stack([
        model.read_frame([layer[:, frame] for layer in fields], (2, 3))
        for frame in range(t)
    ], 1)
    folded = model.read_frame([layer.reshape(b * t, n, d) for layer in fields], (2, 3))
    assert torch.allclose(explicit, folded.view(b, t, 25, d), atol=1e-6, rtol=1e-5)
