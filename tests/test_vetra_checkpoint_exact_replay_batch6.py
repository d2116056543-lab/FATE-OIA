import torch

from vetra_test_utils import build_model, fake_base


def test_transport_checkpoint_replays_batch6_exactly(tmp_path):
    model = build_model()
    base = fake_base(batch=6)
    expected = model.decode_base_output(base, alpha=.37)["action_logits_final"]
    checkpoint = tmp_path / "state.pth"
    torch.save(model.state_dict(), checkpoint)
    restored = build_model()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = restored.decode_base_output(base, alpha=.37)["action_logits_final"]
    assert torch.equal(expected, actual)

