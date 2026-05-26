import torch
from torch import nn

from fate_oia.engine.train_fate_oia import load_resume_checkpoint


def test_load_resume_checkpoint_restores_model_optimizer_and_epoch(tmp_path):
    source = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
    with torch.no_grad():
        source.weight.fill_(1.25)
        source.bias.fill_(-0.5)
    loss = source(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()

    checkpoint = {
        "epoch": 4,
        "model": source.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_state_dict": {"last_epoch": 4},
        "best_test_score": 0.42,
        "best_val_score": 0.37,
    }
    path = tmp_path / "checkpoint_latest.pth"
    torch.save(checkpoint, path)

    target = nn.Linear(3, 2)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.01)
    state = load_resume_checkpoint(path, target, target_optimizer, device="cpu")

    assert state.start_epoch == 5
    assert state.best_test_score == 0.42
    assert state.best_val_score == 0.37
    assert state.optimizer_restored is True
    assert state.scheduler_state == {"last_epoch": 4}
    for expected, actual in zip(source.parameters(), target.parameters()):
        assert torch.allclose(expected, actual)
    assert target_optimizer.state_dict()["state"]
