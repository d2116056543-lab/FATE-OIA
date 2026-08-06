import torch

from fate_oia.engine.train_aie_oia import normalize_accumulated_gradients


def test_tail_window_scales_by_actual_microbatch_count():
    parameter = torch.nn.Parameter(torch.tensor(1.0)); parameter.grad = torch.tensor(6.0)
    normalize_accumulated_gradients([parameter], 2)
    torch.testing.assert_close(parameter.grad, torch.tensor(3.0))

