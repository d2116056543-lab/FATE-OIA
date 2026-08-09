import torch

from fate_oia.utils.pact_pair_queue import PACTBalancedPairQueue
from fate_oia.utils.pact_pareto_controller import PACTParetoController


def test_controller_and_queue_resume_exactly():
    controller = PACTParetoController([0, 0.25, 0.5], 0.001, 0.8)
    controller.semantic_share_license.fill_(0.375)
    restored = PACTParetoController([0, 0.25, 0.5], 0.001, 0.8)
    restored.load_state_dict(controller.state_dict())
    assert torch.equal(controller.semantic_share_license, restored.semantic_share_license)
    queue = PACTBalancedPairQueue(21, 4, 4)
    queue.enqueue(torch.zeros(2, 21), torch.stack((torch.ones(21), torch.zeros(21))), 3, torch.ones(2, 21))
    queue2 = PACTBalancedPairQueue(21, 4, 4); queue2.load_state_dict(queue.state_dict())
    _, before = queue.pairs(3, torch.device("cpu")); _, after = queue2.pairs(3, torch.device("cpu"))
    assert before == after
