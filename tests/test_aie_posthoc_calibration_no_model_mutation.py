import torch

from fate_oia.models.aie_oia_model import AIEOIAModel
from fate_oia.utils.aie_calibration import fit_posthoc_thresholds
from fate_oia.utils.aie_hashes import state_dict_sha256


def test_threshold_fit_does_not_mutate_model():
    model = AIEOIAModel(dim=32, mock_dim=32, use_mock_dino=True); before = state_dict_sha256(model.state_dict())
    fit_posthoc_thresholds(torch.randn(32, 25), torch.randint(0, 2, (32, 25)).float(), [list(range(4)), list(range(4, 25))])
    assert state_dict_sha256(model.state_dict()) == before

