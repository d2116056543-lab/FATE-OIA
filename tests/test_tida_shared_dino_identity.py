from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor
from fate_oia.models.tida_context_encoder import TIDAContextEncoder


def test_context_encoder_weakly_references_same_dino():
    dino = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=8)
    context = TIDAContextEncoder(dino_extractor=dino, query_reader=None, context_chunk_size=2)
    assert context.dino_extractor is dino
    assert context.dino_extractor.backbone is dino.backbone
