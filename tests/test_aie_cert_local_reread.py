import torch
from fate_oia.models.aie_cert_deformable_reread import AIECertDeformableReread


def test_local_query_depends_on_global_and_map():
    m=AIECertDeformableReread(dim=16,grid_hw=(4,5),num_layers=3,points_per_layer=2)
    probes=torch.randn(1,4,4,16); field=torch.randn(1,3,20,16); amap=torch.softmax(torch.randn(1,4,4,20),-1)
    a=m(probes,field,amap,torch.zeros_like(probes)); b=m(probes,field,amap.roll(1,-1),torch.ones_like(probes))
    assert not torch.allclose(a['local_query'],b['local_query'])
    assert a['sampling_offsets'].abs().max() <= .25 + 1e-6
    assert torch.allclose(a['sampling_weights'].sum((3,4)),torch.ones(1,4,4))
