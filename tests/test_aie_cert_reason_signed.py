import torch
from fate_oia.models.aie_cert_reason_rereader import AIECertReasonRereader


def test_support_and_inhibition_use_contribution_sign():
    m=AIECertReasonRereader(dim=16,num_layers=3,predicate_names=[str(i) for i in range(32)])
    args=dict(reason_nodes=torch.randn(1,21,16),field=torch.randn(1,3,20,16),atom_token=torch.randn(1,4,4,16),
        atom_map=torch.softmax(torch.randn(1,4,4,20),-1),predicate_attention=torch.softmax(torch.randn(1,32,20),-1),
        predicate_probs=torch.rand(1,32),primary_logits=torch.randn(1,21))
    pos=m(contribution=torch.ones(1,4,4),**args); neg=m(contribution=-torch.ones(1,4,4),**args)
    assert pos['reason_action_support_prior'].sum()>0 and pos['reason_action_inhibit_prior'].sum()==0
    assert neg['reason_action_inhibit_prior'].sum()>0 and neg['reason_action_support_prior'].sum()==0
