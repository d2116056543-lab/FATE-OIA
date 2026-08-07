import torch
from fate_oia.utils.aie_cert_preference_queue import AIECertPreferenceQueue,PreferenceBatch


def test_queue_capacity_age_and_resume():
    q=AIECertPreferenceQueue(capacity=2,max_age=3)
    b=PreferenceBatch(torch.zeros(3,21),torch.zeros(3,21),torch.zeros(3,21),torch.zeros(3,21),torch.ones(3,21),torch.tensor([0,1,2]),['a','b','c'])
    q.enqueue(b); assert [x['sample_id'] for x in q.records]==['b','c'] and len(q.eligible(5))==1
    clone=AIECertPreferenceQueue(); clone.load_state_dict(q.state_dict()); assert clone.records[0]['sample_id']=='b'
    assert all(not torch.is_tensor(value) or value.device.type == 'cpu'
               for row in clone.records for value in row.values())
