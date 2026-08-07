import yaml


def test_runtime_contract_values():
    cfg=yaml.safe_load(open('configs/fate_oia_train_360x640_aie_cert_oia_v1.yaml',encoding='utf-8'))
    assert cfg['training']['epochs']==16 and cfg['training']['batch_size']==6
    assert cfg['runtime']['max_reserved_memory_gb']==44.5 and cfg['data']['num_workers']==8
    assert cfg['experiment']['feature_cache_enabled'] is False and cfg['experiment']['token_compression']=='none'
