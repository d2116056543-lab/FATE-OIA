from pathlib import Path

def test_supervisor_source_foreground_only_and_requires_review_pass():
    src = Path('fate_oia/engine/supervise_diva_caf_oia_foreground.py').read_text(encoding='utf-8')
    forbidden = ['Start-Process','Start-Job','nohup','WindowStyle Hidden']
    for token in forbidden:
        assert token not in src
    assert 'RequireReviewPass' in src or 'require_review_pass' in src
