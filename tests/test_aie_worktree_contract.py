import subprocess


def test_aie_uses_dedicated_branch():
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    assert branch in {
        "acpr_aie_oia_v1_direct_image",
        "vetra_trainable_073_040_v2",
        "vetra_from_scratch_staged_v1",
        "tida_oia_v1_video",
        "tida_oia_flow_credit_v1",
        "tida_oia_flow_credit_v1_exact",
    }

