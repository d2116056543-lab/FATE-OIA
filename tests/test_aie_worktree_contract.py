import subprocess


def test_aie_uses_dedicated_branch():
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    assert branch == "acpr_aie_oia_v1_direct_image"

