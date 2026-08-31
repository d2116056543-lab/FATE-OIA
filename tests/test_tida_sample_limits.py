from argparse import Namespace

from fate_oia.engine.train_tida_oia import _optional_partition_limit


def test_split_limit_inherits_common_eval_limit_when_unspecified():
    args = Namespace(max_test_samples=None)
    assert _optional_partition_limit(args, "max_test_samples", 8) == 8


def test_explicit_split_limit_overrides_common_eval_limit():
    args = Namespace(max_test_samples=17)
    assert _optional_partition_limit(args, "max_test_samples", 8) == 17
