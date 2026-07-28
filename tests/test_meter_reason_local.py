from test_meter_reason_global import test_global_reason_view_is_directly_computed


def test_local_reason_view_shares_the_formal_decoder_path() -> None:
    # The global test exercises the same complete forward contract; this
    # named test makes the local-view requirement explicit for the audit.
    test_global_reason_view_is_directly_computed()
