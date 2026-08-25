from fate_oia.engine.repair_tida_zero_frame_clips import ffmpeg_select_filter, frame_window


def test_frame_window_preserves_target_and_five_second_history():
    start, end, count = frame_window(target_frame=600, fps=30.0, seconds=5.0)

    assert (start, end, count) == (450, 600, 151)


def test_frame_window_clamps_at_video_start():
    start, end, count = frame_window(target_frame=80, fps=30.0, seconds=5.0)

    assert (start, end, count) == (0, 80, 81)


def test_ffmpeg_filter_is_direct_process_safe():
    value = ffmpeg_select_filter(450, 600, 30.0)

    assert "'" not in value
    assert value == "select=between(n\\,450\\,600),setpts=N/30.000000000000/TB"
