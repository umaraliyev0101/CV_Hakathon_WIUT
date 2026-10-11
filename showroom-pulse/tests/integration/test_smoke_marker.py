import os
import imageio_ffmpeg
import pytest

@pytest.mark.slow
def test_imageio_ffmpeg_binary_available():
    assert os.path.exists(imageio_ffmpeg.get_ffmpeg_exe())
