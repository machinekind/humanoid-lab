"""The one mp4 writer every video consumer routes through."""

import shutil
from pathlib import Path


def write_video(out, frames, fps) -> None:
    """Write `frames` to `out` (parent directories created) at `fps`."""
    import mediapy

    if shutil.which("ffmpeg") is None:
        # No system ffmpeg (common on a bare Mac); fall back to the binary
        # bundled with imageio-ffmpeg, already a project dependency.
        import imageio_ffmpeg

        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(out), frames, fps=fps)
