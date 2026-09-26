import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger("composer")


def run(cmd):
    log.info("FFmpeg: %s", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed:\n" + p.stderr[-6000:]
        )


class Composer:
    def __init__(self):
        self.width = int(os.getenv("VIDEO_WIDTH", "1920"))
        self.height = int(os.getenv("VIDEO_HEIGHT", "1080"))
        self.fps = int(os.getenv("VIDEO_FPS", "30"))
        self.crf = int(os.getenv("VIDEO_CRF", "20"))
        self.preset = os.getenv("VIDEO_PRESET", "veryfast")

    def _scene(self, scene, index: int, outdir: Path) -> Path:
        outdir.mkdir(parents=True, exist_ok=True)
        output = outdir / f"scene_{index:04d}.mp4"

        visuals = scene["visuals"]
        audio = scene["audio_path"]
        duration = float(scene["duration"])

        # Alternate between the selected clips. Each clip is independently looped/
        # trimmed to cover half the narration duration.
        half = max(0.5, duration / len(visuals))

        inputs = []
        for v in visuals:
            inputs += ["-stream_loop", "-1", "-i", v["path"]]
        inputs += ["-i", audio]

        filters = []
        labels = []

        for i in range(len(visuals)):
            label = f"v{i}"
            filters.append(
                f"[{i}:v]scale={self.width}:{self.height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{self.height},fps={self.fps},"
                f"trim=duration={half},setpts=PTS-STARTPTS[{label}]"
            )
            labels.append(f"[{label}]")

        if len(labels) == 1:
            video = labels[0]
        else:
            # Concatenate visual segments without a transition between every tiny clip.
            concat_inputs = "".join(labels)
            filters.append(
                f"{concat_inputs}concat=n={len(labels)}:v=1:a=0[vcat]"
            )
            video = "[vcat]"

        filters.append(
            f"{video}format=yuv420p[vout]"
        )

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", ";".join(filters),
            "-map", "[vout]",
            "-map", f"{len(visuals)}:a:0",
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output),
        ]
        run(cmd)
        return output

    def render(self, scenes, final_path: Path):
        work = final_path.parent / "rendered_scenes"
        work.mkdir(parents=True, exist_ok=True)

        scene_files = []
        for i, scene in enumerate(scenes, 1):
            scene_files.append(self._scene(scene, i, work))

        concat_file = work / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{p.resolve()}'\n" for p in scene_files),
            encoding="utf-8",
        )

        # Re-encode once at the final stage. This avoids the previous xfade chain
        # repeatedly decoding/re-encoding the entire growing video.
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(final_path),
        ]
        run(cmd)

        if not final_path.exists() or final_path.stat().st_size < 100_000:
            raise RuntimeError("Final render did not produce a valid video.")
