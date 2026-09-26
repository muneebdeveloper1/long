import asyncio
import logging
from pathlib import Path

import edge_tts
from pydub import AudioSegment

log = logging.getLogger("audio-engine")


class AudioEngine:
    async def _one(self, text: str, path: Path, rate: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(
            text,
            voice="en-US-GuyNeural",
            rate=rate,
        )
        await communicate.save(str(path))

    async def generate_for_scenes(self, scenes, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        rate = str(__import__("os").getenv("TTS_RATE", "-2%"))

        for index, scene in enumerate(scenes, 1):
            path = output_dir / f"{index:03d}.mp3"
            await self._one(scene["narration"], path, rate)

            duration = len(AudioSegment.from_file(path)) / 1000.0
            if duration <= 0.1:
                raise RuntimeError(f"Invalid audio duration for scene {scene.get('id')}.")

            scene["audio_path"] = str(path)
            scene["duration"] = round(duration, 3)

            # Small pause prevents throttling without adding large fixed delays.
            await asyncio.sleep(0.15)
