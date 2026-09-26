import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from modules.brain import Brain
from modules.audio import AudioEngine
from modules.asset_manager import AssetManager
from modules.composer import Composer

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("long-form-pipeline")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def require_env():
    required = ["GEMINI_API_KEY", "PEXELS_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def save_json(path: Path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


async def run():
    require_env()

    target_minutes = float(os.getenv("TARGET_DURATION_MINUTES", "15"))
    min_scenes = int(os.getenv("MIN_SCENES", "24"))
    max_scenes = int(os.getenv("MAX_SCENES", "60"))

    brain = Brain()
    audio = AudioEngine()
    assets = AssetManager()
    composer = Composer()

    topic = os.getenv("TOPIC", "").strip()
    if not topic:
        topic = brain.get_trending_topic()

    log.info("Topic: %s", topic)
    log.info("Target duration: %.2f minutes", target_minutes)

    # Long-form structured script. Do not silently fall back to a short-form script.
    script = brain.generate_long_script(
        topic=topic,
        target_minutes=target_minutes,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
    )
    if not script or not script.get("scenes"):
        raise RuntimeError("Brain returned no usable scenes.")

    save_json(OUTPUT_DIR / "script.json", script)

    scenes = script["scenes"]

    # Generate narration first. Audio duration becomes the source of truth for timing.
    await audio.generate_for_scenes(scenes, OUTPUT_DIR / "audio")
    scenes = [s for s in scenes if s.get("audio_path") and s.get("duration", 0) > 0]
    if not scenes:
        raise RuntimeError("No scene audio was generated.")

    save_json(OUTPUT_DIR / "scenes_with_audio.json", {"topic": topic, "scenes": scenes})

    # Search multiple candidates per scene and rank them deterministically.
    await assets.prepare_assets(
        scenes,
        output_dir=OUTPUT_DIR / "visuals",
        orientation="landscape",
    )

    missing = [
        i for i, s in enumerate(scenes)
        if not s.get("visuals")
    ]
    if missing:
        raise RuntimeError(
            f"Visual acquisition failed for {len(missing)} scenes: {missing[:10]}"
        )

    save_json(OUTPUT_DIR / "scenes_with_visuals.json", {"topic": topic, "scenes": scenes})

    final_path = OUTPUT_DIR / "final_documentary.mp4"
    composer.render(scenes, final_path)

    if not final_path.exists() or final_path.stat().st_size < 100_000:
        raise RuntimeError("Renderer reported success but final video is missing/too small.")

    log.info("SUCCESS: %s (%d bytes)", final_path, final_path.stat().st_size)


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.error("Interrupted.")
        raise
    except Exception:
        log.exception("PIPELINE FAILED")
        # Critical for GitHub Actions: never convert a real failure into exit code 0.
        sys.exit(1)


if __name__ == "__main__":
    main()
