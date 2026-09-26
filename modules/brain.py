import json
import logging
import os
import re
from typing import Any, Dict, List

import google.generativeai as genai
import requests

log = logging.getLogger("long-form-brain")


class Brain:
    def __init__(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        genai.configure(api_key=key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model = genai.GenerativeModel(self.model_name)

    def get_trending_topic(self) -> str:
        # Keep this deterministic and dependency-light. A configured topic is preferred.
        configured = os.getenv("TOPIC", "").strip()
        if configured:
            return configured
        return os.getenv(
            "DEFAULT_TOPIC",
            "The hidden history of a technology that changed everyday life",
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise

    def _normalize_scene(self, scene: Dict[str, Any], index: int) -> Dict[str, Any]:
        narration = str(scene.get("narration", "")).strip()
        if not narration:
            raise ValueError(f"Scene {index} has empty narration.")

        visual_queries = scene.get("visual_queries") or scene.get("visuals") or []
        if isinstance(visual_queries, str):
            visual_queries = [visual_queries]
        visual_queries = [str(x).strip() for x in visual_queries if str(x).strip()]

        # At least two distinct queries gives the asset layer room to find different footage.
        while len(visual_queries) < 3:
            visual_queries.append(
                f"{scene.get('title', 'documentary scene')} relevant documentary footage"
            )

        return {
            "id": scene.get("id", f"scene_{index:03d}"),
            "title": str(scene.get("title", f"Scene {index}")).strip(),
            "narration": narration,
            "visual_queries": visual_queries[:5],
            "visual_intent": str(scene.get("visual_intent", "")).strip(),
            "keywords": scene.get("keywords", []),
        }

    def generate_long_script(
        self,
        topic: str,
        target_minutes: float = 15,
        min_scenes: int = 24,
        max_scenes: int = 60,
    ) -> Dict[str, Any]:
        target_words = max(800, round(target_minutes * 130))
        scene_target = max(
            min_scenes,
            min(max_scenes, round(target_minutes * 2.5)),
        )

        prompt = f"""
You are a professional long-form documentary script planner.

TOPIC:
{topic}

TARGET:
- approximately {target_minutes:.1f} minutes
- approximately {target_words} narration words
- {scene_target} scenes
- narration pacing about 125-135 WPM

Create a coherent documentary, NOT a short-form script stretched with filler.

Requirements:
1. Strong hook and context.
2. Clear chronological or causal progression.
3. Multiple substantive sections.
4. Each scene advances the explanation.
5. Do not repeat the same fact.
6. Do not invent specific facts, quotations, statistics, dates, or sources.
7. Each scene needs 3-5 distinct visual search queries.
8. Visual queries must describe what should visibly appear on screen.
9. Prefer real-world documentary footage, locations, people, objects, diagrams, maps,
   archives, factories, nature, or other concrete visuals.
10. Avoid generic queries such as "interesting footage", "cinematic video", or "documentary".
11. Keep each scene's narration roughly 45-90 seconds so visual timing can follow narration.

Return ONLY valid JSON:
{{
  "title": "...",
  "description": "...",
  "topic": "...",
  "scenes": [
    {{
      "id": "scene_001",
      "title": "...",
      "narration": "...",
      "visual_intent": "...",
      "visual_queries": ["...", "...", "..."],
      "keywords": ["...", "..."]
    }}
  ]
}}
"""

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.45,
                "response_mime_type": "application/json",
            },
        )

        data = self._extract_json(response.text)
        raw_scenes = data.get("scenes", [])
        if not isinstance(raw_scenes, list):
            raise ValueError("LLM response contains invalid scenes.")

        scenes = [
            self._normalize_scene(scene, i + 1)
            for i, scene in enumerate(raw_scenes)
        ]

        if len(scenes) < max(8, min_scenes // 2):
            raise ValueError(
                f"LLM returned too few scenes ({len(scenes)}); refusing to make a fake long-form video."
            )

        data["topic"] = topic
        data["scenes"] = scenes
        data["target_minutes"] = target_minutes
        return data

    # Backwards-compatible alias for any older code.
    def generate_script(self, topic: str):
        return self.generate_long_script(
            topic=topic,
            target_minutes=float(os.getenv("TARGET_DURATION_MINUTES", "15")),
        )
