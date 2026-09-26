import json
import logging
import os
import re
import time
from typing import Any, Dict, List

import requests

log = logging.getLogger("long-form-brain")


class Brain:
    """
    Gemini brain for the long-form documentary pipeline.

    Important:
    - Uses Gemini REST API directly.
    - Does NOT use google.generativeai / gRPC.
    - Uses controlled retries.
    - Supports multiple fallback models.
    - Never waits 600 seconds on one failed request.
    - Returns structured JSON for the rest of the pipeline.
    """

    GEMINI_API_BASE = (
        "https://generativelanguage.googleapis.com/v1beta/models"
    )

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        self.timeout = int(
            os.getenv("GEMINI_REQUEST_TIMEOUT", "45")
        )

        self.max_retries = int(
            os.getenv("GEMINI_MAX_RETRIES", "4")
        )

        self.retry_delay = float(
            os.getenv("GEMINI_RETRY_DELAY", "3")
        )

        # Primary model.
        primary_model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash",
        ).strip()

        # Optional comma-separated fallback models.
        fallback_models_raw = os.getenv(
            "GEMINI_FALLBACK_MODELS",
            "gemini-3.5-flash-lite,gemini-3.0-flash",
        )

        fallback_models = [
            x.strip()
            for x in fallback_models_raw.split(",")
            if x.strip()
        ]

        # Preserve order and remove duplicates.
        self.models = []

        for model in [primary_model] + fallback_models:
            if model and model not in self.models:
                self.models.append(model)

        if not self.models:
            raise RuntimeError(
                "No Gemini models are configured."
            )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "long-form-documentary-pipeline/1.0",
                "x-goog-api-key": self.api_key,
            }
        )

        log.info(
            "Gemini REST client initialized. Models: %s",
            ", ".join(self.models),
        )

    # ------------------------------------------------------------------
    # BASIC HELPERS
    # ------------------------------------------------------------------

    def get_trending_topic(self) -> str:
        """
        Returns a configured topic.

        The long-form workflow should normally receive TOPIC from
        GitHub Actions. This fallback prevents an empty-topic crash.
        """

        configured = os.getenv("TOPIC", "").strip()

        if configured:
            return configured

        return os.getenv(
            "DEFAULT_TOPIC",
            "The hidden history of a technology that changed everyday life",
        ).strip()

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Robustly extracts JSON from Gemini output.

        Handles:
        - normal JSON
        - ```json ... ```
        - accidental surrounding text
        """

        if not text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        text = text.strip()

        # Remove markdown fences.
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        # First attempt: entire response.
        try:
            result = json.loads(text)

            if isinstance(result, dict):
                return result

            raise ValueError(
                "Gemini JSON response is not an object."
            )

        except json.JSONDecodeError:
            pass

        # Second attempt: find the outermost JSON object.
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            candidate = text[start:end + 1]

            try:
                result = json.loads(candidate)

                if isinstance(result, dict):
                    return result

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Could not parse valid JSON from Gemini response."
        )

    def _normalize_model_name(self, model: str) -> str:
        """
        Allows users to provide either:
        gemini-2.5-flash

        or:

        models/gemini-2.5-flash
        """

        model = model.strip()

        if model.startswith("models/"):
            model = model[len("models/"):]

        return model

    # ------------------------------------------------------------------
    # GEMINI REST REQUEST
    # ------------------------------------------------------------------

    def _request_gemini(
        self,
        model: str,
        prompt: str,
    ) -> str:
        """
        Make one direct REST request to Gemini.

        This intentionally does NOT use google.generativeai because
        the previous implementation was failing with:

            503 Illegal metadata

        inside the gRPC stack.
        """

        model = self._normalize_model_name(model)

        url = (
            f"{self.GEMINI_API_BASE}/"
            f"{model}:generateContent"
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.45,
                "responseMimeType": "application/json",
            },
        }

        log.info(
            "Calling Gemini REST API: %s",
            model,
        )

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout,
            )

        except requests.Timeout as exc:
            raise RuntimeError(
                f"Gemini request timed out after "
                f"{self.timeout} seconds."
            ) from exc

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Gemini network request failed: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # HTTP ERROR HANDLING
        # --------------------------------------------------------------

        if response.status_code != 200:

            body = response.text[:2000]

            if response.status_code == 429:
                raise RuntimeError(
                    f"Gemini rate limited (HTTP 429): {body}"
                )

            if response.status_code in (500, 502, 503, 504):
                raise RuntimeError(
                    f"Gemini temporary server error "
                    f"(HTTP {response.status_code}): {body}"
                )

            if response.status_code == 400:
                raise RuntimeError(
                    f"Gemini rejected request (HTTP 400): {body}"
                )

            if response.status_code == 401:
                raise RuntimeError(
                    "Gemini authentication failed (HTTP 401). "
                    "Check GEMINI_API_KEY."
                )

            if response.status_code == 403:
                raise RuntimeError(
                    "Gemini API access denied (HTTP 403). "
                    "Check API key/project/model permissions."
                )

            if response.status_code == 404:
                raise RuntimeError(
                    f"Gemini model not found (HTTP 404): "
                    f"{model}. Response: {body}"
                )

            raise RuntimeError(
                f"Gemini HTTP {response.status_code}: {body}"
            )

        # --------------------------------------------------------------
        # PARSE RESPONSE
        # --------------------------------------------------------------

        try:
            data = response.json()

        except ValueError as exc:
            raise RuntimeError(
                "Gemini returned invalid JSON at HTTP level."
            ) from exc

        try:
            candidates = data.get("candidates", [])

            if not candidates:
                raise RuntimeError(
                    f"Gemini returned no candidates: {data}"
                )

            candidate = candidates[0]

            content = candidate.get("content", {})

            parts = content.get("parts", [])

            if not parts:
                raise RuntimeError(
                    f"Gemini returned no content parts: {data}"
                )

            text_parts = []

            for part in parts:
                text = part.get("text")

                if text:
                    text_parts.append(text)

            text = "".join(text_parts).strip()

            if not text:
                raise RuntimeError(
                    "Gemini response contained no text."
                )

            return text

        except Exception as exc:

            if isinstance(exc, RuntimeError):
                raise

            raise RuntimeError(
                f"Unexpected Gemini response structure: {data}"
            ) from exc

    # ------------------------------------------------------------------
    # RETRY + FALLBACK
    # ------------------------------------------------------------------

    def _generate_with_fallbacks(
        self,
        prompt: str,
    ) -> str:
        """
        Try every configured model.

        Example:

        gemini-2.5-flash
            ↓ retry
            ↓ retry
            ↓ retry
            ↓ retry

        then:

        gemini-2.5-flash-lite
            ↓ retry...

        then:

        gemini-2.0-flash
        """

        errors = []

        for model in self.models:

            for attempt in range(
                1,
                self.max_retries + 1,
            ):

                log.info(
                    "Gemini attempt %d/%d using %s",
                    attempt,
                    self.max_retries,
                    model,
                )

                try:

                    result = self._request_gemini(
                        model=model,
                        prompt=prompt,
                    )

                    if not result.strip():
                        raise RuntimeError(
                            "Gemini returned an empty result."
                        )

                    log.info(
                        "Gemini request succeeded using %s",
                        model,
                    )

                    return result

                except Exception as exc:

                    error_message = str(exc)

                    errors.append(
                        f"{model} attempt {attempt}: "
                        f"{error_message}"
                    )

                    log.warning(
                        "Gemini attempt failed: %s",
                        error_message,
                    )

                    # Do not sleep after the final attempt.
                    if attempt < self.max_retries:

                        delay = (
                            self.retry_delay
                            * attempt
                        )

                        log.info(
                            "Retrying Gemini in %.1f seconds...",
                            delay,
                        )

                        time.sleep(delay)

            log.warning(
                "Model %s exhausted all %d attempts. "
                "Moving to next fallback model.",
                model,
                self.max_retries,
            )

        error_text = "\n".join(errors[-12:])

        raise RuntimeError(
            "ALL GEMINI MODELS FAILED.\n\n"
            + error_text
        )

    # ------------------------------------------------------------------
    # SCENE NORMALIZATION
    # ------------------------------------------------------------------

    def _normalize_scene(
        self,
        scene: Dict[str, Any],
        index: int,
    ) -> Dict[str, Any]:

        if not isinstance(scene, dict):
            raise ValueError(
                f"Scene {index} is not a JSON object."
            )

        narration = str(
            scene.get("narration", "")
        ).strip()

        if not narration:
            raise ValueError(
                f"Scene {index} has empty narration."
            )

        title = str(
            scene.get(
                "title",
                f"Scene {index}",
            )
        ).strip()

        visual_queries = (
            scene.get("visual_queries")
            or scene.get("visuals")
            or []
        )

        if isinstance(
            visual_queries,
            str,
        ):
            visual_queries = [
                visual_queries
            ]

        if not isinstance(
            visual_queries,
            list,
        ):
            visual_queries = []

        visual_queries = [
            str(x).strip()
            for x in visual_queries
            if str(x).strip()
        ]

        # We want several independent visual search ideas.
        if len(visual_queries) < 3:

            fallback_query = (
                f"{title} documentary footage"
            )

            while len(visual_queries) < 3:
                visual_queries.append(
                    fallback_query
                )

        keywords = scene.get(
            "keywords",
            [],
        )

        if isinstance(
            keywords,
            str,
        ):
            keywords = [keywords]

        if not isinstance(
            keywords,
            list,
        ):
            keywords = []

        return {
            "id": scene.get(
                "id",
                f"scene_{index:03d}",
            ),
            "title": title,
            "narration": narration,
            "visual_queries": visual_queries[:5],
            "visual_intent": str(
                scene.get(
                    "visual_intent",
                    "",
                )
            ).strip(),
            "keywords": [
                str(x).strip()
                for x in keywords
                if str(x).strip()
            ],
        }

    # ------------------------------------------------------------------
    # LONG-FORM SCRIPT GENERATION
    # ------------------------------------------------------------------

    def generate_long_script(
        self,
        topic: str,
        target_minutes: float = 15,
        min_scenes: int = 24,
        max_scenes: int = 60,
    ) -> Dict[str, Any]:

        topic = str(topic).strip()

        if not topic:
            raise ValueError(
                "Topic cannot be empty."
            )

        target_minutes = float(
            target_minutes
        )

        if target_minutes <= 0:
            raise ValueError(
                "target_minutes must be greater than zero."
            )

        target_words = max(
            800,
            round(
                target_minutes * 130
            ),
        )

        scene_target = max(
            min_scenes,
            min(
                max_scenes,
                round(
                    target_minutes * 2.5
                ),
            ),
        )

        prompt = f"""
You are a professional long-form documentary
script planner and researcher.

TOPIC:
{topic}

TARGET LENGTH:
Approximately {target_minutes:.1f} minutes.

TARGET WORD COUNT:
Approximately {target_words} narration words.

TARGET NUMBER OF SCENES:
Approximately {scene_target} scenes.

NARRATION SPEED:
Approximately 125-135 words per minute.

IMPORTANT:
Create a genuine long-form documentary.

DO NOT:
- create a short-form script stretched with filler
- repeat the same information
- repeat the same visual idea unnecessarily
- invent quotations
- invent statistics
- invent dates
- invent historical events
- make unsupported factual claims
- use generic filler narration

STRUCTURE:

1. Strong opening hook.
2. Introduction and context.
3. Multiple substantive sections.
4. Clear chronological, causal, or thematic progression.
5. Important turning points.
6. Consequences and implications.
7. Strong conclusion.

SCENE REQUIREMENTS:

Every scene must contain:

- a unique title
- meaningful narration
- a clear visual intention
- 3-5 different visual search queries
- useful keywords

Each scene should normally contain
approximately 45-90 seconds of narration.

VISUAL SEARCH REQUIREMENTS:

Visual queries must describe things that can
actually appear in documentary footage.

GOOD EXAMPLES:

- 1980s Tokyo city streets
- workers assembling automobiles in factory
- satellite view of hurricane formation
- ancient Roman ruins aerial footage
- modern financial trading floor
- historical newspaper printing press

BAD EXAMPLES:

- interesting footage
- cinematic documentary
- beautiful background
- cool video
- professional footage

The visual queries should be specific enough
for a stock-video search engine.

OUTPUT:

Return ONLY valid JSON.

Do not use markdown.

Use exactly this general structure:

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
      "visual_queries": [
        "...",
        "...",
        "..."
      ],
      "keywords": [
        "...",
        "..."
      ]
    }}
  ]
}}
"""

        log.info(
            "Generating long-form script: "
            "topic=%s target=%.1f min scenes=%d",
            topic,
            target_minutes,
            scene_target,
        )

        # Gemini REST request with controlled retry/fallback.
        response_text = (
            self._generate_with_fallbacks(
                prompt
            )
        )

        try:
            data = self._extract_json(
                response_text
            )

        except Exception as exc:

            # Save a small diagnostic copy in the logs,
            # without exposing the API key.
            preview = response_text[:3000]

            log.error(
                "Gemini returned non-parseable JSON."
            )

            log.error(
                "Gemini response preview:\n%s",
                preview,
            )

            raise RuntimeError(
                "Invalid JSON returned by Gemini."
            ) from exc

        raw_scenes = data.get(
            "scenes",
            [],
        )

        if not isinstance(
            raw_scenes,
            list,
        ):
            raise ValueError(
                "Gemini response contains invalid scenes."
            )

        scenes: List[Dict[str, Any]] = []

        for index, scene in enumerate(
            raw_scenes,
            start=1,
        ):

            normalized = (
                self._normalize_scene(
                    scene,
                    index,
                )
            )

            scenes.append(
                normalized
            )

        minimum_acceptable_scenes = max(
            8,
            min_scenes // 2,
        )

        if len(scenes) < minimum_acceptable_scenes:

            raise ValueError(
                f"Gemini returned only "
                f"{len(scenes)} scenes. "
                f"Expected at least "
                f"{minimum_acceptable_scenes}. "
                f"Refusing to generate a fake "
                f"long-form video."
            )

        data["topic"] = topic

        data["scenes"] = scenes

        data["target_minutes"] = (
            target_minutes
        )

        data["target_words"] = (
            target_words
        )

        data["scene_count"] = (
            len(scenes)
        )

        log.info(
            "Long-form script generated successfully: "
            "%d scenes",
            len(scenes),
        )

        return data

    # ------------------------------------------------------------------
    # BACKWARD COMPATIBILITY
    # ------------------------------------------------------------------

    def generate_script(
        self,
        topic: str,
    ):

        """
        Backwards-compatible method.

        Older code can still call:

            brain.generate_script(topic)

        and it will use the long-form generator.
        """

        return self.generate_long_script(
            topic=topic,
            target_minutes=float(
                os.getenv(
                    "TARGET_DURATION_MINUTES",
                    "15",
                )
            ),
            min_scenes=int(
                os.getenv(
                    "MIN_SCENES",
                    "24",
                )
            ),
            max_scenes=int(
                os.getenv(
                    "MAX_SCENES",
                    "60",
                )
            ),
        )
