import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import requests

log = logging.getLogger("asset-manager")


class AssetManager:
    API = "https://api.pexels.com/videos/search"

    def __init__(self):
        self.api_key = os.getenv("PEXELS_API_KEY")
        if not self.api_key:
            raise RuntimeError("PEXELS_API_KEY is not configured.")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": self.api_key})

        self.per_query = int(os.getenv("PEXELS_RESULTS_PER_QUERY", "15"))
        self.min_score = float(os.getenv("PEXELS_MIN_SCORE", "0.30"))
        self.used_ids = set()

    @staticmethod
    def _tokens(text: str):
        return {
            x.lower()
            for x in re.findall(r"[A-Za-z0-9]{3,}", text or "")
        }

    def _score(self, query: str, video: Dict, intent: str) -> float:
        q = self._tokens(query + " " + intent)
        text = " ".join([
            str(video.get("url", "")),
            str(video.get("user", {}).get("name", "")),
        ])
        t = self._tokens(text)

        # Pexels does not provide semantic tags reliably, so use metadata as a weak
        # signal and let multiple query variants increase recall.
        lexical = len(q & t) / max(1, len(q))
        duration = float(video.get("duration") or 0)
        duration_score = 1.0 if duration >= 5 else duration / 5.0
        return 0.7 * lexical + 0.3 * duration_score

    def _best_file(self, video: Dict):
        files = video.get("video_files", [])
        landscape = []
        for f in files:
            width = f.get("width") or 0
            height = f.get("height") or 0
            link = f.get("link")
            if not link:
                continue
            if width >= 1280 and height >= 720 and width / max(1, height) >= 1.5:
                landscape.append(f)

        if not landscape:
            return None

        # Prefer a good HD source without blindly choosing enormous 4K files.
        landscape.sort(
            key=lambda f: (
                abs((f.get("width") or 0) - 1920),
                -(f.get("height") or 0),
            )
        )
        return landscape[0]

    def _search(self, query: str, intent: str) -> List[Tuple[float, Dict, Dict]]:
        try:
            r = self.session.get(
                self.API,
                params={
                    "query": query,
                    "per_page": self.per_query,
                    "orientation": "landscape",
                    "size": "medium",
                },
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            log.warning("Pexels search failed for %r: %s", query, e)
            return []

        ranked = []
        for video in payload.get("videos", []):
            vid = video.get("id")
            if not vid or vid in self.used_ids:
                continue
            vf = self._best_file(video)
            if not vf:
                continue
            score = self._score(query, video, intent)
            ranked.append((score, video, vf))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def _download(self, url: str, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".part")
        with self.session.get(url, stream=True, timeout=90) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp.replace(destination)

    async def prepare_assets(self, scenes: List[Dict], output_dir: Path, orientation="landscape"):
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, scene in enumerate(scenes, 1):
            queries = scene.get("visual_queries", [])
            candidates = []

            for query in queries:
                candidates.extend(self._search(query, scene.get("visual_intent", "")))

            # De-duplicate candidate videos while retaining the best score.
            best = {}
            for score, video, vf in candidates:
                vid = video["id"]
                if vid not in best or score > best[vid][0]:
                    best[vid] = (score, video, vf)

            ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)

            selected = []
            for score, video, vf in ranked:
                if score < self.min_score and selected:
                    continue
                selected.append((score, video, vf))
                if len(selected) >= 2:
                    break

            if not selected:
                raise RuntimeError(
                    f"No suitable landscape footage found for scene {scene.get('id')}."
                )

            visuals = []
            for j, (score, video, vf) in enumerate(selected, 1):
                filename = f"{idx:03d}_{j:02d}_{video['id']}.mp4"
                destination = output_dir / filename
                if not destination.exists():
                    self._download(vf["link"], destination)

                self.used_ids.add(video["id"])
                visuals.append({
                    "path": str(destination),
                    "provider": "pexels",
                    "provider_id": video["id"],
                    "score": round(score, 4),
                    "query": scene.get("visual_queries", [""])[0],
                })

            scene["visuals"] = visuals
            log.info(
                "Scene %s: selected %d distinct landscape clips",
                scene.get("id"),
                len(visuals),
            )
