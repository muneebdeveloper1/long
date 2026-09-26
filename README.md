# Automated Widescreen Documentary Pipeline

A GitHub Actions pipeline for generating landscape 16:9 documentary videos.

Pipeline:

1. Topic
2. Long-form structured script
3. Edge TTS narration
4. Landscape Pexels search
5. Candidate filtering and duplicate avoidance
6. Scene rendering
7. Final 1920x1080 H.264/AAC video
8. GitHub Actions artifact

## Required GitHub Secrets

- `GEMINI_API_KEY`
- `PEXELS_API_KEY`

Optional:

- `GEMINI_MODEL`

## Manual run

Use **Actions → Automated Widescreen Documentary Pipeline → Run workflow**.

You can provide:

- Topic
- Target duration in minutes

## Local run

```bash
python -m pip install -r requirements.txt
python main.py
```

Set the environment variables before running.

## Important behavior

The application exits with a non-zero status on real failures. It does not silently report failed generation as a successful GitHub Actions job.

The visual pipeline requests landscape footage for the 16:9 output and selects distinct candidates rather than randomly selecting one of five results.
