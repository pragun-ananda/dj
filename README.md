# SonicDJ 🎧

AI-powered DJ curation, semantic search, mix preparation, and live performance companion application with full compatibility for **Algoriddim djay Pro**.

---

## Features (Phase 1 Delivered)

- **Local Audio Scanner & Fingerprinting:** Recursively index audio collections (`FLAC`, `MP3`, `M4A/AAC`, `AIFF`, `WAV`) into an optimized local SQLite database with SHA-256 deduplication.
- **djay Pro ID3 & Vorbis Tag Engine:** Robust metadata extraction and writing of Camelot Keys (`8A`, `11B`), BPMs, Energy scores, 1–5 star ratings (`POPM`), and structural cue markers into file tags.
- **Algoriddim djay Pro Collection Exporter:** Generate native Rekordbox XML (`<DJ_PLAYLISTS>`) and extended M3U8 smart playlists with hot cues and beatgrids that import directly into djay Pro.
- **Tidal Ingestion & Batch Downloader Architecture:** Modular Tidal API client and batch downloader pipeline with metadata tagging.
- **Interactive CLI & Rich Formatting:** `sonicdj scan`, `sonicdj list`, `sonicdj tag`, `sonicdj info`, and `sonicdj export-djay`.

---

## Quickstart

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Scan an audio folder
sonicdj scan ~/Music/DJ_Tracks

# 3. List tracks filtered by Camelot key or BPM
sonicdj list --camelot 8A --min-bpm 120 --max-bpm 126

# 4. Tag a track for djay Pro
sonicdj tag track.mp3 --key "8A / Am" --bpm 123.0 --energy 0.85 --rating 5

# 5. Export to Rekordbox XML for Algoriddim djay Pro
sonicdj export-djay djay_collection.xml --m3u8
```

## Running Tests

```bash
uv run pytest -v
```
