import os
import hashlib
from pathlib import Path
from typing import Optional
import mutagen
from mutagen.id3 import (
    ID3,
    TIT2,
    TPE1,
    TALB,
    TCON,
    TDRC,
    TBPM,
    TKEY,
    COMM,
    POPM,
    TXXX,
    ID3NoHeaderError,
)
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.aiff import AIFF
from mutagen.wave import WAVE
from mutagen.mp3 import MP3

from sonicdj.metadata.models import TrackMetadata, CueMetadata, normalize_key_to_camelot


class AudioTagEngine:
    """High-level audio metadata extractor and ID3 tag writer for djay Pro."""

    @staticmethod
    def calculate_file_hash(file_path: Path, block_size: int = 65536) -> str:
        """Compute SHA-256 hash of audio file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def read_metadata(cls, file_path: Path) -> TrackMetadata:
        """Extract all metadata and audio stream metrics from an audio file."""
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        meta = TrackMetadata()
        meta.format = file_path.suffix.lstrip(".").lower()
        meta.file_size_bytes = os.path.getsize(file_path)
        meta.file_hash = cls.calculate_file_hash(file_path) if meta.file_size_bytes > 0 else f"hash_{file_path.stem}"
        meta.title = file_path.stem

        try:
            audio = mutagen.File(file_path, easy=False)
        except Exception:
            audio = None

        # Technical audio properties
        if audio is not None and hasattr(audio, "info") and audio.info is not None:
            info = audio.info
            meta.duration_sec = float(getattr(info, "length", 0.0))
            meta.sample_rate = int(getattr(info, "sample_rate", 44100))
            meta.channels = int(getattr(info, "channels", 2))
            meta.bitrate_kbps = int(getattr(info, "bitrate", 0) // 1000)

        # Parse tags based on format
        if audio is not None and hasattr(audio, "tags") and audio.tags is not None:
            if isinstance(audio, (MP3, AIFF, WAVE)) or isinstance(audio.tags, ID3):
                cls._read_id3_tags(audio.tags, meta)
            elif isinstance(audio, FLAC):
                cls._read_flac_tags(audio, meta)
            elif isinstance(audio, MP4):
                cls._read_mp4_tags(audio, meta)
        else:
            # Fallback for ID3 tagged files if audio container parser failed
            if file_path.suffix.lower() in (".mp3", ".aif", ".aiff", ".wav"):
                try:
                    fallback_id3 = ID3(file_path)
                    cls._read_id3_tags(fallback_id3, meta)
                except Exception:
                    pass

        # Standardize Camelot & Musical Key
        if meta.camelot or meta.key_raw:
            cam, raw_key = normalize_key_to_camelot(meta.camelot or meta.key_raw)
            if cam:
                meta.camelot = cam
            if raw_key:
                meta.key_raw = raw_key

        # Extract cues from comments if present
        if meta.comments and not meta.cues:
            import re
            pattern = r"\[([^:]+):\s*(\d+)s\]"
            matches = re.findall(pattern, meta.comments)
            if matches:
                meta.cues = [
                    CueMetadata(
                        name=name.strip(),
                        timestamp_ms=int(sec) * 1000,
                        hot_cue_index=idx,
                        cue_type="intro" if "intro" in name.lower() else ("drop" if "drop" in name.lower() else "hot_cue")
                    )
                    for idx, (name, sec) in enumerate(matches)
                ]

        return meta

    @staticmethod
    def _read_id3_tags(tags: Optional[ID3], meta: TrackMetadata) -> None:
        if tags is None:
            return

        def get_frame_text(frame_id: str) -> str:
            frame = tags.get(frame_id)
            if frame and hasattr(frame, "text") and frame.text:
                return str(frame.text[0]).strip()
            return ""

        title = get_frame_text("TIT2")
        if title:
            meta.title = title

        artist = get_frame_text("TPE1")
        if artist:
            meta.artist = artist

        album = get_frame_text("TALB")
        if album:
            meta.album = album

        genre = get_frame_text("TCON")
        if genre:
            meta.genre = genre

        # Year
        year_str = get_frame_text("TDRC")
        if year_str and year_str[:4].isdigit():
            meta.year = int(year_str[:4])

        # BPM
        bpm_str = get_frame_text("TBPM")
        if bpm_str:
            try:
                meta.bpm = float(bpm_str)
            except ValueError:
                pass

        # Key / Camelot
        key_str = get_frame_text("TKEY")
        if key_str:
            meta.key_raw = key_str

        # TXXX Extended frames (e.g. Camelot, EnergyLevel)
        for frame in tags.getall("TXXX"):
            desc = getattr(frame, "desc", "").lower()
            text = str(frame.text[0]) if frame.text else ""
            if desc in ("camelot", "initialkey"):
                meta.camelot = text
            elif desc in ("energy", "energylevel"):
                try:
                    meta.energy = float(text)
                except ValueError:
                    pass

        # Rating / POPM (Popularimeter)
        for popm in tags.getall("POPM"):
            if hasattr(popm, "rating"):
                r = popm.rating
                if r >= 255: meta.rating = 5
                elif r >= 196: meta.rating = 4
                elif r >= 128: meta.rating = 3
                elif r >= 64: meta.rating = 2
                elif r >= 1: meta.rating = 1
                break

        # Comments
        for frame in tags.getall("COMM"):
            text = str(frame.text[0]) if frame.text else ""
            if text:
                meta.comments = text
                break

    @staticmethod
    def _read_flac_tags(flac: FLAC, meta: TrackMetadata) -> None:
        def get_tag(tag_name: str) -> str:
            vals = flac.get(tag_name, [])
            return str(vals[0]).strip() if vals else ""

        title = get_tag("TITLE")
        if title:
            meta.title = title

        artist = get_tag("ARTIST")
        if artist:
            meta.artist = artist

        album = get_tag("ALBUM")
        if album:
            meta.album = album

        genre = get_tag("GENRE")
        if genre:
            meta.genre = genre
        
        date_str = get_tag("DATE")
        if date_str and date_str[:4].isdigit():
            meta.year = int(date_str[:4])

        bpm_str = get_tag("BPM")
        if bpm_str:
            try:
                meta.bpm = float(bpm_str)
            except ValueError:
                pass

        key_str = get_tag("INITIALKEY") or get_tag("KEY")
        if key_str:
            cam, raw_k = normalize_key_to_camelot(key_str)
            meta.camelot = cam or meta.camelot
            meta.key_raw = raw_k or key_str

        camelot_str = get_tag("CAMELOT")
        if camelot_str:
            meta.camelot = camelot_str

        energy_str = get_tag("ENERGY")
        if energy_str:
            try:
                meta.energy = float(energy_str)
            except ValueError:
                pass

        rating_str = get_tag("RATING")
        if rating_str and rating_str.isdigit():
            meta.rating = int(rating_str)

        comments = get_tag("COMMENT")
        if comments:
            meta.comments = comments

    @staticmethod
    def _read_mp4_tags(mp4: MP4, meta: TrackMetadata) -> None:
        tags = mp4.tags
        if not tags:
            return

        def get_tag(tag_key: str) -> str:
            vals = tags.get(tag_key, [])
            return str(vals[0]).strip() if vals else ""

        title = get_tag("\xa9nam")
        if title:
            meta.title = title

        artist = get_tag("\xa9ART")
        if artist:
            meta.artist = artist

        album = get_tag("\xa9alb")
        if album:
            meta.album = album

        genre = get_tag("\xa9gen")
        if genre:
            meta.genre = genre

        day_str = get_tag("\xa9day")
        if day_str and day_str[:4].isdigit():
            meta.year = int(day_str[:4])

        tmpo = tags.get("tmpo")
        if tmpo and isinstance(tmpo, list) and tmpo[0]:
            meta.bpm = float(tmpo[0])

        comments = get_tag("\xa9cmt")
        if comments:
            meta.comments = comments

        for key, val in tags.items():
            if "CAMELOT" in key.upper():
                meta.camelot = str(val[0]) if isinstance(val, list) and val else str(val)
            elif "KEY" in key.upper() and not meta.key_raw:
                meta.key_raw = str(val[0]) if isinstance(val, list) and val else str(val)
            elif "ENERGY" in key.upper():
                try:
                    meta.energy = float(val[0]) if isinstance(val, list) and val else float(val)
                except ValueError:
                    pass

    @classmethod
    def write_djay_pro_tags(cls, file_path: Path, meta: TrackMetadata) -> bool:
        """
        Writes enriched DJ performance tags (BPM, Camelot Key, Energy, Comments, Hot Cues)
        into the audio file in a format natively readable by Algoriddim djay Pro.
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        cam, raw_key = normalize_key_to_camelot(meta.camelot or meta.key_raw)
        djay_key_display = f"{cam} / {raw_key}" if (cam and raw_key and cam != raw_key) else (cam or raw_key)
        suffix = file_path.suffix.lower()

        comment_str = meta.comments or ""
        if meta.cues:
            cue_summary = " ".join([f"[{c.name}: {c.timestamp_ms // 1000}s]" for c in meta.cues])
            comment_str = f"{comment_str} | {cue_summary}".strip(" |")

        # 1. MP3
        if suffix == ".mp3":
            try:
                audio = MP3(file_path)
            except Exception:
                audio = None

            if audio is None or audio.tags is None:
                try:
                    tags = ID3(file_path)
                except ID3NoHeaderError:
                    tags = ID3()
            else:
                tags = audio.tags

            cls._populate_id3_frames(tags, meta, cam, djay_key_display, comment_str)
            tags.save(file_path)
            return True

        # 2. WAV
        elif suffix == ".wav":
            try:
                wav = WAVE(file_path)
                if wav.tags is None:
                    wav.add_tags()
                cls._populate_id3_frames(wav.tags, meta, cam, djay_key_display, comment_str)
                wav.save()
                return True
            except Exception:
                return False

        # 3. AIFF
        elif suffix in (".aif", ".aiff"):
            try:
                aiff = AIFF(file_path)
                if aiff.tags is None:
                    aiff.add_tags()
                cls._populate_id3_frames(aiff.tags, meta, cam, djay_key_display, comment_str)
                aiff.save()
                return True
            except Exception:
                return False

        # 4. FLAC
        elif suffix == ".flac":
            try:
                flac = FLAC(file_path)
                flac["TITLE"] = meta.title
                flac["ARTIST"] = meta.artist
                flac["ALBUM"] = meta.album
                flac["GENRE"] = meta.genre

                if meta.year:
                    flac["DATE"] = str(meta.year)
                if meta.bpm > 0:
                    flac["BPM"] = f"{meta.bpm:.2f}"
                if djay_key_display:
                    flac["INITIALKEY"] = djay_key_display
                    flac["KEY"] = djay_key_display
                    flac["CAMELOT"] = cam or ""
                if meta.energy > 0:
                    flac["ENERGY"] = f"{meta.energy:.2f}"
                if meta.rating > 0:
                    flac["RATING"] = str(meta.rating)
                if comment_str:
                    flac["COMMENT"] = comment_str

                flac.save()
                return True
            except Exception:
                return False

        # 5. MP4 / M4A / AAC
        elif suffix in (".m4a", ".mp4", ".aac"):
            try:
                mp4 = MP4(file_path)
                if mp4.tags is None:
                    mp4.add_tags()

                mp4.tags["\xa9nam"] = [meta.title]
                mp4.tags["\xa9ART"] = [meta.artist]
                mp4.tags["\xa9alb"] = [meta.album]
                mp4.tags["\xa9gen"] = [meta.genre]

                if meta.year:
                    mp4.tags["\xa9day"] = [str(meta.year)]
                if meta.bpm > 0:
                    mp4.tags["tmpo"] = [int(round(meta.bpm))]

                if djay_key_display:
                    mp4.tags["----:com.apple.iTunes:INITIALKEY"] = [djay_key_display.encode("utf-8")]
                    mp4.tags["----:com.apple.iTunes:CAMELOT"] = [(cam or "").encode("utf-8")]
                if meta.energy > 0:
                    mp4.tags["----:com.apple.iTunes:ENERGY"] = [f"{meta.energy:.2f}".encode("utf-8")]

                if comment_str:
                    mp4.tags["\xa9cmt"] = [comment_str]

                mp4.save()
                return True
            except Exception:
                return False

        return False

    @staticmethod
    def _populate_id3_frames(tags: ID3, meta: TrackMetadata, cam: str, djay_key_display: str, comment_str: str) -> None:
        tags["TIT2"] = TIT2(encoding=3, text=meta.title)
        tags["TPE1"] = TPE1(encoding=3, text=meta.artist)
        tags["TALB"] = TALB(encoding=3, text=meta.album)
        tags["TCON"] = TCON(encoding=3, text=meta.genre)

        if meta.year:
            tags["TDRC"] = TDRC(encoding=3, text=str(meta.year))

        if meta.bpm > 0:
            tags["TBPM"] = TBPM(encoding=3, text=f"{meta.bpm:.2f}")

        if djay_key_display:
            tags["TKEY"] = TKEY(encoding=3, text=djay_key_display)
            tags.add(TXXX(encoding=3, desc="Camelot", text=cam or ""))
            tags.add(TXXX(encoding=3, desc="InitialKey", text=djay_key_display))

        if meta.energy > 0:
            tags.add(TXXX(encoding=3, desc="EnergyLevel", text=f"{meta.energy:.2f}"))

        if meta.rating > 0:
            popm_val = {1: 1, 2: 64, 3: 128, 4: 196, 5: 255}.get(meta.rating, 128)
            tags["POPM"] = POPM(email="djay@sonicdj.ai", rating=popm_val, count=0)

        if comment_str:
            tags["COMM"] = COMM(encoding=3, lang="eng", desc="", text=comment_str)
