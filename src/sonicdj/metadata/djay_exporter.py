import urllib.parse
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

from sonicdj.db.schema import Track, CuePoint, Playlist


class DjayProExporter:
    """Exports library tracks, playlists, and cue points to djay Pro compatible XML and M3U8 formats."""

    @staticmethod
    def _file_path_to_uri(file_path: str) -> str:
        """Converts absolute local path to file://localhost URI standard."""
        abs_path = Path(file_path).resolve()
        return "file://localhost" + urllib.parse.quote(str(abs_path))

    @classmethod
    def export_rekordbox_xml(
        cls,
        tracks: List[Track],
        output_file: Path,
        playlists: Optional[List[Playlist]] = None,
    ) -> Path:
        """
        Generates standard Rekordbox XML (<DJ_PLAYLISTS>) natively importable into Algoriddim djay Pro.
        Includes full track metadata, Camelot keys, BPMs, and Hot Cues / Memory Cues.
        """
        output_file = Path(output_file).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
        ET.SubElement(
            root,
            "PRODUCT",
            Name="SonicDJ",
            Version="0.1.0",
            Company="SonicDJ AI",
        )

        # 1. COLLECTION
        collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks)))

        for idx, track in enumerate(tracks, start=1):
            fmt = (track.format or "mp3").upper()
            track_elem = ET.SubElement(
                collection,
                "TRACK",
                TrackID=str(track.id or idx),
                Name=track.title or "Unknown Title",
                Artist=track.artist or "Unknown Artist",
                Album=track.album or "",
                Genre=track.genre or "Electronic",
                Kind=f"{fmt} audio file",
                Size=str(track.file_size_bytes or 0),
                TotalTime=str(int(round(track.duration_sec or 0))),
                AverageBpm=f"{track.bpm:.2f}" if track.bpm else "0.00",
                Tonality=track.camelot or track.key_raw or "",
                Rating=str(min(5, max(0, track.rating or 0))),
                Comments=track.comments or "",
                Location=cls._file_path_to_uri(track.file_path),
            )

            # Hot Cues and Structural Markers
            if track.cues:
                for cue_idx, cue in enumerate(track.cues):
                    start_sec = cue.timestamp_ms / 1000.0
                    hot_cue_num = cue.hot_cue_index if cue.hot_cue_index is not None else cue_idx
                    ET.SubElement(
                        track_elem,
                        "POSITION_MARK",
                        Name=cue.name or f"Cue {cue_idx + 1}",
                        Type="0",  # 0 = Hot Cue / Memory Cue
                        Start=f"{start_sec:.3f}",
                        Num=str(hot_cue_num),
                        Red="0",
                        Green="255",
                        Blue="204",
                    )

        # 2. PLAYLISTS
        playlists_root = ET.SubElement(root, "PLAYLISTS")
        root_node = ET.SubElement(
            playlists_root, "NODE", Type="0", Name="ROOT", Count=str(len(playlists or []))
        )

        if playlists:
            for pl in playlists:
                pl_node = ET.SubElement(
                    root_node,
                    "NODE",
                    Name=pl.name,
                    Type="1",  # 1 = Playlist
                    KeyType="0",
                    Entries=str(len(pl.tracks) if hasattr(pl, "tracks") and pl.tracks else 0),
                )
                if hasattr(pl, "tracks") and pl.tracks:
                    for pt in pl.tracks:
                        ET.SubElement(pl_node, "TRACK", Key=str(pt.track_id))

        # Pretty print XML string
        xml_str = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(
            indent="  ", encoding="utf-8"
        )
        with open(output_file, "wb") as f:
            f.write(xml_str)

        return output_file

    @classmethod
    def export_m3u8(cls, tracks: List[Track], output_file: Path, playlist_name: str = "SonicDJ Crate") -> Path:
        """Exports extended M3U8 playlist with rich metadata comments."""
        output_file = Path(output_file).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        lines = ["#EXTM3U", f"#PLAYLIST:{playlist_name}"]

        for track in tracks:
            duration = int(round(track.duration_sec or 0))
            display_title = f"{track.artist} - {track.title}"
            if track.camelot or track.bpm:
                display_title += f" [{track.camelot} | {track.bpm:.1f} BPM]"

            lines.append(f"#EXTINF:{duration},{display_title}")
            lines.append(f"#EXT-X-DJAY-KEY:{track.camelot or track.key_raw}")
            lines.append(f"#EXT-X-DJAY-BPM:{track.bpm:.2f}")
            lines.append(str(Path(track.file_path).resolve()))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return output_file
