from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from sonicdj.db.schema import Track
from sonicdj.db.repository import DatabaseManager, TrackRepository
from sonicdj.search.hybrid_engine import HybridSearchEngine, SearchResult
from sonicdj.sandbox.transition_graph import TransitionGraphEngine, TransitionEvaluation
from sonicdj.hardware.flx4 import FLX4Controller, FLX4State


@dataclass
class LiveCoPilotHUD:
    active_master_deck: int
    deck1_track: Optional[Dict[str, Any]]
    deck2_track: Optional[Dict[str, Any]]
    active_warnings: List[str]
    suggested_next_tracks: List[Dict[str, Any]]
    hardware_state: Dict[str, Any]


class LiveCopilotEngine:
    """
    Real-time Live Set Co-Pilot and HUD Engine.
    Passively monitors active DDJ-FLX4 hardware faders and djay Pro playback state,
    providing instant harmonic next-track recommendations and live collision alerts.
    """

    def __init__(self, db: DatabaseManager, flx4: Optional[FLX4Controller] = None):
        self.db = db
        self.repo = TrackRepository(db)
        self.search_engine = HybridSearchEngine(db)
        self.graph_engine = TransitionGraphEngine(db)
        self.flx4 = flx4 or FLX4Controller()

        # Live Deck State
        self.deck1_track_id: Optional[int] = None
        self.deck2_track_id: Optional[int] = None
        self.deck1_pos_sec: float = 0.0
        self.deck2_pos_sec: float = 0.0

    def load_deck_track(self, deck: int, track_id: int) -> None:
        """Assigns a track to Deck 1 or Deck 2."""
        if deck == 1:
            self.deck1_track_id = track_id
        elif deck == 2:
            self.deck2_track_id = track_id

    def update_deck_pos(self, deck: int, pos_sec: float) -> None:
        if deck == 1:
            self.deck1_pos_sec = max(0.0, pos_sec)
        elif deck == 2:
            self.deck2_pos_sec = max(0.0, pos_sec)

    def _get_active_master_track(self) -> Optional[Track]:
        master_deck = self.flx4.state.master_deck
        target_id = self.deck1_track_id if master_deck == 1 else self.deck2_track_id
        if target_id is None:
            # Fallback to whichever deck is loaded
            target_id = self.deck1_track_id or self.deck2_track_id
        if target_id:
            return self.repo.get_track_by_id(target_id)
        return None

    def check_warnings(self, track_a: Optional[Track], track_b: Optional[Track]) -> List[str]:
        """Detects real-time mixing hazards (vocal clashes, key clashes, bass overload)."""
        warnings = []
        if not track_a or not track_b:
            return warnings

        # 1. Low EQ Bass Collision
        if (
            self.flx4.state.deck1_volume > 0.5
            and self.flx4.state.deck2_volume > 0.5
            and self.flx4.state.deck1_eq_low > 0.6
            and self.flx4.state.deck2_eq_low > 0.6
        ):
            warnings.append("⚠️ Bass Overload: Both Low-EQ knobs active simultaneously! Swap low faders.")

        # 2. Harmonic Clash
        if track_a.camelot and track_b.camelot:
            score, _ = HybridSearchEngine.compute_harmonic_compatibility(track_a.camelot, track_b.camelot)
            if score <= 0.25:
                warnings.append(f"⚠️ Dissonant Key Clash: {track_a.camelot} into {track_b.camelot} will clash harmonically.")

        # 3. Vocal Overlap Warning
        comm_a = (track_a.comments or "").lower()
        comm_b = (track_b.comments or "").lower()
        if "vocal" in comm_a and "vocal" in comm_b:
            if self.flx4.state.deck1_volume > 0.6 and self.flx4.state.deck2_volume > 0.6:
                warnings.append("⚠️ Vocal Collision Alert: Simultaneous vocal phrases detected on both decks.")

        return warnings

    def get_hud_state(self, limit_suggestions: int = 5) -> LiveCoPilotHUD:
        """Constructs complete live Co-Pilot HUD telemetry."""
        track1 = self.repo.get_track_by_id(self.deck1_track_id) if self.deck1_track_id else None
        track2 = self.repo.get_track_by_id(self.deck2_track_id) if self.deck2_track_id else None

        master_track = self._get_active_master_track()
        warnings = self.check_warnings(track1, track2)

        # Generate harmonic next-track suggestions based on active master track
        suggestions = []
        if master_track:
            results = self.search_engine.search(
                reference_track_id=master_track.id,
                target_camelot=master_track.camelot,
                target_bpm=master_track.bpm,
                bpm_tolerance=6.0,
                limit=limit_suggestions,
            )
            for r in results:
                t = r.track
                suggestions.append({
                    "id": t.id,
                    "title": t.title,
                    "artist": t.artist,
                    "camelot": t.camelot,
                    "bpm": round(t.bpm or 124.0, 1),
                    "energy": int((t.energy or 0.7) * 100),
                    "match_pct": int(r.composite_score * 100),
                    "subgenre": r.subgenre_info.primary_subgenre,
                    "mix_advice": r.mix_recommendation,
                })

        def track_to_dict(t: Optional[Track], pos_sec: float) -> Optional[Dict[str, Any]]:
            if not t:
                return None
            return {
                "id": t.id,
                "title": t.title,
                "artist": t.artist,
                "camelot": t.camelot,
                "bpm": round(t.bpm or 124.0, 1),
                "energy": int((t.energy or 0.7) * 100),
                "duration_sec": t.duration_sec,
                "position_sec": round(pos_sec, 1),
                "cues": [{"name": c.name, "time_ms": c.timestamp_ms, "color": c.color_hex} for c in (t.cues or [])],
            }

        hw_state = {
            "crossfader": self.flx4.state.crossfader,
            "deck1_vol": self.flx4.state.deck1_volume,
            "deck2_vol": self.flx4.state.deck2_volume,
            "deck1_eq_low": self.flx4.state.deck1_eq_low,
            "deck2_eq_low": self.flx4.state.deck2_eq_low,
            "master_deck": self.flx4.state.master_deck,
        }

        return LiveCoPilotHUD(
            active_master_deck=self.flx4.state.master_deck,
            deck1_track=track_to_dict(track1, self.deck1_pos_sec),
            deck2_track=track_to_dict(track2, self.deck2_pos_sec),
            active_warnings=warnings,
            suggested_next_tracks=suggestions,
            hardware_state=hw_state,
        )
