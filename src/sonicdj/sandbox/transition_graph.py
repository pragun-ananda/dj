import heapq
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

from sonicdj.db.schema import Track
from sonicdj.db.repository import DatabaseManager, TrackRepository
from sonicdj.search.hybrid_engine import HybridSearchEngine


@dataclass
class TransitionEvaluation:
    track_a: Track
    track_b: Track
    overall_compatibility: float  # 0.0 to 1.0
    harmonic_score: float
    bpm_score: float
    energy_score: float
    vocal_safety_score: float
    recommended_bars: int         # 8, 16, or 32 bars
    transition_style: str         # "Bass Swap Crossfade", "High-Pass Filter Blend", "Drop Cut"
    mix_out_time_sec: float
    mix_in_time_sec: float
    explanation: str


@dataclass
class SetListStep:
    step_number: int
    track: Track
    transition_from_prev: Optional[TransitionEvaluation]


class TransitionGraphEngine:
    """
    Constructs a directed harmonic compatibility graph across audio tracks
    and finds optimal DJ setlist progression paths using energy ramp heuristics.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.repo = TrackRepository(db)

    @classmethod
    def evaluate_transition(cls, track_a: Track, track_b: Track) -> TransitionEvaluation:
        """
        Computes a comprehensive multi-factor DJ transition quality evaluation.
        """
        # 1. Harmonic Compatibility
        harm_score, harm_tip = HybridSearchEngine.compute_harmonic_compatibility(track_a.camelot, track_b.camelot)

        # 2. BPM Proximity
        bpm_a = track_a.bpm or 124.0
        bpm_b = track_b.bpm or 124.0
        bpm_diff = abs(bpm_b - bpm_a)
        bpm_score = float(max(0.0, 1.0 - (bpm_diff / 10.0)))

        # 3. Energy Flow Score
        e_a = track_a.energy or 0.7
        e_b = track_b.energy or 0.7
        e_diff = e_b - e_a  # Positive is building energy, negative is dropping
        if -0.15 <= e_diff <= 0.25:
            energy_score = 1.0  # Smooth progression
        elif e_diff > 0.25:
            energy_score = 0.85  # Big energy jump
        else:
            energy_score = 0.70  # Energy reset

        # 4. Vocal Safety Score
        comments_a = (track_a.comments or "").lower()
        comments_b = (track_b.comments or "").lower()
        vocal_a = "vocal" in comments_a and "instrumental" not in comments_a
        vocal_b = "vocal" in comments_b and "instrumental" not in comments_b

        if vocal_a and vocal_b:
            vocal_safety_score = 0.75  # Potential vocal clash if mixed during vocal sections
            transition_style = "Bass Swap Crossfade (Wait for vocal break)"
        elif not vocal_b:
            vocal_safety_score = 1.0   # Instrumental B is 100% safe to mix under vocals
            transition_style = "High-Pass Filter Blend"
        else:
            vocal_safety_score = 0.95
            transition_style = "Bass Swap Crossfade"

        # 5. Composite Compatibility Score
        overall = round(
            (0.40 * harm_score) + (0.30 * bpm_score) + (0.15 * energy_score) + (0.15 * vocal_safety_score), 3
        )

        # Mix points (cues)
        mix_out = max(0.0, (track_a.duration_sec or 180.0) - 30.0)
        mix_in = 0.0

        try:
            cues_a = track_a.cues
            if cues_a:
                outro_cue = next((c for c in cues_a if "outro" in c.name.lower()), None)
                if outro_cue:
                    mix_out = outro_cue.timestamp_ms / 1000.0
        except Exception:
            pass

        try:
            cues_b = track_b.cues
            if cues_b:
                intro_cue = next((c for c in cues_b if "intro" in c.name.lower()), None)
                if intro_cue:
                    mix_in = intro_cue.timestamp_ms / 1000.0
        except Exception:
            pass

        recommended_bars = 16 if bpm_diff < 4.0 else 8

        explanation = f"{harm_tip} | BPM: {bpm_a:.1f} -> {bpm_b:.1f} (Δ{bpm_diff:.1f}) | {transition_style}"

        return TransitionEvaluation(
            track_a=track_a,
            track_b=track_b,
            overall_compatibility=overall,
            harmonic_score=round(harm_score, 2),
            bpm_score=round(bpm_score, 2),
            energy_score=round(energy_score, 2),
            vocal_safety_score=round(vocal_safety_score, 2),
            recommended_bars=recommended_bars,
            transition_style=transition_style,
            mix_out_time_sec=round(mix_out, 2),
            mix_in_time_sec=round(mix_in, 2),
            explanation=explanation,
        )

    def find_optimal_set_path(
        self,
        start_track_id: int,
        target_track_id: int,
        max_hops: int = 5,
    ) -> List[SetListStep]:
        """
        Finds the smoothest harmonic and energy-coherent path of tracks
        from start_track to target_track using Dijkstra's shortest path algorithm.
        """
        all_tracks, _ = self.repo.list_tracks(limit=1000)
        track_map = {t.id: t for t in all_tracks}

        if start_track_id not in track_map or target_track_id not in track_map:
            return []

        start_track = track_map[start_track_id]
        target_track = track_map[target_track_id]

        if start_track_id == target_track_id:
            return [SetListStep(step_number=1, track=start_track, transition_from_prev=None)]

        # Dijkstra priority queue: (cumulative_cost, current_track_id, path_list)
        pq = [(0.0, start_track_id, [start_track_id])]
        visited: Set[int] = set()

        while pq:
            cost, curr_id, path = heapq.heappop(pq)

            if curr_id == target_track_id:
                # Reconstruct path into SetListStep objects
                steps = []
                for i, tid in enumerate(path):
                    t = track_map[tid]
                    if i == 0:
                        steps.append(SetListStep(step_number=1, track=t, transition_from_prev=None))
                    else:
                        prev_t = track_map[path[i - 1]]
                        trans = self.evaluate_transition(prev_t, t)
                        steps.append(SetListStep(step_number=i + 1, track=t, transition_from_prev=trans))
                return steps

            if curr_id in visited or len(path) > max_hops:
                continue
            visited.add(curr_id)

            curr_track = track_map[curr_id]

            for next_track in all_tracks:
                if next_track.id in visited:
                    continue

                trans = self.evaluate_transition(curr_track, next_track)
                # Cost is inverted compatibility (lower cost = better transition)
                edge_cost = max(0.01, 1.0 - trans.overall_compatibility)

                # Penalize large BPM jumps (>8 BPM)
                if abs((next_track.bpm or 124.0) - (curr_track.bpm or 124.0)) > 8.0:
                    edge_cost += 5.0

                heapq.heappush(pq, (cost + edge_cost, next_track.id, path + [next_track.id]))

        # Fallback direct path if no intermediate path found
        direct_trans = self.evaluate_transition(start_track, target_track)
        return [
            SetListStep(step_number=1, track=start_track, transition_from_prev=None),
            SetListStep(step_number=2, track=target_track, transition_from_prev=direct_trans),
        ]
