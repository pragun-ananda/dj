import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Optional, Dict, Any

from sonicdj.db.repository import DatabaseManager, TrackRepository
from sonicdj.search.hybrid_engine import HybridSearchEngine
from sonicdj.hardware.flx4 import FLX4Controller
from sonicdj.copilot.live_engine import LiveCopilotEngine


class SonicDJServerHandler(BaseHTTPRequestHandler):
    db: Optional[DatabaseManager] = None
    flx4: Optional[FLX4Controller] = None
    copilot: Optional[LiveCopilotEngine] = None

    def _send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/api/tracks":
            repo = TrackRepository(self.db)
            tracks, count = repo.list_tracks(limit=100)
            res = [
                {
                    "id": t.id,
                    "title": t.title,
                    "artist": t.artist,
                    "bpm": t.bpm,
                    "camelot": t.camelot,
                    "energy": t.energy,
                    "genre": t.genre,
                }
                for t in tracks
            ]
            self._send_json({"tracks": res, "total": count})

        elif parsed.path == "/api/search":
            q = params.get("q", [None])[0]
            key = params.get("key", [None])[0]
            bpm_str = params.get("bpm", [None])[0]
            bpm = float(bpm_str) if bpm_str else None

            engine = HybridSearchEngine(self.db)
            results = engine.search(prompt=q, target_camelot=key, target_bpm=bpm, limit=10)
            out = [
                {
                    "id": r.track.id,
                    "title": r.track.title,
                    "artist": r.track.artist,
                    "bpm": r.track.bpm,
                    "camelot": r.track.camelot,
                    "energy": r.track.energy,
                    "match_pct": int(r.composite_score * 100),
                    "subgenre": r.subgenre_info.primary_subgenre,
                    "mix_advice": r.mix_recommendation,
                }
                for r in results
            ]
            self._send_json({"results": out})

        elif parsed.path == "/api/copilot":
            hud = self.copilot.get_hud_state()
            self._send_json(hud.__dict__)

        elif parsed.path == "/" or parsed.path == "/index.html":
            html = self._get_html_dashboard()
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        if parsed.path == "/api/copilot/load":
            deck = int(payload.get("deck", 1))
            track_id = int(payload.get("track_id", 1))
            self.copilot.load_deck_track(deck, track_id)
            self._send_json({"status": "loaded", "deck": deck, "track_id": track_id})

        elif parsed.path == "/api/flx4/event":
            crossfader = float(payload.get("crossfader", 0.5))
            d1_vol = float(payload.get("deck1_vol", 1.0))
            d2_vol = float(payload.get("deck2_vol", 1.0))
            self.flx4.update_faders(crossfader, d1_vol, d2_vol)
            self._send_json({"status": "updated", "master_deck": self.flx4.state.master_deck})

        else:
            self.send_error(404, "Endpoint not found")

    def _get_html_dashboard(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SonicDJ — AI DJ Workstation & Live Co-Pilot</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0b0f19; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .gradient-deck { background: linear-gradient(180deg, #161e2e 0%, #0d1321 100%); }
    </style>
</head>
<body class="p-6">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Top Bar -->
        <header class="flex justify-between items-center bg-gray-900/80 p-4 rounded-xl border border-gray-800 backdrop-blur">
            <div class="flex items-center space-x-3">
                <div class="w-4 h-4 rounded-full bg-cyan-500 animate-pulse"></div>
                <h1 class="text-2xl font-black tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-500">SONICDJ</h1>
                <span class="text-xs px-2.5 py-1 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800 font-semibold">AI CO-PILOT ACTIVE</span>
            </div>
            <div class="flex items-center space-x-4 text-xs">
                <div class="flex items-center space-x-2 bg-gray-800 px-3 py-1.5 rounded-lg">
                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
                    <span class="text-gray-300 font-medium">Pioneer DDJ-FLX4 Connected</span>
                </div>
                <div class="bg-gray-800 px-3 py-1.5 rounded-lg text-gray-300 font-medium">
                    djay Pro Sync: <span class="text-emerald-400">Ready</span>
                </div>
            </div>
        </header>

        <!-- Search Bar -->
        <div class="relative">
            <input type="text" id="searchInput" placeholder="Search by vibe, mood, or prompt (e.g. 'dark hypnotic afrohouse with heavy rolling bassline')..." 
                class="w-full bg-gray-900/90 border border-gray-700 rounded-xl px-5 py-4 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 shadow-xl text-lg">
            <div class="absolute right-4 top-4 text-gray-400 text-sm">Press Enter to Search</div>
        </div>

        <!-- Main Decks Layout -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Deck 1 -->
            <div class="gradient-deck rounded-2xl p-6 border border-cyan-900/40 shadow-2xl relative">
                <div class="flex justify-between items-center mb-4">
                    <span class="text-sm font-bold text-cyan-400 uppercase tracking-wider">Deck 1 (Master)</span>
                    <span class="px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 text-xs font-mono font-bold">KEY 8A | 123.0 BPM</span>
                </div>
                <h2 class="text-xl font-bold text-white mb-1" id="d1-title">Black Coffee - Subconsciously</h2>
                <p class="text-sm text-gray-400 mb-4" id="d1-genre">Afro House • Energy: 85%</p>
                <div class="h-20 bg-gray-950/80 rounded-lg flex items-center justify-center border border-gray-800 relative overflow-hidden mb-4">
                    <div class="absolute inset-y-0 left-0 bg-cyan-500/20 w-3/5"></div>
                    <div class="absolute left-3/5 top-0 bottom-0 w-0.5 bg-cyan-400"></div>
                    <span class="text-xs text-cyan-300 font-mono z-10">[Pad A: Intro] • [Pad B: Vocal 15s] • [Pad C: Main Drop 45s]</span>
                </div>
            </div>

            <!-- Deck 2 -->
            <div class="gradient-deck rounded-2xl p-6 border border-indigo-900/40 shadow-2xl relative">
                <div class="flex justify-between items-center mb-4">
                    <span class="text-sm font-bold text-indigo-400 uppercase tracking-wider">Deck 2 (Cue)</span>
                    <span class="px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 text-xs font-mono font-bold">KEY 9A | 123.0 BPM</span>
                </div>
                <h2 class="text-xl font-bold text-white mb-1" id="d2-title">THEMBA - Sound of Freedom</h2>
                <p class="text-sm text-gray-400 mb-4" id="d2-genre">Afro House • Energy: 90%</p>
                <div class="h-20 bg-gray-950/80 rounded-lg flex items-center justify-center border border-gray-800 relative overflow-hidden mb-4">
                    <div class="absolute inset-y-0 left-0 bg-indigo-500/20 w-1/4"></div>
                    <div class="absolute left-1/4 top-0 bottom-0 w-0.5 bg-indigo-400"></div>
                    <span class="text-xs text-indigo-300 font-mono z-10">[Pad A: Beat 1] • [Pad B: Main Drop 32s]</span>
                </div>
            </div>
        </div>

        <!-- Live Co-Pilot Next Track Suggestions -->
        <div class="bg-gray-900/80 rounded-2xl p-6 border border-gray-800">
            <h3 class="text-lg font-bold text-white mb-4 flex items-center space-x-2">
                <span>⚡ Live Co-Pilot Mix Recommendations</span>
                <span class="text-xs font-normal text-gray-400">(Harmonically Compatible with Deck 1)</span>
            </h3>
            <div class="space-y-3" id="suggestionsContainer">
                <div class="flex items-center justify-between p-3.5 bg-gray-800/60 rounded-xl hover:bg-gray-800 transition">
                    <div class="flex items-center space-x-4">
                        <span class="text-emerald-400 font-bold font-mono">96%</span>
                        <div>
                            <div class="font-bold text-white">MK - Burning Piano</div>
                            <div class="text-xs text-gray-400">Vocal House • Camelot: 8B • 124.0 BPM</div>
                        </div>
                    </div>
                    <span class="text-xs text-yellow-400 italic">Relative Major/Minor Mix (8A -> 8B)</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""


class SonicDJServer:
    def __init__(self, db: DatabaseManager, port: int = 8000):
        self.db = db
        self.port = port
        self.flx4 = FLX4Controller()
        self.copilot = LiveCopilotEngine(db, self.flx4)
        
        # Inject into Handler class
        SonicDJServerHandler.db = self.db
        SonicDJServerHandler.flx4 = self.flx4
        SonicDJServerHandler.copilot = self.copilot
        self.httpd = HTTPServer(("127.0.0.1", self.port), SonicDJServerHandler)

    def start_background(self) -> Thread:
        """Starts server in a background thread."""
        t = Thread(target=self.httpd.serve_forever, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
