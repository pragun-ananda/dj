import json
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
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
        .gradient-deck-1 { background: linear-gradient(180deg, #132238 0%, #0c1524 100%); }
        .gradient-deck-2 { background: linear-gradient(180deg, #231838 0%, #150d24 100%); }
        .waveform-bar { transition: height 0.2s ease; }
    </style>
</head>
<body class="p-6 min-h-screen">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Top Bar -->
        <header class="flex justify-between items-center bg-gray-900/90 p-4 rounded-2xl border border-gray-800 backdrop-blur shadow-2xl">
            <div class="flex items-center space-x-3">
                <div class="w-3.5 h-3.5 rounded-full bg-cyan-400 animate-ping"></div>
                <h1 class="text-2xl font-black tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-indigo-400 to-purple-500">SONICDJ</h1>
                <span class="text-xs px-2.5 py-1 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-700/50 font-semibold tracking-wide">LIVE CO-PILOT ACTIVE</span>
            </div>
            <div class="flex items-center space-x-4 text-xs">
                <div class="flex items-center space-x-2 bg-gray-800/80 border border-gray-700 px-3 py-1.5 rounded-xl">
                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
                    <span class="text-gray-200 font-medium">Pioneer DDJ-FLX4 Connected</span>
                </div>
                <div class="bg-gray-800/80 border border-gray-700 px-3 py-1.5 rounded-xl text-gray-200 font-medium">
                    Algoriddim djay Pro: <span class="text-emerald-400">Synced</span>
                </div>
            </div>
        </header>

        <!-- Warnings Banner Area -->
        <div id="warningsBanner" class="hidden space-y-2"></div>

        <!-- Search Bar -->
        <div class="relative">
            <input type="text" id="searchInput" placeholder="🔍 Search library by vibe, prompt, mood (e.g. 'dark hypnotic afrohouse', 'uplifting vocal house', '126 bpm techno')..." 
                class="w-full bg-gray-900/90 border border-gray-700 rounded-2xl px-6 py-4 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 shadow-2xl text-lg transition">
        </div>

        <!-- Main Decks Layout -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Deck 1 -->
            <div class="gradient-deck-1 rounded-2xl p-6 border border-cyan-900/50 shadow-2xl relative" id="deck1Card">
                <div class="flex justify-between items-center mb-3">
                    <div class="flex items-center space-x-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
                        <span class="text-xs font-bold text-cyan-400 uppercase tracking-widest" id="d1-badge">DECK 1 (MASTER)</span>
                    </div>
                    <span class="px-2.5 py-1 rounded-lg bg-cyan-950 text-cyan-300 text-xs font-mono font-bold border border-cyan-800" id="d1-key-bpm">8A • 123.0 BPM</span>
                </div>
                <h2 class="text-2xl font-black text-white truncate mb-1" id="d1-title">Black Coffee - Subconsciously</h2>
                <p class="text-xs text-gray-400 mb-4" id="d1-genre">Afro House • Energy: 85% • Instrumental Intro: 16 Bars</p>
                
                <!-- Waveform Canvas -->
                <div class="h-24 bg-gray-950 rounded-xl flex items-end justify-between px-3 py-2 border border-cyan-950 relative overflow-hidden mb-4">
                    <div class="absolute inset-0 bg-cyan-500/10 w-2/3"></div>
                    <div class="absolute left-2/3 top-0 bottom-0 w-0.5 bg-cyan-400 z-10 shadow-[0_0_8px_#22d3ee]"></div>
                    <div class="absolute top-2 left-3 text-[10px] text-cyan-300 font-mono z-10">[Pad A: Intro 0.0s] [Pad B: Vocal 15.0s] [Pad C: Drop 45.0s] [Pad D: Outro]</div>
                    <!-- Waveform Bars Mock -->
                    <div class="w-1.5 bg-cyan-500/60 h-8 rounded-t"></div>
                    <div class="w-1.5 bg-cyan-500/70 h-12 rounded-t"></div>
                    <div class="w-1.5 bg-cyan-500/80 h-16 rounded-t"></div>
                    <div class="w-1.5 bg-cyan-400 h-20 rounded-t"></div>
                    <div class="w-1.5 bg-cyan-400 h-14 rounded-t"></div>
                    <div class="w-1.5 bg-cyan-500/80 h-18 rounded-t"></div>
                    <div class="w-1.5 bg-cyan-400 h-22 rounded-t"></div>
                    <div class="w-1.5 bg-cyan-500/70 h-12 rounded-t"></div>
                    <div class="w-1.5 bg-cyan-500/60 h-10 rounded-t"></div>
                </div>
            </div>

            <!-- Deck 2 -->
            <div class="gradient-deck-2 rounded-2xl p-6 border border-purple-900/50 shadow-2xl relative" id="deck2Card">
                <div class="flex justify-between items-center mb-3">
                    <div class="flex items-center space-x-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-purple-400"></span>
                        <span class="text-xs font-bold text-purple-400 uppercase tracking-widest" id="d2-badge">DECK 2 (CUE)</span>
                    </div>
                    <span class="px-2.5 py-1 rounded-lg bg-purple-950 text-purple-300 text-xs font-mono font-bold border border-purple-800" id="d2-key-bpm">9A • 123.0 BPM</span>
                </div>
                <h2 class="text-2xl font-black text-white truncate mb-1" id="d2-title">THEMBA - Sound of Freedom</h2>
                <p class="text-xs text-gray-400 mb-4" id="d2-genre">Afro House • Energy: 90% • Vocal Verse: 32 Bars</p>
                
                <!-- Waveform Canvas -->
                <div class="h-24 bg-gray-950 rounded-xl flex items-end justify-between px-3 py-2 border border-purple-950 relative overflow-hidden mb-4">
                    <div class="absolute inset-0 bg-purple-500/10 w-1/4"></div>
                    <div class="absolute left-1/4 top-0 bottom-0 w-0.5 bg-purple-400 z-10 shadow-[0_0_8px_#c084fc]"></div>
                    <div class="absolute top-2 left-3 text-[10px] text-purple-300 font-mono z-10">[Pad A: Intro 0.0s] [Pad B: Main Drop 32.0s]</div>
                    <!-- Waveform Bars Mock -->
                    <div class="w-1.5 bg-purple-500/60 h-6 rounded-t"></div>
                    <div class="w-1.5 bg-purple-500/70 h-10 rounded-t"></div>
                    <div class="w-1.5 bg-purple-500/80 h-14 rounded-t"></div>
                    <div class="w-1.5 bg-purple-400 h-18 rounded-t"></div>
                    <div class="w-1.5 bg-purple-400 h-22 rounded-t"></div>
                    <div class="w-1.5 bg-purple-500/80 h-16 rounded-t"></div>
                    <div class="w-1.5 bg-purple-400 h-14 rounded-t"></div>
                    <div class="w-1.5 bg-purple-500/70 h-8 rounded-t"></div>
                    <div class="w-1.5 bg-purple-500/60 h-6 rounded-t"></div>
                </div>
            </div>
        </div>

        <!-- Hardware Fader Controls Simulation -->
        <div class="bg-gray-900/80 rounded-2xl p-5 border border-gray-800 shadow-xl flex flex-wrap items-center justify-between gap-4">
            <div class="flex items-center space-x-4">
                <span class="text-xs font-bold text-gray-400 uppercase">FLX4 Crossfader:</span>
                <input type="range" id="crossfaderSlider" min="0" max="100" value="50" class="w-48 accent-cyan-400 cursor-pointer">
                <span class="text-xs font-mono text-cyan-400" id="crossfaderVal">Center (0.50)</span>
            </div>
            <div class="flex items-center space-x-6 text-xs">
                <div>D1 Vol: <input type="range" id="d1Vol" min="0" max="100" value="100" class="w-24 accent-cyan-400 cursor-pointer"></div>
                <div>D2 Vol: <input type="range" id="d2Vol" min="0" max="100" value="100" class="w-24 accent-purple-400 cursor-pointer"></div>
            </div>
        </div>

        <!-- Live Co-Pilot & Search Results -->
        <div class="bg-gray-900/90 rounded-2xl p-6 border border-gray-800 shadow-2xl">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-lg font-bold text-white flex items-center space-x-2">
                    <span>⚡ Live Co-Pilot Harmonic Mix Recommendations</span>
                </h3>
                <span class="text-xs text-gray-400 font-mono">Matched against Active Master Deck</span>
            </div>
            <div class="space-y-3" id="resultsContainer">
                <div class="text-center py-6 text-gray-500 text-sm">Loading library tracks...</div>
            </div>
        </div>
    </div>

    <script>
        async function fetchCopilot() {
            try {
                const res = await fetch('/api/copilot');
                const data = await res.json();
                
                // Update Warnings
                const warnBox = document.getElementById('warningsBanner');
                if (data.active_warnings && data.active_warnings.length > 0) {
                    warnBox.innerHTML = data.active_warnings.map(w => `<div class="p-3 bg-red-950/80 border border-red-800 text-red-300 rounded-xl text-xs font-semibold">${w}</div>`).join('');
                    warnBox.classList.remove('hidden');
                } else {
                    warnBox.classList.add('hidden');
                }

                // Update Suggestions
                const container = document.getElementById('resultsContainer');
                if (data.suggested_next_tracks && data.suggested_next_tracks.length > 0) {
                    container.innerHTML = data.suggested_next_tracks.map(t => `
                        <div class="flex items-center justify-between p-4 bg-gray-800/60 hover:bg-gray-800/90 border border-gray-700/50 rounded-xl transition">
                            <div class="flex items-center space-x-4">
                                <span class="text-emerald-400 font-black font-mono text-base">${t.match_pct}%</span>
                                <div>
                                    <div class="font-bold text-white text-sm">${t.artist} - ${t.title}</div>
                                    <div class="text-xs text-gray-400 font-mono mt-0.5">Key: <span class="text-yellow-400 font-bold">${t.camelot}</span> • ${t.bpm} BPM • Energy: ${t.energy}% • <span class="text-gray-300">${t.subgenre}</span></div>
                                </div>
                            </div>
                            <div class="flex items-center space-x-3">
                                <span class="text-xs text-yellow-300 italic hidden md:inline">${t.mix_advice}</span>
                                <button onclick="loadDeck(1, ${t.id})" class="px-3 py-1.5 bg-cyan-950 hover:bg-cyan-900 border border-cyan-700 text-cyan-300 text-xs font-bold rounded-lg transition">Deck 1</button>
                                <button onclick="loadDeck(2, ${t.id})" class="px-3 py-1.5 bg-purple-950 hover:bg-purple-900 border border-purple-700 text-purple-300 text-xs font-bold rounded-lg transition">Deck 2</button>
                            </div>
                        </div>
                    `).join('');
                }
            } catch (err) {
                console.error(err);
            }
        }

        async function loadDeck(deck, trackId) {
            await fetch('/api/copilot/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ deck, track_id: trackId })
            });
            fetchCopilot();
        }

        async function sendFaderEvent() {
            const cf = parseFloat(document.getElementById('crossfaderSlider').value) / 100.0;
            const d1 = parseFloat(document.getElementById('d1Vol').value) / 100.0;
            const d2 = parseFloat(document.getElementById('d2Vol').value) / 100.0;
            document.getElementById('crossfaderVal').innerText = cf < 0.4 ? `Deck 1 (${cf.toFixed(2)})` : (cf > 0.6 ? `Deck 2 (${cf.toFixed(2)})` : `Center (${cf.toFixed(2)})`);
            
            await fetch('/api/flx4/event', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ crossfader: cf, deck1_vol: d1, deck2_vol: d2 })
            });
            fetchCopilot();
        }

        document.getElementById('crossfaderSlider').addEventListener('input', sendFaderEvent);
        document.getElementById('d1Vol').addEventListener('input', sendFaderEvent);
        document.getElementById('d2Vol').addEventListener('input', sendFaderEvent);

        document.getElementById('searchInput').addEventListener('keyup', async (e) => {
            const query = e.target.value.trim();
            if (query.length > 2) {
                const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
                const data = await res.json();
                const container = document.getElementById('resultsContainer');
                if (data.results && data.results.length > 0) {
                    container.innerHTML = data.results.map(t => `
                        <div class="flex items-center justify-between p-4 bg-gray-800/60 hover:bg-gray-800/90 border border-gray-700/50 rounded-xl transition">
                            <div class="flex items-center space-x-4">
                                <span class="text-cyan-400 font-black font-mono text-base">${t.match_pct}%</span>
                                <div>
                                    <div class="font-bold text-white text-sm">${t.artist} - ${t.title}</div>
                                    <div class="text-xs text-gray-400 font-mono mt-0.5">Key: <span class="text-yellow-400 font-bold">${t.camelot}</span> • ${t.bpm} BPM • <span class="text-gray-300">${t.subgenre}</span></div>
                                </div>
                            </div>
                            <div class="flex items-center space-x-3">
                                <span class="text-xs text-yellow-300 italic hidden md:inline">${t.mix_advice}</span>
                                <button onclick="loadDeck(1, ${t.id})" class="px-3 py-1.5 bg-cyan-950 hover:bg-cyan-900 border border-cyan-700 text-cyan-300 text-xs font-bold rounded-lg transition">Deck 1</button>
                                <button onclick="loadDeck(2, ${t.id})" class="px-3 py-1.5 bg-purple-950 hover:bg-purple-900 border border-purple-700 text-purple-300 text-xs font-bold rounded-lg transition">Deck 2</button>
                            </div>
                        </div>
                    `).join('');
                }
            } else {
                fetchCopilot();
            }
        });

        // Initialize and Poll
        fetchCopilot();
        setInterval(fetchCopilot, 3000);
    </script>
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
        ThreadingHTTPServer.allow_reuse_address = True
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), SonicDJServerHandler)

    def start_background(self) -> Thread:
        """Starts server in a background thread."""
        t = Thread(target=self.httpd.serve_forever, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
