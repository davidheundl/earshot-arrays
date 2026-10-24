"""The live map in a browser: a tiny local HTTP server (stdlib only) and one self-contained page.

    server = LiveMapServer(config, port=8765)   # starts serving in a background thread
    server.publish(state.to_json())             # once per second from the audio loop
    server.url                                   # http://127.0.0.1:8765/

GET /        the page (canvas, polls /state about three times a second)
GET /config  arrays, poses, map radius, classifier, source; sent once
GET /state   the latest tracker state (track.State.to_json), or {} before the first second
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


class LiveMapServer:
    def __init__(self, config: dict, port: int = 8765, host: str = "127.0.0.1"):
        self._lock = threading.Lock()
        self._state: dict = {}
        self._config = config
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    body, kind = PAGE.encode(), "text/html; charset=utf-8"
                elif path == "/config":
                    body, kind = json.dumps(server._config).encode(), "application/json"
                elif path == "/state":
                    with server._lock:
                        body = json.dumps(server._state).encode()
                    kind = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        for p in range(port, port + 20):  # the next free port if this one is taken
            try:
                self._httpd = ThreadingHTTPServer((host, p), Handler)
                break
            except OSError:
                continue
        else:
            raise SystemExit(f"no free port in {port}..{port + 19}")
        self.url = f"http://{host}:{self._httpd.server_address[1]}/"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def publish(self, state: dict) -> None:
        with self._lock:
            self._state = state

    def set_config(self, **kv) -> None:
        with self._lock:
            self._config.update(kv)

    def close(self) -> None:
        self._httpd.shutdown()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Earshot live map</title>
<style>
  :root { --bg:#0d1117; --fg:#e6edf3; --dim:#7d8590; --grid:#21262d; --panel:#161b22; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--fg);
               font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; overflow:hidden; }
  canvas { display:block; width:100vw; height:100vh; }
  #hud { position:absolute; top:24px; left:28px; pointer-events:none; }
  #label { font-size:clamp(34px, 7vw, 84px); font-weight:800; letter-spacing:2px; line-height:1; }
  #prob { font-size:clamp(18px, 2.4vw, 30px); color:var(--dim); margin-top:6px; }
  #bars { margin-top:18px; width:300px; }
  .bar { display:flex; align-items:center; gap:10px; font-size:17px; margin:5px 0; }
  .bar span { width:70px; color:var(--dim); }
  .bar div { height:12px; border-radius:6px; }
  #status { position:absolute; top:24px; right:28px; max-width:40vw; text-align:right; font-size:clamp(12px, 1.5vw, 19px); color:var(--dim); line-height:1.6; }
  #status b { color:var(--fg); }
  #foot { position:absolute; bottom:18px; left:28px; font-size:15px; color:var(--dim); }
  #banner { position:absolute; bottom:18px; right:28px; font-size:18px; color:#e3b341; }
  @media (max-aspect-ratio: 1/1) { #status { top:auto; bottom:72px; right:auto; left:28px; text-align:left; max-width:none; } }
</style></head><body>
<canvas id="c"></canvas>
<div id="hud"><div id="label">–</div><div id="prob">waiting for audio…</div><div id="bars"></div></div>
<div id="status"></div><div id="foot"></div><div id="banner"></div>
<script>
const COLORS = {drone:"#ff4d4f", plane:"#4ea1ff", person:"#3fb950", music:"#bc8cff", car:"#ff9f43",
                fan:"#39c5cf", quiet:"#7d8590", other:"#7d8590", "–":"#7d8590"};
const col = k => COLORS[k] || "#c9d1d9";
const cv = document.getElementById("c"), ctx = cv.getContext("2d");
let cfg = null, st = {}, lastOk = 0;

function resize() { const d = window.devicePixelRatio || 1; cv.width = innerWidth * d; cv.height = innerHeight * d;
                    ctx.setTransform(d, 0, 0, d, 0, 0); draw(); }
addEventListener("resize", resize);

// room metres -> screen pixels: map centred on the arrays, radius cfg.radius fits the shorter side
function view() {
  const xs = cfg.arrays.map(a => a.x), ys = cfg.arrays.map(a => a.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const spanX = Math.max(...xs) - Math.min(...xs), spanY = Math.max(...ys) - Math.min(...ys);
  const R = cfg.radius, w = innerWidth, h = innerHeight;
  const s = Math.min(w / (2 * R + spanX), h / (2 * R + spanY)) * 0.92;
  return {s, R, cx, cy, X: x => w / 2 + (x - cx) * s, Y: y => h / 2 - (y - cy) * s};
}
const dir = b => [Math.sin(b * Math.PI / 180), Math.cos(b * Math.PI / 180)];

function draw() {
  const w = innerWidth, h = innerHeight;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, w, h);
  if (!cfg) return;
  const v = view(), {s, R, X, Y} = v;
  const arr = st.arrays || {};
  const label = st.label || "–", lc = col(label);

  // range rings every metre and bearing ticks every 30 deg: around the array, or the centre of several
  ctx.lineWidth = 1;
  for (const a of (cfg.arrays.length > 1 ? [{x: v.cx, y: v.cy}] : cfg.arrays)) {
    for (let r = 1; r <= Math.ceil(R); r++) {
      ctx.strokeStyle = "#21262d"; ctx.beginPath(); ctx.arc(X(a.x), Y(a.y), r * s, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillStyle = "#484f58"; ctx.font = "13px sans-serif"; ctx.fillText(r + " m", X(a.x) + 4, Y(a.y) - r * s - 4);
    }
    for (let b = 0; b < 360; b += 30) {
      const [dx, dy] = dir(b);
      ctx.strokeStyle = "#1b2027"; ctx.beginPath(); ctx.moveTo(X(a.x), Y(a.y));
      ctx.lineTo(X(a.x + dx * R), Y(a.y + dy * R)); ctx.stroke();
      ctx.fillStyle = "#484f58"; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
      ctx.fillText(b + "°", X(a.x + dx * (R + 0.18)), Y(a.y + dy * (R + 0.18)) + 4); ctx.textAlign = "left";
    }
  }

  // the SRP scan: wedges that light up toward the sound, and its outline
  for (const a of cfg.arrays) {
    const o = arr[a.name]; if (!o || !o.scan || !o.scan.length) continue;
    const on = o.clear && st.clear;
    const n = o.scan.length, step = 360 / n, c = on ? lc : "#7d8590";
    const lo = Math.min(...o.scan), hi = Math.max(...o.scan), sc = o.scan.map(v => (v - lo) / Math.max(hi - lo, 1e-6));
    for (let i = 0; i < n; i++) {
      const v4 = Math.pow(sc[i], 3) * (on ? 0.6 : 0.2);
      if (v4 < 0.01) continue;
      const b0 = i * step, b1 = b0 + step;
      ctx.globalAlpha = v4; ctx.fillStyle = c; ctx.beginPath(); ctx.moveTo(X(a.x), Y(a.y));
      for (let b = b0; b <= b1; b += step / 4) { const [dx, dy] = dir(b); ctx.lineTo(X(a.x + dx * R), Y(a.y + dy * R)); }
      ctx.closePath(); ctx.fill();
    }
    ctx.globalAlpha = 0.9; ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const b = (i % n) * step + step / 2, r = R * (0.12 + 0.45 * sc[i % n]); const [dx, dy] = dir(b);
      i ? ctx.lineTo(X(a.x + dx * r), Y(a.y + dy * r)) : ctx.moveTo(X(a.x + dx * r), Y(a.y + dy * r));
    }
    ctx.stroke(); ctx.globalAlpha = 1;
  }

  // trail: past clear seconds, older ones fainter
  const trail = st.trail || [];
  trail.forEach((p, i) => {
    const age = (trail.length - 1 - i) / Math.max(1, trail.length - 1), alpha = 0.85 - 0.7 * age;
    ctx.globalAlpha = alpha; ctx.fillStyle = col(p.label);
    if (p.fix) { ctx.beginPath(); ctx.arc(X(p.fix[0]), Y(p.fix[1]), 7, 0, 2 * Math.PI); ctx.fill(); }
    else for (const a of cfg.arrays) {
      const b = p.bearings[a.name]; if (b === undefined) continue; const [dx, dy] = dir(b);
      ctx.beginPath(); ctx.arc(X(a.x + dx * R * 0.8), Y(a.y + dy * R * 0.8), 8, 0, 2 * Math.PI); ctx.fill();
    }
  });
  ctx.globalAlpha = 1;

  // current bearing rays
  for (const a of cfg.arrays) {
    const o = arr[a.name]; if (!o) continue; const [dx, dy] = dir(o.bearing), on = o.clear && st.clear;
    ctx.strokeStyle = on ? lc : "#484f58"; ctx.lineWidth = on ? 5 : 2;
    ctx.setLineDash(on ? [] : [8, 8]); ctx.beginPath(); ctx.moveTo(X(a.x), Y(a.y));
    ctx.lineTo(X(a.x + dx * R), Y(a.y + dy * R)); ctx.stroke(); ctx.setLineDash([]);
    if (on && !st.fix) {  // one array: a marker on the ray where the label sits
      ctx.fillStyle = lc; ctx.beginPath(); ctx.arc(X(a.x + dx * R * 0.8), Y(a.y + dy * R * 0.8), 13, 0, 2 * Math.PI); ctx.fill();
    }
  }

  // fix from two arrays, with its 1-sigma circle
  if (st.fix) {
    ctx.fillStyle = lc; ctx.globalAlpha = 0.18; ctx.beginPath();
    ctx.arc(X(st.fix.x), Y(st.fix.y), Math.max(st.fix.sigma * s, 10), 0, 2 * Math.PI); ctx.fill();
    ctx.globalAlpha = 1; ctx.beginPath(); ctx.arc(X(st.fix.x), Y(st.fix.y), 14, 0, 2 * Math.PI); ctx.fill();
  }

  // arrays and north
  for (const a of cfg.arrays) {
    ctx.fillStyle = "#e6edf3"; ctx.fillRect(X(a.x) - 9, Y(a.y) - 9, 18, 18);
    ctx.fillStyle = "#e6edf3"; ctx.font = "bold 16px sans-serif"; ctx.fillText(a.name, X(a.x) + 13, Y(a.y) + 20);
  }
  const nx = w - 70, ny = h - 110;
  ctx.strokeStyle = "#e6edf3"; ctx.fillStyle = "#e6edf3"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(nx, ny + 40); ctx.lineTo(nx, ny); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(nx - 10, ny + 12); ctx.lineTo(nx, ny - 4); ctx.lineTo(nx + 10, ny + 12); ctx.fill();
  ctx.font = "bold 20px sans-serif"; ctx.textAlign = "center"; ctx.fillText("N", nx, ny - 12); ctx.textAlign = "left";
}

function hud() {
  const label = st.label || "–";
  const L = document.getElementById("label"); L.textContent = label.toUpperCase(); L.style.color = col(label);
  document.getElementById("prob").textContent = st.t === undefined ? "waiting for audio…" :
    (label === "–" ? (st.why || "no clear direction") : Math.round(100 * st.prob) + " %");
  const probs = Object.entries(st.probs || {}).sort((a, b) => b[1] - a[1]).slice(0, 5);
  document.getElementById("bars").innerHTML = probs.map(([k, p]) =>
    `<div class="bar"><span>${k}</span><div style="width:${Math.round(220 * p)}px;background:${col(k)}"></div></div>`).join("");
  const arr = st.arrays || {}, lines = [];
  lines.push(`<b>${cfg.source}</b>` + (st.t !== undefined ? ` · t = ${Math.round(st.t)} s` : ""));
  for (const [n, o] of Object.entries(arr))
    lines.push(`${n}: <b>${Math.round(o.bearing)}°</b> · clarity ${o.power.toFixed(2)} · ${Math.round(o.level_db)} dBFS`);
  if (st.fix) lines.push(`fix (${st.fix.x.toFixed(2)}, ${st.fix.y.toFixed(2)}) m ± ${st.fix.sigma.toFixed(2)}`);
  document.getElementById("status").innerHTML = lines.join("<br>");
  document.getElementById("foot").textContent = `classifier: ${cfg.classifier} · audio: ${cfg.audio}` +
    ` · class shown when clarity ≥ ${cfg.gate} (loudest ${Math.round(100 * cfg.loudest)} % of frames), not quiet, and ≥ ${Math.round(100 * cfg.min_prob)} % sure · direction over ${cfg.doa_s} s`;
  const stale = Date.now() - lastOk > 4000;
  document.getElementById("banner").textContent = st.done ? "end of recording" : st.note ? st.note :
    (stale && lastOk ? "no data from the audio loop" : "");
}

async function poll() {
  try {
    if (!cfg) cfg = await (await fetch("/config")).json();
    const s = await (await fetch("/state")).json();
    if (s && s.t !== undefined) { st = s; lastOk = Date.now(); } else if (s && s.done) st = {...st, done: true};
  } catch (e) {}
  if (cfg) { draw(); hud(); }
  setTimeout(poll, 300);
}
resize(); poll();
</script></body></html>
"""


STALL_S = 3.0  # live: no block for this long -> reopen the device
BAD_S = 3  # live: this many seconds in a row of unusable audio -> reopen the device
RATE_WINDOW_S = 6.0  # live: blocks arriving much faster than one per second -> reopen


def block_problem(block: np.ndarray, n_channels: int, n_samples: int) -> str:
    """Why a device block cannot be used, or "" if it is fine."""
    if block.shape != (n_channels, n_samples):
        return f"block shape {block.shape}, expected {(n_channels, n_samples)}"
    if not np.all(np.isfinite(block)):
        return f"{int(np.sum(~np.isfinite(block)))} non-finite samples"
    if float(np.max(np.ptp(block, axis=1))) == 0.0:
        return "all channels constant"
    return ""


def add_parser(sub, live) -> None:
    """Register `arrays live` (called from cli.main; `live` adds --replay/--realtime)."""
    p = live(sub.add_parser("live", help="interactive live map in the browser: direction, scan, class (+ fix)"))
    p.add_argument("--setup", help="setup YAML (default: the replay directory's, else setup.yaml)")
    p.add_argument("--radius", type=float, default=4.0, help="metres shown around the arrays")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--mix", action="store_true", help="classify the mic mean instead of a beam toward the bearing")
    p.add_argument("--doa-s", type=float, default=2.0, help="seconds of audio per direction estimate (steadier when longer)")
    p.add_argument("--gate", type=float, default=None, help="SRP clarity needed to show a class (default 1.2)")
    p.add_argument("--loudest", type=float, default=0.3,
                   help="share of the loudest frames the direction is computed on (1 = all frames)")
    p.add_argument("--min-prob", type=float, default=None, help="smoothed class probability needed to show it (default 0.6)")
    p.add_argument("--fast", action="store_true", help="replay as fast as possible instead of in real time")
    p.add_argument("--no-browser", action="store_true", help="only print the URL")
    p.set_defaults(fn=cmd_live)


def cmd_live(a) -> None:
    import queue
    import sys
    import time
    import webbrowser
    from collections import deque
    from pathlib import Path

    from earshot_arrays.classify import load_classifier
    from earshot_arrays.cli import open_sources
    from earshot_arrays.config import load_setup
    from earshot_arrays.devices import BLOCK_S, DeviceSource, mics
    from earshot_arrays.doa import beam, estimate_bearing
    from earshot_arrays.track import GATE, MIN_PROB, Obs, Tracker

    setup = a.setup
    if not setup and a.replay and Path(a.replay[0]).is_dir() and (Path(a.replay[0]) / "setup.yaml").exists():
        setup = str(Path(a.replay[0]) / "setup.yaml")
    s = load_setup(setup or "setup.yaml")
    if a.replay:
        a.realtime = not a.fast
    sources = open_sources(s.arrays, a)
    clf = load_classifier()
    reason = getattr(clf, "reason", "")
    source = f"REPLAY {Path(a.replay[0]).name}" if a.replay else "LIVE"
    server = LiveMapServer({
        "arrays": [{"name": arr.name, "x": arr.pose.x, "y": arr.pose.y, "rotation": arr.pose.rotation_deg}
                   for arr in s.arrays],
        "radius": a.radius, "classifier": clf.name + (f" ({reason})" if reason else ""),
        "audio": "mic mean" if a.mix else "beam toward the bearing", "source": source,
        "gate": a.gate or GATE, "min_prob": a.min_prob or MIN_PROB, "doa_s": a.doa_s, "loudest": a.loudest,
    }, port=a.port)
    print(f"live map: {server.url}  ({source}, {len(s.arrays)} array(s), classifier {clf.name}). Ctrl-C to stop.")
    if not a.no_browser:
        webbrowser.open(server.url)
    tracker = Tracker({arr.name: arr.pose for arr in s.arrays}, min_prob=a.min_prob or MIN_PROB)
    gate = a.gate or GATE
    history = {arr.name: deque(maxlen=max(1, round(a.doa_s))) for arr in s.arrays}  # last blocks per array
    t, start, last = 0.0, time.monotonic(), {}
    bad = {arr.name: 0 for arr in s.arrays}  # live: seconds in a row without a usable direction
    arrivals = {arr.name: deque() for arr in s.arrays}  # live: when blocks came, to catch a runaway stream

    def log(msg: str) -> None:
        print(f"[audio] t={time.monotonic() - start:6.0f}s  {msg}", file=sys.stderr, flush=True)

    def reopen(i: int, why: str) -> None:
        """Close array i's stream and open it again (by name, so a changed device index is fine)."""
        arr = s.arrays[i]
        log(f"{arr.name}: {why} -> reopening the device")
        server.publish({**last, "note": f"audio of {arr.name}: {why}, reconnecting…"})
        try:
            sources[i].close()
        except Exception as e:  # noqa: BLE001 - the old stream may already be gone
            log(f"{arr.name}: closing the old stream failed: {e}")
        while True:
            try:
                sources[i] = DeviceSource(arr.device, arr.profile)
                break
            except (Exception, SystemExit) as e:  # noqa: BLE001 - unplugged: keep trying until it is back
                log(f"{arr.name}: cannot open the device yet ({e}); retrying in 2 s")
                server.publish({**last, "note": f"audio of {arr.name}: device not found, retrying…"})
                time.sleep(2)
        history[arr.name].clear()
        arrivals[arr.name].clear()
        bad[arr.name] = 0
        log(f"{arr.name}: device open again")

    try:
        while True:
            blocks = []
            for i, (arr, src) in enumerate(zip(s.arrays, sources)):
                if a.replay:
                    blocks.append(src.read(latest=True))
                    continue
                try:
                    block = src.read(latest=True, timeout=STALL_S)
                except queue.Empty:
                    reopen(i, f"no audio for {STALL_S:.0f} s")
                    block = None
                if block is not None:
                    n = int(arr.profile.fs * BLOCK_S)
                    why = block_problem(block, arr.profile.n_channels, n)
                    now = time.monotonic()
                    arrivals[arr.name].append(now)
                    while arrivals[arr.name] and arrivals[arr.name][0] < now - RATE_WINDOW_S:
                        arrivals[arr.name].popleft()
                    if not why and len(arrivals[arr.name]) > 1.5 * RATE_WINDOW_S / BLOCK_S:
                        why = f"{len(arrivals[arr.name]) / RATE_WINDOW_S:.1f} blocks/s instead of {1 / BLOCK_S:.0f}"
                    if why:
                        reopen(i, why)
                        block = None
                blocks.append(block)
            if a.replay and any(b is None for b in blocks):
                break
            if any(b is None for b in blocks):
                continue  # a device was reopened: start over with fresh blocks
            obs, m = {}, {}
            for arr, block in zip(s.arrays, blocks):
                m[arr.name] = mics(block, arr.profile)
                history[arr.name].append(m[arr.name])
                scan: list = []  # direction from the last doa_s seconds, class from the newest second
                b, p = estimate_bearing(np.concatenate(history[arr.name], axis=1), arr.profile.fs,
                                        arr.profile.mic_xy, arr.pose.rotation_deg, scan=scan, loudest=a.loudest)
                level = 20 * np.log10(max(float(np.sqrt(np.mean(np.square(m[arr.name])))), 1e-12))
                obs[arr.name] = Obs(b, p, tuple(scan), level, gate)
                if not a.replay:  # SRP found nothing at all although audio arrives: the stream is broken
                    if p == 0.0 and level > -100:
                        bad[arr.name] += 1
                        sd = np.std(m[arr.name], axis=1)
                        log(f"{arr.name}: no direction at {level:.0f} dBFS (channel std {sd.min():.2e}..{sd.max():.2e}, "
                            f"{len(arrivals[arr.name]) / RATE_WINDOW_S:.1f} blocks/s), {bad[arr.name]} s in a row")
                    else:
                        bad[arr.name] = 0
            arr = max(s.arrays, key=lambda x: obs[x.name].level_db)  # classify the loudest array
            sig = m[arr.name].mean(axis=0) if a.mix else beam(m[arr.name], arr.profile.fs, arr.profile.mic_xy,
                                                               obs[arr.name].bearing, arr.pose.rotation_deg)
            r = clf(sig, arr.profile.fs)
            state = tracker.update(t, obs, r.fine or r.probs)
            last = state.to_json()
            server.publish(last)
            for i, arr in enumerate(s.arrays):
                if bad[arr.name] >= BAD_S:
                    reopen(i, f"no direction for {BAD_S} s although audio arrives")
            print(f"t={t:5.0f}s  " + "  ".join(f"{n} {o.bearing:5.1f}° p{o.power:4.2f}" for n, o in obs.items())
                  + f"  -> {state.label} {state.prob:.0%}" + (f"  ({state.why})" if state.why else ""),
                  flush=True)
            t = (t + blocks[0].shape[1] / arr.profile.fs) if a.replay else time.monotonic() - start
        server.publish({**last, "done": True})
        print("end of recording; the map stays up until Ctrl-C")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for src in sources:
            src.close()
        server.close()
