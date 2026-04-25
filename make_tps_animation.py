"""
Build a self-contained side-by-side generation animation from evaluator JSONL.

Example:
  uv run python make_tps_animation.py \
      --task-id 3781 \
      --left-results lcb_v6_2025_01_01_free_n50/results.jsonl \
      --left-mode free \
      --left-label FREE \
      --left-seconds 237 \
      --right-results lcb_v6_2025_01_01_fsm_lcb_plan_n50/results.jsonl \
      --right-mode fsm \
      --right-label FSM_PLAN \
      --right-seconds 279 \
      --out lcb_3781_free_vs_fsm_plan.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


TOKEN_RE = re.compile(r"\s+|[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\sA-Za-z_0-9]", re.UNICODE)


def load_result(path: Path, task_id: str, mode: str) -> dict:
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("task_id")) == str(task_id):
                if mode not in row:
                    raise SystemExit(f"{path}: task {task_id!r} has no mode {mode!r}")
                return row[mode]
    raise SystemExit(f"{path}: task {task_id!r} not found")


def split_chunks(text: str) -> list[str]:
    chunks = TOKEN_RE.findall(text)
    return chunks if chunks else [text]


def pane_payload(label: str, mode: str, result: dict, seconds: float | None, tps: float) -> dict:
    text = result.get("raw_response") or result.get("err") or ""
    total_tokens = int(result.get("total_tokens") or max(1, len(split_chunks(text))))
    if seconds is None:
        seconds = total_tokens / max(tps, 0.001)
    return {
        "label": label,
        "mode": mode,
        "pass": bool(result.get("pass")),
        "failure_type": result.get("failure_type", "unknown"),
        "think_tokens": result.get("think_tokens"),
        "total_tokens": total_tokens,
        "post_think_tokens": result.get("post_think_tokens"),
        "comment_tokens": result.get("code_comment_tokens"),
        "extraction_issue": result.get("extraction_issue"),
        "seconds": float(seconds),
        "chunks": split_chunks(text),
    }


def _json_for_script(data: dict) -> str:
    return json.dumps(data).replace("</", "<\\/")


def build_html(data: dict) -> str:
    payload = _json_for_script(data)
    title = html.escape(f"{data['task_id']} FREE vs FSM")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #101114;
  --panel: #181a20;
  --panel-2: #20232b;
  --text: #f2f2f0;
  --muted: #a9adba;
  --line: #343844;
  --good: #56d364;
  --bad: #ff7b72;
  --think: #b7c7ff;
  --code: #f8e3a1;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.wrap {{
  height: 100vh;
  display: grid;
  grid-template-rows: auto auto 1fr;
  overflow: hidden;
}}
header {{
  padding: 18px 24px 14px;
  border-bottom: 1px solid var(--line);
  background: #111319;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 20px;
  font-weight: 650;
  letter-spacing: 0;
}}
.sub {{
  color: var(--muted);
  font-size: 13px;
}}
.controls {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 24px;
  border-bottom: 1px solid var(--line);
  background: #141720;
}}
button {{
  border: 1px solid var(--line);
  background: var(--panel-2);
  color: var(--text);
  border-radius: 6px;
  padding: 8px 12px;
  font-weight: 600;
  cursor: pointer;
}}
button:hover {{ border-color: #6e7681; }}
label {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 13px;
}}
input[type="range"] {{ width: 180px; }}
.grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  min-height: 0;
  overflow: hidden;
  background: var(--line);
}}
.pane {{
  min-width: 0;
  min-height: 0;
  background: var(--panel);
  display: grid;
  grid-template-rows: auto 1fr;
  overflow: hidden;
}}
.pane-head {{
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
  background: #171a22;
}}
.pane-title {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}}
.label {{
  font-size: 18px;
  font-weight: 700;
}}
.status {{
  color: var(--muted);
  font-size: 13px;
}}
.status.pass {{ color: var(--good); }}
.status.fail {{ color: var(--bad); }}
.stats {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 8px;
}}
.stat {{
  border: 1px solid var(--line);
  background: #11141b;
  border-radius: 6px;
  padding: 7px 8px;
  min-width: 0;
}}
.stat span {{
  display: block;
  color: var(--muted);
  font-size: 11px;
  margin-bottom: 3px;
}}
.stat strong {{
  display: block;
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.text {{
  min-height: 0;
  margin: 0;
  overflow: auto;
  padding: 18px;
  font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  white-space: pre-wrap;
  word-break: break-word;
  overscroll-behavior: contain;
  display: flex;
  align-items: stretch;
}}
.tail {{
  display: block;
  min-width: 100%;
  margin-top: auto;
}}
.cursor {{
  display: inline-block;
  width: 7px;
  height: 1.2em;
  transform: translateY(3px);
  background: var(--text);
  margin-left: 2px;
  animation: blink 1s steps(2, start) infinite;
}}
@keyframes blink {{ 50% {{ opacity: 0; }} }}
.think {{ color: var(--think); }}
.code {{ color: var(--code); }}
@media (max-width: 900px) {{
  .grid {{ grid-template-columns: 1fr; }}
  .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{title}</h1>
    <div class="sub">Side-by-side approximate token generation. Model seconds are scaled for viewing; counters use each run's completion-token total.</div>
  </header>
  <div class="controls">
    <button id="play">Play</button>
    <button id="reset">Reset</button>
    <label>Playback speed <input id="speed" type="range" min="1" max="120" value="{data['playback_speed']}"><span id="speedText">{data['playback_speed']}x</span></label>
  </div>
  <main class="grid">
    <section class="pane" id="leftPane">
      <div class="pane-head"></div>
      <pre class="text"><code class="tail"><span class="out"></span><span class="cursor"></span></code></pre>
    </section>
    <section class="pane" id="rightPane">
      <div class="pane-head"></div>
      <pre class="text"><code class="tail"><span class="out"></span><span class="cursor"></span></code></pre>
    </section>
  </main>
</div>
<script>
const DATA = {payload};
const panes = [DATA.left, DATA.right];
let playing = false;
let startedAt = 0;
let pausedAtModel = 0;

function esc(s) {{
  return s.replace(/[&<>]/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[ch]));
}}

function renderHead(el, pane) {{
  const tps = pane.total_tokens / Math.max(pane.seconds, 0.001);
  el.querySelector('.pane-head').innerHTML = `
    <div class="pane-title">
      <div class="label">${{esc(pane.label)}}</div>
      <div class="status ${{pane.pass ? 'pass' : 'fail'}}">${{pane.pass ? 'pass' : pane.failure_type}}</div>
    </div>
    <div class="stats">
      <div class="stat"><span>think</span><strong>${{pane.think_tokens ?? '-'}}</strong></div>
      <div class="stat"><span>total</span><strong>${{pane.total_tokens}}</strong></div>
      <div class="stat"><span>post-think</span><strong>${{pane.post_think_tokens ?? '-'}}</strong></div>
      <div class="stat"><span>comments</span><strong>${{pane.comment_tokens ?? '-'}}</strong></div>
      <div class="stat"><span>TPS</span><strong>${{tps.toFixed(1)}}</strong></div>
    </div>`;
}}

function colorize(text) {{
  const safe = esc(text);
  const close = safe.indexOf('&lt;/think&gt;');
  if (close === -1) return `<span class="think">${{safe}}</span>`;
  const end = close + '&lt;/think&gt;'.length;
  return `<span class="think">${{safe.slice(0, end)}}</span><span class="code">${{safe.slice(end)}}</span>`;
}}

function textAt(pane, modelSeconds) {{
  const frac = Math.min(1, Math.max(0, modelSeconds / Math.max(pane.seconds, 0.001)));
  const chunks = Math.floor(frac * pane.chunks.length);
  return pane.chunks.slice(0, chunks).join('');
}}

function render(modelSeconds) {{
  ['leftPane', 'rightPane'].forEach((id, i) => {{
    const el = document.getElementById(id);
    const pane = panes[i];
    const shown = textAt(pane, modelSeconds);
    el.querySelector('.out').innerHTML = colorize(shown);
    const textEl = el.querySelector('.text');
    textEl.scrollTop = textEl.scrollHeight;
  }});
}}

function maxSeconds() {{
  return Math.max(DATA.left.seconds, DATA.right.seconds);
}}

function tick(ts) {{
  if (!playing) return;
  const speed = Number(document.getElementById('speed').value);
  const modelSeconds = pausedAtModel + ((ts - startedAt) / 1000) * speed;
  render(modelSeconds);
  if (modelSeconds >= maxSeconds()) {{
    playing = false;
    pausedAtModel = maxSeconds();
    document.getElementById('play').textContent = 'Play';
    render(pausedAtModel);
    return;
  }}
  requestAnimationFrame(tick);
}}

document.getElementById('play').addEventListener('click', () => {{
  playing = !playing;
  document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
  if (playing) {{
    startedAt = performance.now();
    requestAnimationFrame(tick);
  }} else {{
    const speed = Number(document.getElementById('speed').value);
    pausedAtModel += ((performance.now() - startedAt) / 1000) * speed;
  }}
}});

document.getElementById('reset').addEventListener('click', () => {{
  playing = false;
  pausedAtModel = 0;
  document.getElementById('play').textContent = 'Play';
  render(0);
}});

document.getElementById('speed').addEventListener('input', e => {{
  document.getElementById('speedText').textContent = `${{e.target.value}}x`;
}});

renderHead(document.getElementById('leftPane'), DATA.left);
renderHead(document.getElementById('rightPane'), DATA.right);
render(0);
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--left-results", type=Path, required=True)
    p.add_argument("--left-mode", default="free")
    p.add_argument("--left-label", default="FREE")
    p.add_argument("--left-seconds", type=float, default=None,
                   help="Real generation seconds for left pane. If omitted, use --left-tps.")
    p.add_argument("--left-tps", type=float, default=40.0)
    p.add_argument("--right-results", type=Path, required=True)
    p.add_argument("--right-mode", default="fsm")
    p.add_argument("--right-label", default="FSM_PLAN")
    p.add_argument("--right-seconds", type=float, default=None,
                   help="Real generation seconds for right pane. If omitted, use --right-tps.")
    p.add_argument("--right-tps", type=float, default=40.0)
    p.add_argument("--playback-speed", type=int, default=40,
                   help="Model seconds per browser second.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    left = load_result(args.left_results, args.task_id, args.left_mode)
    right = load_result(args.right_results, args.task_id, args.right_mode)
    data = {
        "task_id": str(args.task_id),
        "playback_speed": args.playback_speed,
        "left": pane_payload(args.left_label, args.left_mode, left, args.left_seconds, args.left_tps),
        "right": pane_payload(args.right_label, args.right_mode, right, args.right_seconds, args.right_tps),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(data))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
