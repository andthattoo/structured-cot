"""OpenAI-compatible proxy that injects an EBNF grammar into chat completions.

Pi (or any OpenAI-compatible client) hits this proxy on /v1/chat/completions.
The proxy splices `ebnf` from a grammar file into the request body and
forwards to the real SGLang server. SGLang's xgrammar backend applies the
grammar to constrain output — Pi never has to know.

All other endpoints (models list, /generate, /get_server_info, etc.) pass
through unchanged.

Usage:

    pip install aiohttp

    GRAMMAR_PATH=grammars/fsm_grammar_pi_think.gbnf \\
    SGLANG_URL=http://127.0.0.1:30000 \\
    PROXY_PORT=30001 \\
    python scripts/grammar_proxy.py

Then point Pi at the proxy by editing ~/.pi/agent/models.json so that
provider.baseUrl is "http://127.0.0.1:30001/v1" instead of the SGLang URL.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAMMAR_PATH = REPO_ROOT / "grammars" / "fsm_grammar_pi_think.gbnf"

GRAMMAR_PATH = Path(os.environ.get("GRAMMAR_PATH", DEFAULT_GRAMMAR_PATH))
SGLANG_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30000").rstrip("/")
PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "30001"))
TIMEOUT_SEC = float(os.environ.get("PROXY_TIMEOUT_SEC", "600"))

if not GRAMMAR_PATH.exists():
    sys.exit(f"[proxy] grammar file not found: {GRAMMAR_PATH}")
GRAMMAR = GRAMMAR_PATH.read_text()

logging.basicConfig(level=logging.INFO, format="[proxy] %(message)s")
log = logging.getLogger("proxy")
log.info(f"grammar from {GRAMMAR_PATH} ({len(GRAMMAR)} chars)")
log.info(f"forwarding to {SGLANG_URL}")
log.info(f"listening on http://{PROXY_HOST}:{PROXY_PORT}")


def inject_grammar(body: dict) -> dict:
    if "ebnf" not in body:
        body["ebnf"] = GRAMMAR
    return body


def forward_headers(request: web.Request) -> dict[str, str]:
    drop = {"host", "content-length", "transfer-encoding"}
    return {k: v for k, v in request.headers.items() if k.lower() not in drop}


async def chat_completions(request: web.Request) -> web.StreamResponse:
    raw = await request.read()
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return web.Response(status=400, text="invalid JSON body")
    body = inject_grammar(body)
    is_stream = bool(body.get("stream", False))

    timeout = ClientTimeout(total=TIMEOUT_SEC)
    async with ClientSession(timeout=timeout) as session:
        if is_stream:
            async with session.post(
                f"{SGLANG_URL}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as upstream:
                resp = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        "Content-Type": upstream.headers.get(
                            "Content-Type", "text/event-stream"
                        ),
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
                await resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        else:
            async with session.post(
                f"{SGLANG_URL}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as upstream:
                data = await upstream.read()
                return web.Response(
                    status=upstream.status,
                    body=data,
                    content_type=upstream.headers.get(
                        "Content-Type", "application/json"
                    ),
                )


async def passthrough(request: web.Request) -> web.Response:
    body = await request.read()
    tail = request.match_info.get("tail", "")
    timeout = ClientTimeout(total=TIMEOUT_SEC)
    async with ClientSession(timeout=timeout) as session:
        async with session.request(
            request.method,
            f"{SGLANG_URL}/{tail}",
            data=body if body else None,
            headers=forward_headers(request),
            params=request.query,
        ) as upstream:
            data = await upstream.read()
            return web.Response(
                status=upstream.status,
                body=data,
                content_type=upstream.headers.get(
                    "Content-Type", "application/json"
                ),
            )


def make_app() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_route("*", "/{tail:.*}", passthrough)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host=PROXY_HOST, port=PROXY_PORT)
