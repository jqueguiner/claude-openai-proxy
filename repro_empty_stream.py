#!/usr/bin/env python3
"""Reproduit le bug du proxy : une erreur upstream rendue en HTTP 200 vide.

Faux upstream Anthropic qui repond ce que repond le vrai quand le quota est
sature (429) ou le token expire (401). On lance le proxy contre lui, on
demande un chat en streaming, et on regarde ce que le CLIENT recoit.

Attendu si le proxy est correct : status 429/401 + un corps d'erreur.
Attendu si le bug existe    : status 200 + corps vide (0 octet), que l'agent
                              interprete comme une fin de tour.
"""
import http.server
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

MODE = sys.argv[1] if len(sys.argv) > 1 else "429"
BODIES = {
    "429": (429, {"type": "error", "error": {"type": "rate_limit_error",
            "message": "All 4 accounts exhausted. Retry in 60s."}}),
    "401": (401, {"type": "error", "error": {"type": "authentication_error",
            "message": "No Anthropic credential"}}),
}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Upstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        ln = int(self.headers.get("content-length", 0))
        self.rfile.read(ln)
        status, body = BODIES[MODE]
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


up_port = free_port()
proxy_port = free_port()
srv = http.server.HTTPServer(("127.0.0.1", up_port), Upstream)
threading.Thread(target=srv.serve_forever, daemon=True).start()

env = {
    "PATH": "/usr/bin:/bin",
    "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{up_port}",
    "ANTHROPIC_API_KEY": "sk-test-not-real",
    "PROXY_LISTEN": f"127.0.0.1:{proxy_port}",
    "CLAUDE_CODE_VERSION": "2.1.267",
    "RUST_LOG": "info",
    "HOME": "/tmp/nonexistent-home-for-test",   # pas de vraies creds
}
proc = subprocess.Popen(["./target/release/claude-openai-proxy"], env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
for _ in range(50):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/health", timeout=1).read()
        break
    except Exception:
        time.sleep(0.2)

print(f"=== upstream simule : HTTP {BODIES[MODE][0]} ===")
for streaming in (True, False):
    payload = json.dumps({
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": "bonjour"}],
        "stream": streaming,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        data=payload, headers={"content-type": "application/json",
                               "authorization": "Bearer sk-test-not-real"})
    label = "stream=True " if streaming else "stream=False"
    try:
        r = urllib.request.urlopen(req, timeout=20)
        body = r.read()
        verdict = ("  <-- BUG: succes vide, lu comme une fin de tour"
                   if r.status == 200 and len(body) == 0 else "")
        print(f"  {label} -> HTTP {r.status}, {len(body)} octets{verdict}")
        if body:
            print(f"      corps: {body[:150].decode(errors='replace')}")
    except urllib.error.HTTPError as e:
        print(f"  {label} -> HTTP {e.code} (erreur propagee, correct)")
        print(f"      corps: {e.read()[:150].decode(errors='replace')}")

proc.terminate()
srv.shutdown()
