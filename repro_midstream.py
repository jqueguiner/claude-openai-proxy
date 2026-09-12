#!/usr/bin/env python3
"""Deuxieme hypothese : l'erreur arrive APRES le debut du flux SSE.

Le premier test a montre que le proxy propage correctement une erreur qui
arrive AVANT le flux (HTTP 429 franc a la requete). Ce n'est donc pas la.

Ici on simule ce que fait reellement l'upstream quand il tombe en cours de
route : il repond HTTP 200 + content-type SSE (le flux commence), puis soit
il envoie un evenement `error`, soit il coupe la connexion sans rien dire.
C'est le cas que `chat.rs` lignes 145-166 traite par un `StatusCode::OK` deja
ecrit : le statut est parti, on ne peut plus le changer.

  scenario `error_event` : 200 + `event: error` SSE       (quota/auth en vol)
  scenario `silent_drop` : 200 + connexion coupee, 0 octet (upstream tue)
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

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "error_event"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Upstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        ln = int(self.headers.get("content-length", 0))
        self.rfile.read(ln)
        if SCENARIO == "silent_drop":
            # 200 annonce, puis plus rien : la connexion tombe.
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True
            return
        # 200, le flux demarre normalement, puis l'upstream signale l'erreur
        # DANS le flux (c'est ce que fait l'API quand le quota saute en vol).
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        err = {"type": "error", "error": {"type": "rate_limit_error",
               "message": "All 4 accounts exhausted. Retry in 60s."}}
        self.wfile.write(f"event: error\ndata: {json.dumps(err)}\n\n".encode())
        self.wfile.flush()
        self.close_connection = True

    def do_GET(self):
        self.send_response(200)
        self.send_header("content-length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


up_port, proxy_port = free_port(), free_port()
srv = http.server.ThreadingHTTPServer(("127.0.0.1", up_port), Upstream)
threading.Thread(target=srv.serve_forever, daemon=True).start()

env = {
    "PATH": "/usr/bin:/bin",
    "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{up_port}",
    "ANTHROPIC_API_KEY": "sk-test-not-real",
    "PROXY_LISTEN": f"127.0.0.1:{proxy_port}",
    "CLAUDE_CODE_VERSION": "2.1.267",
    "RUST_LOG": "info",
    "HOME": "/tmp/nonexistent-home-for-test",
}
proc = subprocess.Popen(["./target/release/claude-openai-proxy"], env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
for _ in range(50):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/health", timeout=1).read()
        break
    except Exception:
        time.sleep(0.2)

print(f"=== scenario : {SCENARIO} (l'erreur arrive APRES le debut du flux) ===")
payload = json.dumps({"model": "claude-opus-5", "stream": True,
                      "messages": [{"role": "user", "content": "bonjour"}]}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{proxy_port}/v1/chat/completions", data=payload,
    headers={"content-type": "application/json",
             "authorization": "Bearer sk-test-not-real"})
try:
    r = urllib.request.urlopen(req, timeout=20)
    body = r.read()
    print(f"  client recoit : HTTP {r.status}, {len(body)} octets")
    if body:
        print(f"  corps : {body[:400].decode(errors='replace')}")
    if r.status == 200 and len(body) == 0:
        print("\n  >>> BUG CONFIRME : HTTP 200 + 0 octet.")
        print("      C'est exactement 'empty stream with no finish_reason'.")
    elif r.status == 200 and b"error" in body:
        print("\n  >>> statut 200 MAIS l'erreur est dans le corps SSE.")
        print("      Le client doit lire le corps pour la voir.")
except urllib.error.HTTPError as e:
    print(f"  client recoit : HTTP {e.code} (propage correctement)")
    print(f"  corps : {e.read()[:300].decode(errors='replace')}")

proc.terminate()
srv.shutdown()
