#!/usr/bin/env python3
"""Verifie que le CLIENT OpenAI voit bien l'erreur, au lieu d'une reponse vide.

Les tests precedents prouvent ce que le proxy ECRIT sur le fil. Celui-ci
prouve ce que le client en FAIT, avec la meme bibliotheque que Hermes
(openai), sur les deux cas qui produisaient un faux "end of turn" :

  silent_drop : upstream coupe en plein flux, sans stop_reason
  error_event : upstream envoie un evenement `error` dans le flux

Attendu : une exception ou un finish_reason qui n'est PAS "stop".
A ne surtout pas voir : un tour qui se termine proprement avec un contenu
vide, parce que c'est precisement ce que l'agent lisait comme une fin de tour.
"""
import http.server
import json
import socket
import subprocess
import sys
import threading
import time

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "silent_drop"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Upstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        if SCENARIO == "error_event":
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


up, px = free_port(), free_port()
srv = http.server.ThreadingHTTPServer(("127.0.0.1", up), Upstream)
threading.Thread(target=srv.serve_forever, daemon=True).start()

proc = subprocess.Popen(
    ["./target/release/claude-openai-proxy"],
    env={"PATH": "/usr/bin:/bin", "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{up}",
         "ANTHROPIC_API_KEY": "sk-test", "PROXY_LISTEN": f"127.0.0.1:{px}",
         "CLAUDE_CODE_VERSION": "2.1.267", "RUST_LOG": "warn",
         "HOME": "/tmp/nonexistent-home-for-test"},
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import urllib.request
for _ in range(60):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{px}/health", timeout=1).read()
        break
    except Exception:
        time.sleep(0.2)

sys.path.insert(0, "/Users/jlqueguiner/.hermes/hermes-agent/venv/lib/python3.11/site-packages")
from openai import OpenAI          # la bibliotheque que Hermes utilise

client = OpenAI(base_url=f"http://127.0.0.1:{px}/v1", api_key="sk-test", max_retries=0)
print(f"=== scenario {SCENARIO}, vu par le client OpenAI ===")
try:
    chunks, text, finish, err = 0, "", None, None
    for ch in client.chat.completions.create(
            model="claude-opus-5", stream=True,
            messages=[{"role": "user", "content": "bonjour"}]):
        chunks += 1
        extra = getattr(ch, "model_extra", None) or {}
        if extra.get("error") or getattr(ch, "error", None):
            err = extra.get("error") or getattr(ch, "error", None)
        if ch.choices:
            text += ch.choices[0].delta.content or ""
            finish = ch.choices[0].finish_reason or finish
    print(f"  chunks={chunks} texte={text!r} finish_reason={finish!r}")
    if err:
        print(f"  erreur vue dans le flux : {str(err)[:120]}")
        print("  >>> OK : le client peut distinguer l'echec d'une vraie reponse")
    elif finish == "stop" and not text:
        print("  >>> BUG : fin de tour propre avec un contenu vide")
    else:
        print(f"  >>> finish_reason={finish!r}, pas un faux 'stop'")
except Exception as e:
    print(f"  exception {type(e).__name__}: {str(e)[:160]}")
    print("  >>> OK : l'echec remonte comme une erreur, pas comme une reponse vide")

proc.terminate()
srv.shutdown()
