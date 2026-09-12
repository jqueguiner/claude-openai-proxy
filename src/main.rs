mod anthropic;
mod chat;
mod claude_auth;
mod config;
mod convert;
mod openai;
mod sse;
mod stream;
mod upstream;
mod websearch;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{debug, error, info};
use tracing_subscriber::EnvFilter;

use crate::anthropic::{ErrorResponse, MessagesRequest};
use crate::config::Config;
use crate::openai::ChatResponse;
use crate::stream::Translator;
use crate::upstream::UpstreamError;

struct AppState {
    cfg: Arc<Config>,
    client: reqwest::Client,
    claude_code_version: String,
}

type Shared = Arc<AppState>;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let cfg = Config::from_env();
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .expect("reqwest client");

    info!(listen = %cfg.listen, upstream = %cfg.base_url,
          model_map = ?cfg.model_map, opus = ?cfg.opus_model, sonnet = ?cfg.sonnet_model,
          haiku = ?cfg.haiku_model, default = ?cfg.default_model,
          search_provider = ?cfg.search_provider, "starting claude-openai-proxy");

    let claude_code_version = match cfg.claude_code_version.clone() {
        Some(v) => v,
        None => detect_claude_code_version().await,
    };
    info!(claude_code_version = %claude_code_version, anthropic = %cfg.anthropic_base_url, "OpenAI-compatible side ready");
    let state: Shared = Arc::new(AppState { cfg: Arc::new(cfg), client, claude_code_version });
    let app = Router::new()
        .route("/", get(health))
        .route("/health", get(health))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        // Same API under an `/anthropic` prefix: some clients (Hermes Agent, gateways)
        // detect the Anthropic Messages transport from a base URL ending in `/anthropic`.
        .route("/anthropic/v1/messages", post(messages))
        .route("/anthropic/v1/messages/count_tokens", post(count_tokens))
        .route("/anthropic/v1/models", get(models))
        .layer(DefaultBodyLimit::disable())
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind(&state.cfg.listen)
        .await
        .unwrap_or_else(|e| panic!("cannot bind {}: {e}", state.cfg.listen));
    axum::serve(listener, app).await.expect("server error");
}

/// Version advertised in the OAuth user-agent; Anthropic rejects stale ones.
async fn detect_claude_code_version() -> String {
    const FALLBACK: &str = "2.1.267";
    let fut = tokio::process::Command::new("claude").arg("--version").output();
    match tokio::time::timeout(Duration::from_secs(5), fut).await {
        Ok(Ok(out)) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout);
            s.split_whitespace()
                .next()
                .filter(|v| v.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false))
                .map(str::to_string)
                .unwrap_or_else(|| FALLBACK.into())
        }
        _ => FALLBACK.into(),
    }
}

fn openai_error(e: &chat::OpenAiError) -> Response {
    let code = StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_GATEWAY);
    (code, Json(e.json())).into_response()
}

/// `POST /v1/chat/completions` -> Anthropic Messages API.
async fn chat_completions(State(st): State<Shared>, headers: HeaderMap, body: Bytes) -> Response {
    let req: chat::ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            let err = chat::OpenAiError { status: 400, kind: "invalid_request_error".into(), message: format!("invalid request body: {e}") };
            return openai_error(&err);
        }
    };
    let client_bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")))
        .map(str::to_string);
    let input = chat::ChatInput {
        client: st.client.clone(),
        cfg: st.cfg.clone(),
        req: req.clone(),
        client_bearer,
        claude_code_version: st.claude_code_version.clone(),
    };
    let (tx, rx) = mpsc::channel::<Bytes>(64);
    if req.stream {
        // Resolve credentials and open the upstream stream first so auth/validation
        // failures still come back as a proper HTTP status.
        let (probe_tx, mut probe_rx) = tokio::sync::oneshot::channel::<Option<chat::OpenAiError>>();
        let mut probe_tx = Some(probe_tx);
        let rx_stream = rx;
        tokio::spawn(async move {
            let outcome = chat::run(input, tx).await;
            if let Some(p) = probe_tx.take() {
                let _ = p.send(outcome.error);
            }
        });
        // Wait for either first bytes or an early error.
        //
        // `select!` races two channels, and a closed channel resolves too:
        // when upstream fails before emitting anything (a 429 or a 401 on the
        // very first call), `tx` is dropped and `recv()` returns `None`
        // immediately — often winning the race against the probe. Taking that
        // `None` as "stream finished" produced a 200 with an empty body, i.e.
        // the exact bug this code was meant to prevent. So when `recv()`
        // yields `None`, wait for the probe before deciding: the error is
        // already on its way and the status line has not been written yet.
        let mut rx_stream = rx_stream;
        let first = tokio::select! {
            b = rx_stream.recv() => match b {
                Some(b) => Some(b),
                None => {
                    if let Ok(Some(err)) = (&mut probe_rx).await {
                        return openai_error(&err);
                    }
                    None
                }
            },
            e = &mut probe_rx => {
                if let Ok(Some(err)) = e { return openai_error(&err); }
                // The probe fired first with no error: the run finished
                // cleanly. Drain whatever it managed to queue.
                rx_stream.recv().await
            }
        };
        let body_stream = async_stream::stream! {
            if let Some(b) = first {
                yield Ok::<Bytes, Infallible>(b);
            }
            while let Some(b) = rx_stream.recv().await {
                yield Ok::<Bytes, Infallible>(b);
            }
        };
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header("x-accel-buffering", "no")
            .body(Body::from_stream(body_stream))
            .expect("valid response");
    }
    drop(rx);
    let outcome = chat::run(input, tx).await;
    if let Some(e) = outcome.error {
        return openai_error(&e);
    }
    Json(outcome.converter.completion()).into_response()
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok", "service": "claude-openai-proxy"}))
}

/// Model list in a shape both OpenAI and Anthropic clients accept.
async fn models(State(st): State<Shared>) -> Json<Value> {
    const CLAUDE: [&str; 8] = [
        "claude-fable-5-1", "claude-fable-5", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
        "claude-sonnet-5", "claude-sonnet-4-6", "claude-haiku-4-5",
    ];
    let mut ids: Vec<String> = CLAUDE.iter().map(|s| s.to_string()).collect();
    ids.extend(st.cfg.known_models());
    ids.extend(st.cfg.chat_model_map.keys().cloned());
    ids.sort();
    ids.dedup();
    let data: Vec<Value> = ids
        .into_iter()
        .map(|id| json!({"id": id, "object": "model", "owned_by": "anthropic", "created": 1767225600,
                         "type": "model", "display_name": id, "created_at": "2026-01-01T00:00:00Z"}))
        .collect();
    Json(json!({"object": "list", "data": data, "has_more": false, "first_id": null, "last_id": null}))
}

/// Rough token estimate (~4 chars/token). Claude Code only uses it for context accounting.
async fn count_tokens(body: Bytes) -> Response {
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("invalid JSON: {e}")),
    };
    let mut chars = 0usize;
    for key in ["system", "messages", "tools"] {
        if let Some(part) = v.get(key) {
            chars += part.to_string().len();
        }
    }
    Json(json!({"input_tokens": (chars / 4).max(1)})).into_response()
}

fn anthropic_error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (status, Json(ErrorResponse::new(kind, message))).into_response()
}

fn upstream_error(e: &UpstreamError) -> Response {
    let code = StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_GATEWAY);
    anthropic_error(code, e.kind, e.message.clone())
}

fn upstream_auth(cfg: &Config, headers: &HeaderMap) -> Option<String> {
    if let Some(k) = &cfg.api_key {
        return Some(k.clone());
    }
    if !cfg.forward_client_key {
        return None;
    }
    if let Some(v) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(v.to_string());
    }
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")))
        .map(str::to_string)
}

fn sse_response(rx: mpsc::Receiver<Bytes>) -> Response {
    let body_stream = async_stream::stream! {
        let mut rx = rx;
        while let Some(b) = rx.recv().await {
            yield Ok::<Bytes, Infallible>(b);
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(body_stream))
        .expect("valid response")
}

/// Assemble a non-streaming Anthropic response from what the translator recorded.
fn assemble_message(tr: &Translator, model: &str, stop_reason: &str) -> Value {
    json!({
        "id": tr.id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": tr.blocks(),
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": tr.usage(),
    })
}

async fn messages(State(st): State<Shared>, headers: HeaderMap, body: Bytes) -> Response {
    let req: MessagesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("invalid request body: {e}"))
        }
    };
    let upstream_model = st.cfg.resolve_model(&req.model);
    let chat = convert::build_chat_request(&req, &st.cfg, &upstream_model);
    let auth = upstream_auth(&st.cfg, &headers);
    let web_search = websearch::spec_from_tools(&req.tools);
    info!(model = %req.model, upstream_model = %upstream_model, stream = req.stream,
          messages = chat.messages.len(), tools = chat.tools.as_ref().map(Vec::len).unwrap_or(0),
          web_search = web_search.is_some(), "proxying /v1/messages");

    if let Some(spec) = web_search {
        if st.cfg.search_provider.is_none() {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "Web search is not enabled on this proxy: set SEARCH_PROVIDER (searxng|brave|tavily) and SEARCH_URL / SEARCH_API_KEY.",
            );
        }
        if !spec.allowed_domains.is_empty() && !spec.blocked_domains.is_empty() {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "web_search: allowed_domains and blocked_domains cannot both be set",
            );
        }
        return web_search_turn(st, auth, req, chat, spec).await;
    }

    if req.stream {
        let (tx, rx) = mpsc::channel::<Bytes>(64);
        let client = st.client.clone();
        let cfg = st.cfg.clone();
        let model = req.model.clone();
        // Fail early with a proper HTTP status when the upstream rejects the request.
        let resp = match upstream::send_chat(&client, &cfg, auth.as_deref(), &chat).await {
            Ok(r) => r,
            Err(e) => return upstream_error(&e),
        };
        tokio::spawn(async move {
            let mut tr = Translator::new(convert::new_message_id(), model);
            let mut out = Vec::new();
            tr.start(&mut out);
            upstream::flush(&tx, &mut out).await;
            if upstream::pump(resp, &mut tr, &mut out, &tx, cfg.log_bodies).await.is_ok() {
                tr.finish(&mut out);
                upstream::flush(&tx, &mut out).await;
            }
        });
        return sse_response(rx);
    }

    let resp = match upstream::send_chat(&st.client, &st.cfg, auth.as_deref(), &chat).await {
        Ok(r) => r,
        Err(e) => return upstream_error(&e),
    };
    let text = match resp.text().await {
        Ok(t) => t,
        Err(e) => return anthropic_error(StatusCode::BAD_GATEWAY, "api_error", format!("reading upstream body: {e}")),
    };
    if st.cfg.log_bodies {
        debug!(body = %text, "upstream response");
    }
    match serde_json::from_str::<ChatResponse>(&text) {
        Ok(chat_resp) => Json(convert::from_chat_response(chat_resp, &req.model)).into_response(),
        Err(e) => {
            error!(error = %e, "cannot parse upstream response");
            anthropic_error(StatusCode::BAD_GATEWAY, "api_error", format!("invalid upstream response: {e}"))
        }
    }
}

/// A turn that carries the Anthropic `web_search` server tool: run the
/// proxy-side search loop, streaming or collected into one JSON message.
async fn web_search_turn(
    st: Shared,
    auth: Option<String>,
    req: MessagesRequest,
    chat: openai::ChatRequest,
    spec: websearch::WebSearchSpec,
) -> Response {
    let input = websearch::TurnInput {
        client: st.client.clone(),
        cfg: st.cfg.clone(),
        auth,
        model: req.model.clone(),
        chat,
        spec,
        first_tool_choice: websearch::first_tool_choice(req.tool_choice.as_ref()),
    };
    let (tx, rx) = mpsc::channel::<Bytes>(64);
    if req.stream {
        tokio::spawn(websearch::run_turn(input, tx));
        return sse_response(rx);
    }
    // Non-streaming: drain the event channel, keep only the recorded blocks.
    let drain = tokio::spawn(async move {
        let mut rx = rx;
        while rx.recv().await.is_some() {}
    });
    let outcome = websearch::run_turn(input, tx).await;
    let _ = drain.await;
    if let Some(e) = outcome.error {
        return upstream_error(&e);
    }
    let stop = outcome.stop_reason.unwrap_or_else(|| "end_turn".into());
    Json(assemble_message(&outcome.translator, &req.model, &stop)).into_response()
}
