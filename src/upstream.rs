//! Talking to the OpenAI-compatible upstream: sending a chat request and
//! pumping its SSE stream through a `Translator`, forwarding the produced
//! Anthropic events over a channel.

use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use crate::config::Config;
use crate::openai::{self, ChatChunk, ChatRequest};
use crate::sse::SseParser;
use crate::stream::Translator;

/// An upstream failure mapped to an Anthropic-style error.
#[derive(Debug, Clone)]
pub struct UpstreamError {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
}

pub fn error_kind(status: u16) -> &'static str {
    match status {
        400 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        529 => "overloaded_error",
        _ => "api_error",
    }
}

pub async fn send_chat(
    client: &reqwest::Client,
    cfg: &Config,
    auth: Option<&str>,
    chat: &ChatRequest,
) -> Result<reqwest::Response, UpstreamError> {
    let url = format!("{}/chat/completions", cfg.base_url);
    if cfg.log_bodies {
        debug!(body = %serde_json::to_string(chat).unwrap_or_default(), "upstream request");
    }
    let mut rb = client.post(&url).json(chat);
    if let Some(key) = auth {
        rb = rb.bearer_auth(key);
    }
    if chat.stream {
        rb = rb.header(reqwest::header::ACCEPT, "text/event-stream");
    }
    let resp = rb.send().await.map_err(|e| {
        error!(error = %e, %url, "upstream request failed");
        UpstreamError { status: 502, kind: "api_error", message: format!("upstream request failed: {e}") }
    })?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        let message = format!("upstream {}: {}", status.as_u16(), openai::error_message(&text));
        warn!(status = status.as_u16(), %message, "upstream error");
        return Err(UpstreamError { status: status.as_u16(), kind: error_kind(status.as_u16()), message });
    }
    Ok(resp)
}

/// Send whatever the translator has buffered.
pub async fn flush(tx: &mpsc::Sender<Bytes>, out: &mut Vec<u8>) {
    if !out.is_empty() {
        let _ = tx.send(Bytes::from(std::mem::take(out))).await;
    }
}

/// Consume one upstream SSE response, feeding chunks into `tr` and forwarding
/// the translated events. Returns `Err` if the upstream stream broke (an
/// Anthropic `error` event has then already been emitted).
pub async fn pump(
    resp: reqwest::Response,
    tr: &mut Translator,
    out: &mut Vec<u8>,
    tx: &mpsc::Sender<Bytes>,
    log_bodies: bool,
) -> Result<(), String> {
    let mut upstream = resp.bytes_stream();
    let mut parser = SseParser::default();
    let ping_every = Duration::from_secs(15);
    loop {
        match tokio::time::timeout(ping_every, upstream.next()).await {
            Err(_elapsed) => tr.ping(out),
            Ok(None) => break,
            Ok(Some(Err(e))) => {
                error!(error = %e, "upstream stream error");
                let msg = format!("upstream stream error: {e}");
                tr.error(&msg, out);
                flush(tx, out).await;
                return Err(msg);
            }
            Ok(Some(Ok(bytes))) => {
                let mut done = false;
                for data in parser.push(&bytes) {
                    if log_bodies {
                        debug!(data = %data, "upstream chunk");
                    }
                    if data.trim() == "[DONE]" {
                        done = true;
                        break;
                    }
                    match serde_json::from_str::<ChatChunk>(&data) {
                        Ok(chunk) => tr.handle_chunk(chunk, out),
                        Err(e) => warn!(error = %e, data = %data, "unparseable upstream chunk"),
                    }
                }
                if done {
                    break;
                }
            }
        }
        flush(tx, out).await;
    }
    flush(tx, out).await;
    Ok(())
}
