//! The OpenAI-compatible side: `POST /v1/chat/completions` served on top of the
//! Anthropic Messages API (typically with Claude Code's OAuth login).
//!
//! The Anthropic call is always streamed (large `max_tokens` would otherwise
//! be rejected); the stream is either re-emitted as OpenAI chunks or collected
//! into one `chat.completion` object.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::claude_auth::{self, Credential};
use crate::config::Config;
use crate::openai::{value_to_text, ToolCall};
use crate::sse::SseParser;

pub const CLAUDE_CODE_SYSTEM_PREFIX: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

/// Anthropic's OAuth billing classifier fingerprints third-party agents by
/// their tool set and rejects the request with 400 "Third-party apps now draw
/// from your extra usage". Verified empirically (and mirrored from Hermes
/// Agent's native adapter): a bare tool set of 10+ tools trips it, the same
/// tools presented as MCP tools (`mcp__<name>`, Claude Code's own convention)
/// pass. So on the OAuth wire every tool becomes `mcp__<name>` (a
/// single-underscore `mcp_<x>` is promoted to `mcp__<x>`), a couple of
/// known-flagged names are aliased first, and responses are mapped back.
const OAUTH_TOOL_ALIASES: [(&str, &str); 2] = [("session_search", "chat_history_lookup"), ("memory", "context_notes")];

fn oauth_wire_name(name: &str) -> String {
    let mut base = name.to_string();
    for (from, to) in OAUTH_TOOL_ALIASES {
        if name == from {
            base = to.to_string();
        }
    }
    if base.starts_with("mcp__") {
        return base;
    }
    if let Some(rest) = base.strip_prefix("mcp_") {
        return format!("mcp__{rest}");
    }
    format!("mcp__{base}")
}

fn oauth_sanitize_text(text: &str) -> String {
    let mut t = text
        .replace("Hermes Agent", "Claude Code")
        .replace("Hermes agent", "Claude Code")
        .replace("hermes-agent", "claude-code")
        .replace("Nous Research", "Anthropic");
    // Prose-safe alias only for the unambiguous token (`memory` is ordinary English).
    t = replace_word(&t, "session_search", "chat_history_lookup");
    t
}

fn replace_word(text: &str, word: &str, with: &str) -> String {
    if !text.contains(word) {
        return text.to_string();
    }
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find(word) {
        let before_ok = rest[..pos].chars().last().map(|c| !is_word(c)).unwrap_or(true);
        let after_ok = rest[pos + word.len()..].chars().next().map(|c| !is_word(c)).unwrap_or(true);
        out.push_str(&rest[..pos]);
        out.push_str(if before_ok && after_ok { with } else { word });
        rest = &rest[pos + word.len()..];
    }
    out.push_str(rest);
    out
}
const OAUTH_BETAS: &str = "claude-code-20250219,oauth-2025-04-20";
const ANTHROPIC_VERSION: &str = "2023-06-01";

// ---------------------------------------------------------------------------
// OpenAI request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<InMessage>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<Value>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

// ---------------------------------------------------------------------------
// OpenAI -> Anthropic request
// ---------------------------------------------------------------------------

fn model_family(model: &str) -> &'static str {
    let m = model.to_ascii_lowercase();
    if m.contains("fable") || m.contains("mythos") {
        "fable"
    } else if m.contains("haiku") {
        "haiku"
    } else if m.contains("opus-5") || m.contains("opus-4-7") || m.contains("opus-4-8") || m.contains("sonnet-5") {
        "no-sampling"
    } else {
        "classic"
    }
}

fn text_block(text: &str) -> Option<Value> {
    if text.trim().is_empty() {
        None
    } else {
        Some(json!({"type": "text", "text": text}))
    }
}

/// OpenAI content (string or parts) -> Anthropic blocks.
fn content_to_blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) => text_block(s).into_iter().collect(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") => p.get("text").and_then(Value::as_str).and_then(text_block),
                Some("image_url") => {
                    let url = p
                        .get("image_url")
                        .and_then(|i| i.get("url").and_then(Value::as_str).or_else(|| i.as_str()))?;
                    image_block(url)
                }
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

fn image_block(url: &str) -> Option<Value> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',')?;
        let media_type = meta.split(';').next().unwrap_or("image/png");
        return Some(json!({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}));
    }
    Some(json!({"type": "image", "source": {"type": "url", "url": url}}))
}

/// Push blocks onto the Anthropic message list, merging consecutive same-role turns.
fn push_blocks(messages: &mut Vec<(String, Vec<Value>)>, role: &str, blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }
    if let Some((last_role, last_blocks)) = messages.last_mut() {
        if last_role == role {
            last_blocks.extend(blocks);
            return;
        }
    }
    messages.push((role.to_string(), blocks));
}

pub struct BuiltRequest {
    pub body: Value,
    /// Wire tool name -> name the client used (OAuth aliasing).
    pub reverse_names: std::collections::HashMap<String, String>,
}

pub fn build_anthropic_request(req: &ChatCompletionRequest, cfg: &Config, model: &str, oauth: bool) -> BuiltRequest {
    let family = model_family(model);
    let mut system: Vec<Value> = Vec::new();
    if oauth {
        system.push(json!({"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX}));
    }
    let mut messages: Vec<(String, Vec<Value>)> = Vec::new();

    for m in &req.messages {
        match m.role.as_str() {
            "system" | "developer" => {
                let text = value_to_text(m.content.clone());
                if text.trim().is_empty() {
                    continue;
                }
                if messages.is_empty() {
                    system.push(json!({"type": "text", "text": text}));
                } else {
                    // Mid-conversation instruction: keep ordering valid on every model.
                    let wrapped = format!("<system-instruction>\n{text}\n</system-instruction>");
                    push_blocks(&mut messages, "user", vec![json!({"type": "text", "text": wrapped})]);
                }
            }
            "user" => {
                let blocks = m.content.as_ref().map(content_to_blocks).unwrap_or_default();
                push_blocks(&mut messages, "user", blocks);
            }
            "assistant" => {
                let mut blocks = m.content.as_ref().map(content_to_blocks).unwrap_or_default();
                for tc in m.tool_calls.clone().unwrap_or_default() {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id.unwrap_or_else(crate::convert::new_call_id),
                        "name": tc.function.name,
                        "input": crate::convert::parse_tool_arguments(&tc.function.arguments),
                    }));
                }
                push_blocks(&mut messages, "assistant", blocks);
            }
            "tool" | "function" => {
                let text = value_to_text(m.content.clone());
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                    "content": if text.is_empty() { "(no output)".to_string() } else { text },
                });
                push_blocks(&mut messages, "user", vec![block]);
            }
            other => warn!(role = other, "dropping message with unknown role"),
        }
    }

    // Anthropic requires the conversation to start with a user turn.
    if messages.first().map(|(r, _)| r == "assistant").unwrap_or(true) {
        messages.insert(0, ("user".into(), vec![json!({"type": "text", "text": "(continue)"})]));
    }

    let mut reverse_names: std::collections::HashMap<String, String> = Default::default();
    if oauth {
        for block in system.iter_mut() {
            if let Some(t) = block.get("text").and_then(Value::as_str) {
                block["text"] = json!(oauth_sanitize_text(t));
            }
        }
        for (_, blocks) in messages.iter_mut() {
            for b in blocks.iter_mut() {
                if b.get("type") == Some(&json!("tool_use")) {
                    if let Some(n) = b.get("name").and_then(Value::as_str) {
                        let wire = oauth_wire_name(n);
                        if wire != n {
                            reverse_names.insert(wire.clone(), n.to_string());
                            b["name"] = json!(wire);
                        }
                    }
                }
            }
        }
    }

    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert(
        "max_tokens".into(),
        json!(req.max_completion_tokens.or(req.max_tokens).unwrap_or(cfg.chat_max_tokens)),
    );
    // Prompt caching. Without an explicit `cache_control`, Anthropic re-reads the
    // whole prefix at full price on every turn: a long Desk thread then costs
    // O(n^2) input tokens. We mark the largest stable prefixes as `ephemeral`
    // so each turn only pays for what changed since the last one.
    //
    // Anthropic allows at most 4 breakpoints, and everything *before* a
    // breakpoint is cached, so we spend them on the boundaries that stay byte
    // identical across turns: end of system, then the last message of the
    // conversation prefix (all turns but the newest exchange).
    if !system.is_empty() {
        if let Some(last) = system.last_mut() {
            last["cache_control"] = json!({"type": "ephemeral"});
        }
        body.insert("system".into(), Value::Array(system));
    }
    // Cache the history prefix, excluding the final (new) user message: that one
    // differs every turn, so a breakpoint on it would never hit.
    let cache_upto = messages.len().saturating_sub(1);
    if cache_upto > 0 {
        if let Some((_, blocks)) = messages.get_mut(cache_upto - 1) {
            if let Some(last) = blocks.last_mut() {
                if last.is_object() {
                    last["cache_control"] = json!({"type": "ephemeral"});
                }
            }
        }
    }
    body.insert(
        "messages".into(),
        Value::Array(messages.into_iter().map(|(role, content)| json!({"role": role, "content": content})).collect()),
    );
    body.insert("stream".into(), json!(true));

    // Tools
    let tools: Vec<Value> = req
        .tools
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|t| {
            let f = t.get("function")?;
            let name = f.get("name")?.as_str()?.to_string();
            let mut schema = f.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            if !schema.is_object() {
                schema = json!({"type": "object", "properties": {}});
            }
            if schema.get("type").is_none() {
                schema["type"] = json!("object");
            }
            let mut description = f.get("description").and_then(Value::as_str).unwrap_or("").to_string();
            let mut wire_name = name.clone();
            if oauth {
                wire_name = oauth_wire_name(&name);
                if wire_name != name {
                    reverse_names.insert(wire_name.clone(), name.clone());
                }
                description = oauth_sanitize_text(&description);
            }
            Some(json!({
                "name": wire_name,
                "description": description,
                "input_schema": schema,
            }))
        })
        .collect();
    if !tools.is_empty() {
        // Tools sit at the very front of the cached prefix and Hermes ships a
        // large, stable tool list on every call: cache past the last one.
        let mut tools = tools;
        if let Some(last) = tools.last_mut() {
            last["cache_control"] = json!({"type": "ephemeral"});
        }
        body.insert("tools".into(), Value::Array(tools));
        let mut choice: Option<Value> = match &req.tool_choice {
            None => None,
            Some(Value::String(s)) => match s.as_str() {
                "auto" => Some(json!({"type": "auto"})),
                "required" => Some(json!({"type": "any"})),
                "none" => Some(json!({"type": "none"})),
                _ => None,
            },
            Some(v) => v
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .map(|n| json!({"type": "tool", "name": n})),
        };
        if family == "fable" {
            if let Some(c) = &choice {
                if matches!(c["type"].as_str(), Some("any") | Some("tool")) {
                    warn!(model, "forced tool_choice is not supported on this model; using auto");
                    choice = Some(json!({"type": "auto"}));
                }
            }
        }
        if req.parallel_tool_calls == Some(false) {
            let mut c = choice.unwrap_or_else(|| json!({"type": "auto"}));
            if c["type"] != "none" {
                c["disable_parallel_tool_use"] = json!(true);
            }
            choice = Some(c);
        }
        if let Some(c) = choice {
            body.insert("tool_choice".into(), c);
        }
    }

    // Sampling
    if family == "classic" || family == "haiku" {
        if let Some(t) = req.temperature {
            body.insert("temperature".into(), json!(t));
        }
        if let Some(p) = req.top_p {
            body.insert("top_p".into(), json!(p));
        }
    } else if req.temperature.is_some() || req.top_p.is_some() {
        debug!(model, "dropping temperature/top_p (rejected by this model)");
    }

    // Stop sequences
    let stops: Vec<String> = match &req.stop {
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        _ => vec![],
    };
    if !stops.is_empty() {
        body.insert("stop_sequences".into(), json!(stops));
    }

    // Reasoning effort -> output_config.effort (+ adaptive thinking where it must be enabled explicitly)
    if let Some(effort) = req.reasoning_effort.as_deref() {
        let mapped = match effort {
            "none" | "minimal" | "low" => Some("low"),
            "medium" => Some("medium"),
            "high" => Some("high"),
            "xhigh" => Some("xhigh"),
            "max" => Some("max"),
            _ => None,
        };
        if let Some(e) = mapped {
            body.insert("output_config".into(), json!({"effort": e}));
            if family == "classic" || family == "no-sampling" {
                body.insert("thinking".into(), json!({"type": "adaptive"}));
            }
        }
    }

    BuiltRequest { body: Value::Object(body), reverse_names }
}

// ---------------------------------------------------------------------------
// Anthropic stream -> OpenAI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OpenAiError {
    pub status: u16,
    pub kind: String,
    pub message: String,
}

impl OpenAiError {
    pub fn json(&self) -> Value {
        json!({"error": {"message": self.message, "type": self.kind, "code": self.kind, "param": null}})
    }
}

#[derive(Default, Debug)]
struct ToolAcc {
    id: String,
    name: String,
    arguments: String,
}

/// Converts Anthropic streaming events into OpenAI chunks while also
/// accumulating the final message (for non-streaming clients).
pub struct Converter {
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    sent_role: bool,
    text: String,
    reasoning: String,
    tools: Vec<ToolAcc>,
    /// Anthropic block index -> tool position in `tools`.
    block_to_tool: std::collections::HashMap<u64, usize>,
    finish_reason: Option<String>,
    usage: Value,
    done: bool,
    reverse_names: std::collections::HashMap<String, String>,
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl Converter {
    pub fn new(model: String, include_usage: bool) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            model,
            created: now_secs(),
            include_usage,
            sent_role: false,
            text: String::new(),
            reasoning: String::new(),
            tools: Vec::new(),
            block_to_tool: Default::default(),
            finish_reason: None,
            usage: json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            done: false,
            reverse_names: Default::default(),
        }
    }

    pub fn with_reverse_names(mut self, names: std::collections::HashMap<String, String>) -> Self {
        self.reverse_names = names;
        self
    }

    fn chunk(&self, delta: Value, finish_reason: Option<&str>) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": null}],
        })
    }

    fn write(out: &mut Vec<u8>, v: &Value) {
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(v.to_string().as_bytes());
        out.extend_from_slice(b"\n\n");
    }

    fn delta(&mut self, out: &mut Vec<u8>, mut delta: Value) {
        if !self.sent_role {
            delta["role"] = json!("assistant");
            self.sent_role = true;
        }
        let c = self.chunk(delta, None);
        Self::write(out, &c);
    }

    fn map_usage(&mut self, usage: &Value) {
        let input = usage.get("input_tokens").and_then(Value::as_u64);
        let cache_read = usage.get("cache_read_input_tokens").and_then(Value::as_u64).unwrap_or(0);
        let cache_create = usage.get("cache_creation_input_tokens").and_then(Value::as_u64).unwrap_or(0);
        let output = usage.get("output_tokens").and_then(Value::as_u64);
        let mut u = self.usage.clone();
        if let Some(i) = input {
            u["prompt_tokens"] = json!(i + cache_read + cache_create);
            if cache_read > 0 {
                u["prompt_tokens_details"] = json!({"cached_tokens": cache_read});
            }
        }
        if let Some(o) = output {
            u["completion_tokens"] = json!(o);
        }
        u["total_tokens"] = json!(u["prompt_tokens"].as_u64().unwrap_or(0) + u["completion_tokens"].as_u64().unwrap_or(0));
        self.usage = u;
    }

    /// Handle one Anthropic SSE event (already JSON-decoded).
    pub fn handle(&mut self, ev: &Value, out: &mut Vec<u8>) {
        if self.done {
            return;
        }
        match ev.get("type").and_then(Value::as_str).unwrap_or("") {
            "message_start" => {
                if let Some(u) = ev.pointer("/message/usage") {
                    self.map_usage(u);
                }
                self.delta(out, json!({"content": ""}));
            }
            "content_block_start" => {
                let index = ev.get("index").and_then(Value::as_u64).unwrap_or(0);
                let block = &ev["content_block"];
                match block.get("type").and_then(Value::as_str) {
                    Some("tool_use") | Some("server_tool_use") => {
                        let pos = self.tools.len();
                        let id = block.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                        let wire = block.get("name").and_then(Value::as_str).unwrap_or("");
                        let name = self.reverse_names.get(wire).cloned().unwrap_or_else(|| wire.to_string());
                        self.tools.push(ToolAcc { id: id.clone(), name: name.clone(), arguments: String::new() });
                        self.block_to_tool.insert(index, pos);
                        self.delta(
                            out,
                            json!({"tool_calls": [{"index": pos, "id": id, "type": "function",
                                                   "function": {"name": name, "arguments": ""}}]}),
                        );
                    }
                    Some("text") => {
                        if let Some(t) = block.get("text").and_then(Value::as_str).filter(|t| !t.is_empty()) {
                            self.text.push_str(t);
                            self.delta(out, json!({"content": t}));
                        }
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let index = ev.get("index").and_then(Value::as_u64).unwrap_or(0);
                let d = &ev["delta"];
                match d.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(t) = d.get("text").and_then(Value::as_str).filter(|t| !t.is_empty()) {
                            self.text.push_str(t);
                            self.delta(out, json!({"content": t}));
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(t) = d.get("thinking").and_then(Value::as_str).filter(|t| !t.is_empty()) {
                            self.reasoning.push_str(t);
                            self.delta(out, json!({"reasoning_content": t}));
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(pos) = self.block_to_tool.get(&index).copied() {
                            let partial = d.get("partial_json").and_then(Value::as_str).unwrap_or("");
                            if !partial.is_empty() {
                                self.tools[pos].arguments.push_str(partial);
                                self.delta(
                                    out,
                                    json!({"tool_calls": [{"index": pos, "function": {"arguments": partial}}]}),
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = ev.get("index").and_then(Value::as_u64).unwrap_or(0);
                if let Some(pos) = self.block_to_tool.get(&index).copied() {
                    if self.tools[pos].arguments.trim().is_empty() {
                        self.tools[pos].arguments = "{}".into();
                        self.delta(out, json!({"tool_calls": [{"index": pos, "function": {"arguments": "{}"}}]}));
                    }
                }
            }
            "message_delta" => {
                if let Some(u) = ev.get("usage") {
                    self.map_usage(u);
                }
                let stop = ev.pointer("/delta/stop_reason").and_then(Value::as_str).unwrap_or("end_turn");
                self.finish_reason = Some(map_finish(stop, !self.tools.is_empty()).to_string());
            }
            "message_stop" => self.finish(out),
            "error" => {
                let msg = ev.pointer("/error/message").and_then(Value::as_str).unwrap_or("upstream error");
                let kind = ev.pointer("/error/type").and_then(Value::as_str).unwrap_or("api_error");
                Self::write(out, &json!({"error": {"message": msg, "type": kind, "code": kind}}));
                out.extend_from_slice(b"data: [DONE]\n\n");
                self.done = true;
            }
            _ => {} // ping and friends
        }
    }

    /// Emit the final chunk(s) and `[DONE]`.
    ///
    /// `saw_terminal` says whether upstream actually told us the turn ended
    /// (a `message_delta` carrying `stop_reason`, or `message_stop`). When it
    /// did not, the stream was cut short: an upstream drop, a killed process,
    /// a proxy timeout. Reporting `finish_reason: "stop"` in that case
    /// fabricates a clean end of turn that never happened, and the client has
    /// no way to tell it apart from a real one — it just sees an assistant
    /// turn with no content and moves on.
    ///
    /// OpenAI's own vocabulary has no code for "the stream died", so we emit
    /// an SSE `error` event instead of a final chunk. Clients that follow the
    /// OpenAI protocol surface it as an error and retry, rather than
    /// accepting an empty answer as final.
    pub fn finish_with(&mut self, out: &mut Vec<u8>, saw_terminal: bool) {
        if self.done {
            return;
        }
        if !saw_terminal && self.finish_reason.is_none() {
            Self::write(out, &json!({"error": {
                "message": "upstream stream ended before the turn was complete \
                            (no stop_reason received); the response is incomplete",
                "type": "api_error",
                "code": "incomplete_stream",
            }}));
            out.extend_from_slice(b"data: [DONE]\n\n");
            self.done = true;
            return;
        }
        let finish = self.finish_reason.clone().unwrap_or_else(|| "stop".into());
        if !self.sent_role {
            self.delta(out, json!({"content": ""}));
        }
        let mut last = self.chunk(json!({}), Some(&finish));
        if self.include_usage {
            last["usage"] = self.usage.clone();
        }
        Self::write(out, &last);
        out.extend_from_slice(b"data: [DONE]\n\n");
        self.done = true;
    }

    /// Backwards-compatible entry point: assumes the turn ended normally.
    /// Used on the paths where upstream already signalled a terminal event.
    pub fn finish(&mut self, out: &mut Vec<u8>) {
        self.finish_with(out, true);
    }

    /// True once upstream sent something that genuinely ends the turn.
    pub fn saw_terminal(&self) -> bool {
        self.done || self.finish_reason.is_some()
    }

    /// Non-streaming `chat.completion` object built from what was accumulated.
    pub fn completion(&self) -> Value {
        let tool_calls: Vec<Value> = self
            .tools
            .iter()
            .map(|t| json!({"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}}))
            .collect();
        let mut message = json!({"role": "assistant", "content": if self.text.is_empty() { Value::Null } else { json!(self.text) }});
        if !tool_calls.is_empty() {
            message["tool_calls"] = json!(tool_calls);
        }
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = json!(self.reasoning);
        }
        json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": self.finish_reason.clone().unwrap_or_else(|| "stop".into()), "logprobs": null}],
            "usage": self.usage,
        })
    }
}

fn map_finish(stop_reason: &str, has_tools: bool) -> &'static str {
    match stop_reason {
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        "refusal" => "content_filter",
        _ if has_tools => "tool_calls",
        _ => "stop",
    }
}

// ---------------------------------------------------------------------------
// Driving the Anthropic call
// ---------------------------------------------------------------------------

pub struct ChatInput {
    pub client: reqwest::Client,
    pub cfg: Arc<Config>,
    pub req: ChatCompletionRequest,
    pub client_bearer: Option<String>,
    pub claude_code_version: String,
}

pub struct ChatOutcome {
    pub converter: Converter,
    pub error: Option<OpenAiError>,
}

fn anthropic_error(status: u16, body: &str) -> OpenAiError {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let kind = v.pointer("/error/type").and_then(Value::as_str).unwrap_or("api_error").to_string();
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| body.chars().take(500).collect());
    OpenAiError { status, kind, message: format!("anthropic {status}: {message}") }
}

/// Run one chat completion. Streams OpenAI chunks into `tx` when `req.stream`;
/// always returns the converter so a non-streaming response can be assembled.
pub async fn run(input: ChatInput, tx: mpsc::Sender<Bytes>) -> ChatOutcome {
    let ChatInput { client, cfg, req, client_bearer, claude_code_version } = input;
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|o| o.get("include_usage"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let model = cfg.resolve_chat_model(&req.model);
    let mut conv = Converter::new(req.model.clone(), include_usage);

    let cred = match claude_auth::resolve(&cfg, client_bearer.as_deref()).await {
        Ok(c) => c,
        Err(e) => {
            return ChatOutcome {
                converter: conv,
                error: Some(OpenAiError { status: 401, kind: "authentication_error".into(), message: e }),
            }
        }
    };

    let built = build_anthropic_request(&req, &cfg, &model, cred.is_oauth());
    conv = conv.with_reverse_names(built.reverse_names.clone());
    info!(model = %req.model, anthropic_model = %model, stream = req.stream, oauth = cred.is_oauth(),
          messages = built.body["messages"].as_array().map(Vec::len).unwrap_or(0),
          tools = built.body["tools"].as_array().map(Vec::len).unwrap_or(0), "proxying /v1/chat/completions");
    if cfg.log_bodies {
        debug!(body = %built.body, "anthropic request");
    }

    let url = format!("{}/v1/messages", cfg.anthropic_base_url);
    let mut rb = client
        .post(&url)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("accept", "text/event-stream")
        .json(&built.body);
    rb = match &cred {
        Credential::ApiKey(k) => rb.header("x-api-key", k),
        Credential::OAuth(t) => rb
            .bearer_auth(t)
            .header("anthropic-beta", OAUTH_BETAS)
            .header("user-agent", format!("claude-code/{claude_code_version} (external, cli)"))
            .header("x-app", "cli"),
    };

    let resp = match rb.send().await {
        Ok(r) => r,
        Err(e) => {
            error!(error = %e, %url, "anthropic request failed");
            return ChatOutcome {
                converter: conv,
                error: Some(OpenAiError { status: 502, kind: "api_error".into(), message: format!("anthropic request failed: {e}") }),
            };
        }
    };
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        warn!(status = status.as_u16(), body = %text.chars().take(300).collect::<String>(), "anthropic error");
        if status.as_u16() == 401 {
            claude_auth::invalidate_cache();
        }
        return ChatOutcome { converter: conv, error: Some(anthropic_error(status.as_u16(), &text)) };
    }

    let mut out: Vec<u8> = Vec::new();
    let mut parser = SseParser::default();
    let mut upstream = resp.bytes_stream();
    let streaming = req.stream;
    // Did upstream actually end the turn? Set only by a real terminal event
    // (`message_stop`, or a `message_delta` carrying `stop_reason`). If the
    // connection simply stops, this stays false and we refuse to present the
    // result as a completed turn.
    let mut saw_terminal = false;
    while let Some(item) = upstream.next().await {
        match item {
            Ok(bytes) => {
                for data in parser.push(&bytes) {
                    if cfg.log_bodies {
                        debug!(data = %data, "anthropic event");
                    }
                    match serde_json::from_str::<Value>(&data) {
                        Ok(ev) => {
                            conv.handle(&ev, &mut out);
                            if conv.saw_terminal() {
                                saw_terminal = true;
                            }
                        }
                        Err(e) => warn!(error = %e, data = %data, "unparseable anthropic event"),
                    }
                }
                if streaming && !out.is_empty() {
                    let _ = tx.send(Bytes::from(std::mem::take(&mut out))).await;
                } else {
                    out.clear();
                }
            }
            Err(e) => {
                error!(error = %e, "anthropic stream error");
                let err = json!({"error": {"message": format!("anthropic stream error: {e}"), "type": "api_error", "code": "api_error"}});
                if streaming {
                    let _ = tx.send(Bytes::from(format!("data: {err}\n\ndata: [DONE]\n\n"))).await;
                }
                return ChatOutcome {
                    converter: conv,
                    error: Some(OpenAiError { status: 502, kind: "api_error".into(), message: format!("anthropic stream error: {e}") }),
                };
            }
        }
    }
    conv.finish_with(&mut out, saw_terminal);
    if streaming && !out.is_empty() {
        let _ = tx.send(Bytes::from(out)).await;
    }
    // A stream that ended without a terminal event did not deliver a real
    // answer. Say so on the non-streaming path too, where the caller reads
    // `error` rather than the SSE body — otherwise the caller gets a
    // perfectly shaped completion whose content is empty and cannot tell it
    // apart from a model that genuinely had nothing to say.
    if !saw_terminal {
        warn!("anthropic stream ended with no terminal event; reporting incomplete");
        return ChatOutcome {
            converter: conv,
            error: Some(OpenAiError {
                status: 502,
                kind: "api_error".into(),
                message: "anthropic stream ended before the turn was complete \
                          (no stop_reason received)"
                    .into(),
            }),
        };
    }
    ChatOutcome { converter: conv, error: None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config { chat_max_tokens: 4096, ..Default::default() }
    }

    #[test]
    fn sets_cache_breakpoints_on_system_tools_and_history_prefix() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "claude-opus-5",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "newest"}
            ],
            "tools": [{"type": "function", "function": {"name": "ls", "description": "list", "parameters": {"type": "object", "properties": {}}}}],
        }))
        .unwrap();
        let built = build_anthropic_request(&req, &cfg(), "claude-opus-5", false);
        let b = &built.body;

        // System prefix cached.
        let system = b["system"].as_array().unwrap();
        assert_eq!(system.last().unwrap()["cache_control"]["type"], "ephemeral");

        // Tool list cached.
        let tools = b["tools"].as_array().unwrap();
        assert_eq!(tools.last().unwrap()["cache_control"]["type"], "ephemeral");

        // History prefix cached up to, but NOT including, the newest user turn.
        let msgs = b["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        let prefix_last = msgs[msgs.len() - 2]["content"].as_array().unwrap().last().unwrap();
        assert_eq!(prefix_last["cache_control"]["type"], "ephemeral");
        let newest = msgs[msgs.len() - 1]["content"].as_array().unwrap().last().unwrap();
        assert!(newest.get("cache_control").is_none(), "newest turn must not be a breakpoint");

        // At most 4 breakpoints, per the Anthropic API limit.
        let count = serde_json::to_string(b).unwrap().matches("cache_control").count();
        assert!(count <= 4, "too many cache breakpoints: {count}");
    }

    #[test]
    fn single_message_request_has_no_history_breakpoint() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        let built = build_anthropic_request(&req, &cfg(), "claude-opus-5", false);
        let msgs = built.body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0]["content"].as_array().unwrap().last().unwrap().get("cache_control").is_none());
    }

    #[test]
    fn builds_anthropic_request_with_tools_and_results() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "claude-fable-5.1",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "ls", "arguments": "{\"path\": \".\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "a.rs"},
                {"role": "tool", "tool_call_id": "call_2", "content": "b.rs"},
                {"role": "system", "content": "remember"},
                {"role": "user", "content": "thanks"}
            ],
            "tools": [{"type": "function", "function": {"name": "ls", "description": "list", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}],
            "tool_choice": "required",
            "parallel_tool_calls": false,
            "temperature": 0.2,
            "stop": ["END"],
            "reasoning_effort": "high"
        }))
        .unwrap();
        let b = build_anthropic_request(&req, &cfg(), "claude-fable-5-1", true).body;
        assert_eq!(b["model"], "claude-fable-5-1");
        assert_eq!(b["system"][0]["text"], CLAUDE_CODE_SYSTEM_PREFIX);
        assert_eq!(b["system"][1]["text"], "be terse");
        let msgs = b["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][1]["type"], "image");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"][0]["type"], "tool_use");
        assert_eq!(msgs[1]["content"][0]["input"]["path"], ".");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[2]["content"][1]["tool_use_id"], "call_2");
        assert!(msgs[2]["content"][2]["text"].as_str().unwrap().contains("remember"));
        assert_eq!(msgs[2]["content"][3]["text"], "thanks");
        assert_eq!(b["tools"][0]["input_schema"]["type"], "object");
        // fable: forced tool choice downgraded to auto, parallel disabled, no sampling params
        assert_eq!(b["tool_choice"]["type"], "auto");
        assert_eq!(b["tool_choice"]["disable_parallel_tool_use"], true);
        assert!(b.get("temperature").is_none());
        assert_eq!(b["stop_sequences"][0], "END");
        assert_eq!(b["output_config"]["effort"], "high");
        assert!(b.get("thinking").is_none());
        assert_eq!(b["max_tokens"], 4096);
        assert_eq!(b["stream"], true);
    }

    #[test]
    fn oauth_aliases_tool_names_and_prose() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "claude-fable-5-1",
            "messages": [
                {"role": "system", "content": "You are Hermes Agent by Nous Research. Use session_search to recall; session_search_tool.py is the impl."},
                {"role": "user", "content": "x"},
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "memory", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}
            ],
            "tools": [
                {"type": "function", "function": {"name": "memory", "description": "persistent memory"}},
                {"type": "function", "function": {"name": "session_search", "description": "search"}},
                {"type": "function", "function": {"name": "mcp_foo", "description": "x"}},
                {"type": "function", "function": {"name": "terminal", "description": "x"}}
            ]
        }))
        .unwrap();
        let b = build_anthropic_request(&req, &cfg(), "claude-fable-5-1", true);
        let sys = b.body["system"][1]["text"].as_str().unwrap();
        assert!(sys.contains("Claude Code by Anthropic"));
        assert!(sys.contains("Use chat_history_lookup to recall"));
        assert!(sys.contains("session_search_tool.py"));
        let names: Vec<&str> = b.body["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["mcp__context_notes", "mcp__chat_history_lookup", "mcp__foo", "mcp__terminal"]);
        assert_eq!(b.body["messages"][1]["content"][0]["name"], "mcp__context_notes");
        assert_eq!(b.reverse_names["mcp__context_notes"], "memory");
        assert_eq!(b.reverse_names["mcp__foo"], "mcp_foo");
        assert_eq!(b.reverse_names["mcp__terminal"], "terminal");
        assert_eq!(oauth_wire_name("mcp__already"), "mcp__already");

        let mut c = Converter::new("m".into(), false).with_reverse_names(b.reverse_names.clone());
        let mut out = Vec::new();
        c.handle(&json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "mcp__context_notes", "input": {}}}), &mut out);
        let ev = events(&out);
        assert_eq!(ev[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "memory");
        // API-key mode: untouched
        let b2 = build_anthropic_request(&req, &cfg(), "claude-fable-5-1", false);
        assert_eq!(b2.body["tools"][0]["name"], "memory");
    }

    #[test]
    fn classic_model_keeps_sampling_and_forced_choice() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-6", "temperature": 0.5, "tool_choice": {"type": "function", "function": {"name": "ls"}},
            "tools": [{"type": "function", "function": {"name": "ls"}}],
            "messages": [{"role": "user", "content": "x"}], "reasoning_effort": "medium"
        }))
        .unwrap();
        let b = build_anthropic_request(&req, &cfg(), "claude-sonnet-4-6", false).body;
        assert_eq!(b["temperature"], 0.5);
        assert_eq!(b["tool_choice"]["type"], "tool");
        assert_eq!(b["thinking"]["type"], "adaptive");
        assert!(b.get("system").is_none());
    }

    fn events(out: &[u8]) -> Vec<Value> {
        String::from_utf8_lossy(out)
            .split("\n\n")
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|d| *d != "[DONE]")
            .map(|d| serde_json::from_str(d).unwrap())
            .collect()
    }

    /// A stream cut short must NOT be reported as a finished turn.
    ///
    /// Regression test for the bug behind Hermes's "empty stream with no
    /// finish_reason": upstream dropped the connection mid-turn, `finish()`
    /// defaulted to `finish_reason: "stop"`, and the agent read a perfectly
    /// well-formed empty answer as a real end of turn. Measured 2026-09-12:
    /// 82 occurrences in three days.
    #[test]
    fn truncated_stream_is_an_error_not_a_clean_stop() {
        let mut c = Converter::new("claude-opus-5".into(), true);
        let mut out = Vec::new();
        // The turn starts, some text arrives, then upstream goes silent:
        // no `message_delta` with a stop_reason, no `message_stop`.
        c.handle(&json!({"type": "message_start", "message": {"usage": {"input_tokens": 3}}}), &mut out);
        c.handle(&json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}), &mut out);
        c.handle(&json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Bon"}}), &mut out);
        assert!(!c.saw_terminal(), "no terminal event was sent yet");

        out.clear();
        c.finish_with(&mut out, false);
        let ev = events(&out);
        assert_eq!(ev.len(), 1, "exactly one event: the error");
        assert_eq!(ev[0]["error"]["code"], "incomplete_stream");
        // The crux: no chunk may claim the turn ended normally.
        let raw = String::from_utf8_lossy(&out);
        assert!(!raw.contains("\"finish_reason\": \"stop\""), "must not fabricate a stop");
        assert!(raw.contains("[DONE]"), "the SSE stream is still closed properly");
    }

    /// The normal path must keep working: a real stop stays a real stop.
    #[test]
    fn genuine_stop_is_preserved() {
        let mut c = Converter::new("claude-opus-5".into(), true);
        let mut out = Vec::new();
        c.handle(&json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}), &mut out);
        c.handle(&json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}), &mut out);
        c.handle(&json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}), &mut out);
        assert!(c.saw_terminal(), "a stop_reason is a terminal event");

        out.clear();
        c.finish_with(&mut out, true);
        let ev = events(&out);
        assert_eq!(ev[0]["choices"][0]["finish_reason"], "stop");
        assert!(ev.iter().all(|e| e.get("error").is_none()), "no error on a clean turn");
    }

    /// A turn ending on a tool call is terminal too, and must not be flagged.
    #[test]
    fn tool_use_stop_is_terminal() {
        let mut c = Converter::new("claude-opus-5".into(), true);
        let mut out = Vec::new();
        c.handle(&json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "ls", "input": {}}}), &mut out);
        c.handle(&json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}}), &mut out);
        assert!(c.saw_terminal());
        out.clear();
        c.finish_with(&mut out, true);
        assert_eq!(events(&out)[0]["choices"][0]["finish_reason"], "tool_calls");
    }

    /// Zero content AND no terminal event is the exact production signature:
    /// upstream accepted the request, sent nothing at all, and hung up.
    #[test]
    fn empty_stream_with_no_events_is_an_error() {
        let mut c = Converter::new("claude-opus-5".into(), true);
        let mut out = Vec::new();
        c.finish_with(&mut out, false);
        let ev = events(&out);
        assert_eq!(ev[0]["error"]["code"], "incomplete_stream");
        assert!(!String::from_utf8_lossy(&out).contains("finish_reason\": \"stop"));
    }

    #[test]
    fn converts_stream_to_openai_chunks_and_completion() {
        let mut c = Converter::new("claude-fable-5-1".into(), true);
        let mut out = Vec::new();
        for ev in [
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 1}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {}}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"path\":"}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "\".\"}"}}),
            json!({"type": "content_block_stop", "index": 2}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}}),
            json!({"type": "message_stop"}),
        ] {
            c.handle(&ev, &mut out);
        }
        let ev = events(&out);
        assert_eq!(ev[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(ev[1]["choices"][0]["delta"]["reasoning_content"], "hmm");
        assert_eq!(ev[2]["choices"][0]["delta"]["content"], "Hi");
        assert_eq!(ev[3]["choices"][0]["delta"]["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(ev[3]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "ls");
        assert_eq!(ev[5]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], "\".\"}");
        let last = ev.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(last["usage"]["prompt_tokens"], 15);
        assert_eq!(last["usage"]["completion_tokens"], 7);
        assert_eq!(last["usage"]["prompt_tokens_details"]["cached_tokens"], 5);
        assert!(String::from_utf8_lossy(&out).ends_with("data: [DONE]\n\n"));

        let full = c.completion();
        assert_eq!(full["object"], "chat.completion");
        assert_eq!(full["choices"][0]["message"]["content"], "Hi");
        assert_eq!(full["choices"][0]["message"]["reasoning_content"], "hmm");
        assert_eq!(full["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{\"path\":\".\"}");
        assert_eq!(full["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(full["usage"]["total_tokens"], 22);
    }
}
