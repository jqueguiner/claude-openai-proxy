//! Translate an OpenAI chat-completions SSE stream into Anthropic Messages
//! streaming events (`message_start`, `content_block_*`, `message_delta`, ...).
//!
//! The translator also records every finished content block, so a non-streaming
//! Anthropic response can be assembled from the same code path, and it supports
//! several upstream "rounds" inside one Anthropic message (server-tool loop).

use std::collections::{BTreeMap, HashSet};

use serde_json::{json, Value};
use tracing::warn;

use crate::anthropic::{ServerToolUsage, Usage};
use crate::convert::{map_stop_reason, new_call_id, parse_tool_arguments};
use crate::openai::{value_to_text, ChatChunk, DeltaToolCall, OaUsage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Current {
    None,
    Text(u32),
    Thinking(u32),
    Tool(u32),
}

#[derive(Debug, Default)]
struct ToolState {
    block: Option<u32>,
    /// Upstream (OpenAI) call id – what must go back in `tool_call_id`.
    oa_id: String,
    /// Id exposed to the Anthropic client (`toolu_...` passthrough or `srvtoolu_...`).
    block_id: String,
    name: String,
    pending_args: String,
    full_args: String,
    emitted_args: bool,
    is_server: bool,
}

/// A tool call completed in one upstream round.
#[derive(Debug, Clone)]
pub struct CompletedCall {
    pub oa_id: String,
    pub block_id: String,
    pub name: String,
    pub arguments: String,
    pub is_server: bool,
}

/// What one upstream round produced.
#[derive(Debug, Clone, Default)]
pub struct RoundSummary {
    pub finish_reason: Option<String>,
    pub text: String,
    pub calls: Vec<CompletedCall>,
}

pub struct Translator {
    id: String,
    model: String,
    next_index: u32,
    current: Current,
    tools: BTreeMap<u32, ToolState>,
    next_tool_key: u32,
    finish_reason: Option<String>,
    round_usage: Option<OaUsage>,
    round_text: String,
    cur_text: String,
    cur_thinking: String,
    /// Tool names that are executed by the proxy and surfaced as `server_tool_use`.
    server_tools: HashSet<String>,
    client_tool_seen: bool,
    finished: bool,
    total_usage: Usage,
    web_search_requests: u64,
    blocks: Vec<Value>,
}

fn emit(out: &mut Vec<u8>, event: &str, data: &Value) {
    out.extend_from_slice(b"event: ");
    out.extend_from_slice(event.as_bytes());
    out.extend_from_slice(b"\ndata: ");
    out.extend_from_slice(data.to_string().as_bytes());
    out.extend_from_slice(b"\n\n");
}

impl Translator {
    pub fn new(id: String, model: String) -> Self {
        Self {
            id,
            model,
            next_index: 0,
            current: Current::None,
            tools: BTreeMap::new(),
            next_tool_key: 0,
            finish_reason: None,
            round_usage: None,
            round_text: String::new(),
            cur_text: String::new(),
            cur_thinking: String::new(),
            server_tools: HashSet::new(),
            client_tool_seen: false,
            finished: false,
            total_usage: Usage::default(),
            web_search_requests: 0,
            blocks: Vec::new(),
        }
    }

    pub fn with_server_tools(mut self, names: impl IntoIterator<Item = String>) -> Self {
        self.server_tools = names.into_iter().collect();
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// Finished content blocks so far (Anthropic shapes).
    pub fn blocks(&self) -> &[Value] {
        &self.blocks
    }

    pub fn usage(&self) -> Usage {
        let mut u = self.total_usage.clone();
        if self.web_search_requests > 0 {
            u.server_tool_use = Some(ServerToolUsage { web_search_requests: self.web_search_requests });
        }
        u
    }

    pub fn note_web_search(&mut self) {
        self.web_search_requests += 1;
    }

    /// `message_start` – sent before any upstream data arrives.
    pub fn start(&self, out: &mut Vec<u8>) {
        emit(
            out,
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
            }),
        );
    }

    pub fn ping(&self, out: &mut Vec<u8>) {
        emit(out, "ping", &json!({"type": "ping"}));
    }

    pub fn handle_chunk(&mut self, chunk: ChatChunk, out: &mut Vec<u8>) {
        if self.finished {
            return;
        }
        if let Some(err) = chunk.error {
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| err.to_string());
            self.error(&msg, out);
            return;
        }
        for choice in chunk.choices {
            let d = choice.delta;
            if let Some(r) = d.reasoning_content.or(d.reasoning) {
                if !r.is_empty() {
                    let idx = self.ensure_thinking(out);
                    self.cur_thinking.push_str(&r);
                    emit(
                        out,
                        "content_block_delta",
                        &json!({"type": "content_block_delta", "index": idx,
                                "delta": {"type": "thinking_delta", "thinking": r}}),
                    );
                }
            }
            let text = value_to_text(d.content);
            if !text.is_empty() {
                let idx = self.ensure_text(out);
                self.cur_text.push_str(&text);
                self.round_text.push_str(&text);
                emit(
                    out,
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": idx,
                            "delta": {"type": "text_delta", "text": text}}),
                );
            }
            for tc in d.tool_calls.unwrap_or_default() {
                self.handle_tool_delta(tc, out);
            }
            if let Some(fr) = choice.finish_reason {
                self.finish_reason = Some(fr);
            }
        }
        if let Some(u) = chunk.usage {
            self.round_usage = Some(u);
        }
    }

    /// Append a complete block produced by the proxy itself (e.g. `web_search_tool_result`).
    pub fn push_block(&mut self, mut block: Value, out: &mut Vec<u8>) {
        self.close_current(out);
        let idx = self.alloc_index();
        emit(
            out,
            "content_block_start",
            &json!({"type": "content_block_start", "index": idx, "content_block": block}),
        );
        emit(out, "content_block_stop", &json!({"type": "content_block_stop", "index": idx}));
        if let Some(obj) = block.as_object_mut() {
            obj.remove("index");
        }
        self.blocks.push(block);
    }

    fn ensure_text(&mut self, out: &mut Vec<u8>) -> u32 {
        if let Current::Text(i) = self.current {
            return i;
        }
        self.close_current(out);
        let idx = self.alloc_index();
        emit(
            out,
            "content_block_start",
            &json!({"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}}),
        );
        self.current = Current::Text(idx);
        idx
    }

    fn ensure_thinking(&mut self, out: &mut Vec<u8>) -> u32 {
        if let Current::Thinking(i) = self.current {
            return i;
        }
        self.close_current(out);
        let idx = self.alloc_index();
        emit(
            out,
            "content_block_start",
            &json!({"type": "content_block_start", "index": idx,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
        );
        self.current = Current::Thinking(idx);
        idx
    }

    fn alloc_index(&mut self) -> u32 {
        let i = self.next_index;
        self.next_index += 1;
        i
    }

    fn handle_tool_delta(&mut self, tc: DeltaToolCall, out: &mut Vec<u8>) {
        let key = match tc.index {
            Some(i) => i,
            None => match (tc.id.is_some(), self.current) {
                (false, Current::Tool(k)) => k,
                _ => self.next_tool_key,
            },
        };
        self.next_tool_key = self.next_tool_key.max(key + 1);

        {
            let st = self.tools.entry(key).or_default();
            if let Some(id) = tc.id {
                if st.block.is_none() && !id.is_empty() {
                    st.oa_id = id;
                }
            }
            if let Some(f) = tc.function {
                if let Some(n) = f.name {
                    if st.block.is_none() {
                        st.name.push_str(&n);
                    }
                }
                if let Some(a) = f.arguments {
                    st.pending_args.push_str(&a);
                    st.full_args.push_str(&a);
                }
            }
        }

        let needs_open = {
            let st = &self.tools[&key];
            st.block.is_none() && !st.name.is_empty()
        };
        if needs_open {
            self.open_tool(key, out);
        }
        self.flush_tool_args(key, out);
    }

    fn open_tool(&mut self, key: u32, out: &mut Vec<u8>) {
        self.close_current(out);
        let idx = self.alloc_index();
        let is_server = self.server_tools.contains(&self.tools[&key].name);
        let st = self.tools.get_mut(&key).expect("tool state exists");
        if st.oa_id.is_empty() {
            st.oa_id = new_call_id();
        }
        if st.name.is_empty() {
            st.name = "unknown_tool".into();
        }
        st.is_server = is_server;
        st.block_id = if is_server {
            format!("srvtoolu_{}", uuid::Uuid::new_v4().simple())
        } else {
            st.oa_id.clone()
        };
        st.block = Some(idx);
        let kind = if is_server { "server_tool_use" } else { "tool_use" };
        emit(
            out,
            "content_block_start",
            &json!({"type": "content_block_start", "index": idx,
                    "content_block": {"type": kind, "id": st.block_id, "name": st.name, "input": {}}}),
        );
        self.current = Current::Tool(key);
        if !is_server {
            self.client_tool_seen = true;
        }
    }

    fn flush_tool_args(&mut self, key: u32, out: &mut Vec<u8>) {
        let st = self.tools.get_mut(&key).expect("tool state exists");
        if let Some(idx) = st.block {
            if !st.pending_args.is_empty() {
                emit(
                    out,
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": idx,
                            "delta": {"type": "input_json_delta", "partial_json": st.pending_args}}),
                );
                st.pending_args.clear();
                st.emitted_args = true;
            }
        }
    }

    fn close_current(&mut self, out: &mut Vec<u8>) {
        match self.current {
            Current::None => {}
            Current::Text(i) => {
                emit(out, "content_block_stop", &json!({"type": "content_block_stop", "index": i}));
                let text = std::mem::take(&mut self.cur_text);
                self.blocks.push(json!({"type": "text", "text": text}));
            }
            Current::Thinking(i) => {
                emit(
                    out,
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": i,
                            "delta": {"type": "signature_delta", "signature": ""}}),
                );
                emit(out, "content_block_stop", &json!({"type": "content_block_stop", "index": i}));
                let thinking = std::mem::take(&mut self.cur_thinking);
                self.blocks.push(json!({"type": "thinking", "thinking": thinking, "signature": ""}));
            }
            Current::Tool(k) => {
                let st = self.tools.get_mut(&k).expect("tool state exists");
                let idx = st.block.expect("open tool has a block index");
                if !st.emitted_args {
                    // Claude Code JSON-parses the accumulated partial_json; make it valid.
                    emit(
                        out,
                        "content_block_delta",
                        &json!({"type": "content_block_delta", "index": idx,
                                "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
                    );
                    st.emitted_args = true;
                }
                emit(out, "content_block_stop", &json!({"type": "content_block_stop", "index": idx}));
                let kind = if st.is_server { "server_tool_use" } else { "tool_use" };
                self.blocks.push(json!({
                    "type": kind, "id": st.block_id, "name": st.name,
                    "input": parse_tool_arguments(&st.full_args)
                }));
            }
        }
        self.current = Current::None;
    }

    /// Close the current upstream round: flush open blocks, fold usage into the
    /// totals and hand back the tool calls the model made. The message stays open.
    pub fn end_round(&mut self, out: &mut Vec<u8>) -> RoundSummary {
        // Tool calls whose name never arrived: open them anyway so nothing is lost.
        let unopened: Vec<u32> = self
            .tools
            .iter()
            .filter(|(_, s)| s.block.is_none() && (!s.pending_args.is_empty() || !s.oa_id.is_empty()))
            .map(|(k, _)| *k)
            .collect();
        for k in unopened {
            warn!(key = k, "tool call finished without a function name");
            self.open_tool(k, out);
            self.flush_tool_args(k, out);
        }
        self.close_current(out);

        let calls = std::mem::take(&mut self.tools)
            .into_values()
            .filter(|s| s.block.is_some())
            .map(|s| CompletedCall {
                oa_id: s.oa_id,
                block_id: s.block_id,
                name: s.name,
                arguments: s.full_args,
                is_server: s.is_server,
            })
            .collect();
        self.next_tool_key = 0;

        if let Some(u) = self.round_usage.take() {
            let round = crate::convert::map_usage(Some(&u));
            self.total_usage.input_tokens += round.input_tokens;
            self.total_usage.output_tokens += round.output_tokens;
            if let Some(c) = round.cache_read_input_tokens {
                self.total_usage.cache_read_input_tokens =
                    Some(self.total_usage.cache_read_input_tokens.unwrap_or(0) + c);
            }
        }

        RoundSummary {
            finish_reason: self.finish_reason.take(),
            text: std::mem::take(&mut self.round_text),
            calls,
        }
    }

    /// Close open blocks and emit `message_delta` + `message_stop`.
    pub fn finish(&mut self, out: &mut Vec<u8>) {
        self.finish_with(out, None);
    }

    /// Like `finish`, with an explicit stop reason (`pause_turn`, ...). Returns the stop reason used.
    pub fn finish_with(&mut self, out: &mut Vec<u8>, stop_reason: Option<&str>) -> String {
        if self.finished {
            return String::new();
        }
        let summary = self.end_round(out);
        let stop_reason = match stop_reason {
            Some(s) => s.to_string(),
            None => map_stop_reason(summary.finish_reason.as_deref(), self.client_tool_seen),
        };
        emit(
            out,
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": self.usage()
            }),
        );
        emit(out, "message_stop", &json!({"type": "message_stop"}));
        self.finished = true;
        stop_reason
    }

    /// Emit an Anthropic `error` event and mark the stream as finished.
    pub fn error(&mut self, message: &str, out: &mut Vec<u8>) {
        if self.finished {
            return;
        }
        emit(
            out,
            "error",
            &json!({"type": "error", "error": {"type": "api_error", "message": message}}),
        );
        self.finished = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(v: Value) -> ChatChunk {
        serde_json::from_value(v).unwrap()
    }

    fn events(out: &[u8]) -> Vec<(String, Value)> {
        String::from_utf8_lossy(out)
            .split("\n\n")
            .filter(|b| !b.trim().is_empty())
            .map(|b| {
                let mut ev = String::new();
                let mut data = Value::Null;
                for line in b.lines() {
                    if let Some(e) = line.strip_prefix("event: ") {
                        ev = e.to_string();
                    } else if let Some(d) = line.strip_prefix("data: ") {
                        data = serde_json::from_str(d).unwrap();
                    }
                }
                (ev, data)
            })
            .collect()
    }

    #[test]
    fn text_then_tool_call_stream() {
        let mut t = Translator::new("msg_1".into(), "claude-opus-5".into());
        let mut out = Vec::new();
        t.start(&mut out);
        t.handle_chunk(chunk(json!({"choices": [{"delta": {"role": "assistant", "content": ""}}]})), &mut out);
        t.handle_chunk(chunk(json!({"choices": [{"delta": {"content": "Hel"}}]})), &mut out);
        t.handle_chunk(chunk(json!({"choices": [{"delta": {"content": "lo"}}]})), &mut out);
        t.handle_chunk(
            chunk(json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                "function": {"name": "Read", "arguments": ""}}]}}]})),
            &mut out,
        );
        t.handle_chunk(
            chunk(json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\":"}}]}}]})),
            &mut out,
        );
        t.handle_chunk(
            chunk(json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"a\"}"}}]}, "finish_reason": "tool_calls"}]})),
            &mut out,
        );
        t.handle_chunk(chunk(json!({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})), &mut out);
        t.finish(&mut out);

        let ev = events(&out);
        let names: Vec<&str> = ev.iter().map(|(e, _)| e.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert_eq!(ev[5].1["content_block"]["type"], "tool_use");
        assert_eq!(ev[5].1["content_block"]["name"], "Read");
        assert_eq!(ev[5].1["index"], 1);
        let args: String = ev[6..8].iter().map(|(_, d)| d["delta"]["partial_json"].as_str().unwrap()).collect();
        assert_eq!(args, "{\"path\":\"a\"}");
        assert_eq!(ev[9].1["delta"]["stop_reason"], "tool_use");
        assert_eq!(ev[9].1["usage"]["output_tokens"], 3);
        assert_eq!(ev[9].1["usage"]["input_tokens"], 7);
    }

    #[test]
    fn tool_call_without_arguments_gets_empty_object() {
        let mut t = Translator::new("m".into(), "x".into());
        let mut out = Vec::new();
        t.handle_chunk(
            chunk(json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "Ping"}}]}}]})),
            &mut out,
        );
        t.finish(&mut out);
        let ev = events(&out);
        assert_eq!(ev[1].1["delta"]["partial_json"], "{}");
        assert_eq!(ev[3].1["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn reasoning_becomes_thinking_block() {
        let mut t = Translator::new("m".into(), "x".into());
        let mut out = Vec::new();
        t.handle_chunk(chunk(json!({"choices": [{"delta": {"reasoning_content": "think"}}]})), &mut out);
        t.handle_chunk(chunk(json!({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]})), &mut out);
        t.finish(&mut out);
        let ev = events(&out);
        assert_eq!(ev[0].1["content_block"]["type"], "thinking");
        assert_eq!(ev[1].1["delta"]["type"], "thinking_delta");
        assert_eq!(ev[2].1["delta"]["type"], "signature_delta");
        assert_eq!(ev[4].1["content_block"]["type"], "text");
        assert_eq!(ev.last().unwrap().0, "message_stop");
    }

    #[test]
    fn error_event_terminates_stream() {
        let mut t = Translator::new("m".into(), "x".into());
        let mut out = Vec::new();
        t.handle_chunk(chunk(json!({"error": {"message": "boom"}})), &mut out);
        t.finish(&mut out);
        let ev = events(&out);
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].0, "error");
        assert_eq!(ev[0].1["error"]["message"], "boom");
    }
}
