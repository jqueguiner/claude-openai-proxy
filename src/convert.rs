//! Request / response translation between the Anthropic Messages API and
//! OpenAI Chat Completions (non-streaming). Streaming lives in `stream.rs`.

use serde_json::{json, Value};
use tracing::warn;

use crate::anthropic::{
    ContentBlock, MessageContent, MessagesRequest, MessagesResponse, ResponseBlock, SystemPrompt,
    ToolResultContent, Usage,
};
use crate::config::Config;
use crate::openai::{
    value_to_text, ChatMessage, ChatRequest, ChatResponse, FunctionCall, OaUsage, ToolCall,
};

pub fn new_message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}

pub fn new_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4().simple())
}

// ---------------------------------------------------------------------------
// Anthropic -> OpenAI
// ---------------------------------------------------------------------------

pub fn build_chat_request(req: &MessagesRequest, cfg: &Config, upstream_model: &str) -> ChatRequest {
    let mut messages: Vec<ChatMessage> = Vec::new();

    if let Some(system) = &req.system {
        let text = system_text(system);
        if !text.trim().is_empty() {
            messages.push(ChatMessage {
                role: "system",
                content: Some(Value::String(text)),
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    for m in &req.messages {
        match m.role.as_str() {
            "user" => convert_user(&m.content, &mut messages),
            "assistant" => convert_assistant(&m.content, &mut messages),
            // Mid-conversation operator instructions (Claude Code sends these after tool results).
            "system" => {
                let text = content_text(&m.content);
                if text.trim().is_empty() {
                    continue;
                }
                if cfg.mid_system_as_system {
                    messages.push(ChatMessage {
                        role: "system",
                        content: Some(Value::String(text)),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                } else {
                    merge_into_user_turn(text, &mut messages);
                }
            }
            other => warn!(role = other, "dropping message with unknown role"),
        }
    }

    let tools: Vec<Value> = req
        .tools
        .iter()
        .filter(|t| match t.kind.as_deref() {
            None | Some("custom") => true,
            Some(_) if t.is_web_search() => false, // emulated by the websearch module
            Some(kind) => {
                warn!(tool = %t.name, kind, "dropping Anthropic server-side tool (not expressible in OpenAI API)");
                false
            }
        })
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description.clone().unwrap_or_default(),
                    "parameters": t.input_schema.clone().unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                }
            })
        })
        .collect();

    let (tool_choice, parallel_tool_calls) = if tools.is_empty() {
        (None, None)
    } else {
        match &req.tool_choice {
            None => (None, None),
            Some(tc) => {
                let choice = match tc.kind.as_str() {
                    "auto" => Some(json!("auto")),
                    "any" => Some(json!("required")),
                    "none" => Some(json!("none")),
                    "tool" => tc
                        .name
                        .as_ref()
                        .map(|n| json!({"type": "function", "function": {"name": n}})),
                    _ => None,
                };
                let parallel = if tc.disable_parallel_tool_use { Some(false) } else { None };
                (choice, parallel)
            }
        }
    };

    let max_tokens = match (req.max_tokens, cfg.max_tokens_cap) {
        (Some(m), Some(cap)) => Some(m.min(cap)),
        (Some(m), None) => Some(m),
        (None, cap) => cap,
    };

    ChatRequest {
        model: upstream_model.to_string(),
        messages,
        max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: if req.stop_sequences.is_empty() { None } else { Some(req.stop_sequences.clone()) },
        stream: req.stream,
        stream_options: if req.stream { Some(json!({"include_usage": true})) } else { None },
        tools: if tools.is_empty() { None } else { Some(tools) },
        tool_choice,
        parallel_tool_calls,
        reasoning_effort: reasoning_effort(req, cfg),
    }
}

fn reasoning_effort(req: &MessagesRequest, cfg: &Config) -> Option<String> {
    match cfg.reasoning_effort.as_deref() {
        None => None,
        Some("auto") => {
            let effort = req.output_config.as_ref().and_then(|o| o.effort.clone());
            match effort.as_deref() {
                Some("low") => Some("low".into()),
                Some("medium") => Some("medium".into()),
                Some("high") | Some("xhigh") | Some("max") => Some("high".into()),
                _ => {
                    let thinking_on = req
                        .thinking
                        .as_ref()
                        .and_then(|t| t.get("type"))
                        .and_then(Value::as_str)
                        .map(|t| t != "disabled")
                        .unwrap_or(false);
                    if thinking_on { Some("medium".into()) } else { None }
                }
            }
        }
        Some(fixed) => Some(fixed.to_string()),
    }
}

/// Fold an operator instruction into the current user turn so the upstream
/// conversation keeps strict system/user/assistant ordering.
fn merge_into_user_turn(text: String, out: &mut Vec<ChatMessage>) {
    let wrapped = format!("<system-instruction>\n{text}\n</system-instruction>");
    if let Some(last) = out.last_mut() {
        if last.role == "user" {
            match last.content.take() {
                Some(Value::String(s)) if !s.is_empty() => {
                    last.content = Some(Value::String(format!("{s}\n\n{wrapped}")));
                }
                Some(Value::Array(mut parts)) => {
                    parts.push(json!({"type": "text", "text": wrapped}));
                    last.content = Some(Value::Array(parts));
                }
                _ => last.content = Some(Value::String(wrapped)),
            }
            return;
        }
    }
    // Previous entry is a tool result or an assistant turn: a fresh user message keeps ordering valid.
    out.push(ChatMessage { role: "user", content: Some(Value::String(wrapped)), tool_calls: None, tool_call_id: None });
}

fn system_text(system: &SystemPrompt) -> String {
    match system {
        SystemPrompt::Text(s) => s.clone(),
        SystemPrompt::Blocks(blocks) => blocks_text(blocks),
    }
}

fn content_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Blocks(blocks) => blocks_text(blocks),
    }
}

fn blocks_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn convert_user(content: &MessageContent, out: &mut Vec<ChatMessage>) {
    let blocks = match content {
        MessageContent::Text(s) => {
            out.push(ChatMessage {
                role: "user",
                content: Some(Value::String(s.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
            return;
        }
        MessageContent::Blocks(b) => b,
    };

    // OpenAI content parts for the (single) user message we emit after any tool results.
    let mut parts: Vec<Value> = Vec::new();
    let mut has_image = false;

    for b in blocks {
        match b {
            ContentBlock::Text { text } => parts.push(json!({"type": "text", "text": text})),
            ContentBlock::Image { source } => {
                if let Some(url) = source.to_url() {
                    parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                    has_image = true;
                }
            }
            ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                let (mut text, images) = flatten_tool_result(content.as_ref());
                if text.is_empty() {
                    text = if *is_error { "Error: tool failed with no output".into() } else { "(no output)".into() };
                }
                out.push(ChatMessage {
                    role: "tool",
                    content: Some(Value::String(text)),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                });
                // Tool messages can only carry text; hand images over in the user turn.
                for url in images {
                    parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                    has_image = true;
                }
            }
            ContentBlock::Document {} => {
                parts.push(json!({"type": "text", "text": "[document attachment omitted by proxy]"}))
            }
            _ => {}
        }
    }

    if parts.is_empty() {
        return;
    }
    let content = if has_image {
        Value::Array(parts)
    } else {
        Value::String(
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    };
    out.push(ChatMessage { role: "user", content: Some(content), tool_calls: None, tool_call_id: None });
}

fn flatten_tool_result(content: Option<&ToolResultContent>) -> (String, Vec<String>) {
    match content {
        None => (String::new(), vec![]),
        Some(ToolResultContent::Text(s)) => (s.clone(), vec![]),
        Some(ToolResultContent::Blocks(blocks)) => {
            let mut texts = Vec::new();
            let mut images = Vec::new();
            for b in blocks {
                match b {
                    ContentBlock::Text { text } => texts.push(text.as_str()),
                    ContentBlock::Image { source } => {
                        if let Some(u) = source.to_url() {
                            images.push(u)
                        }
                    }
                    _ => {}
                }
            }
            (texts.join("\n"), images)
        }
    }
}

fn convert_assistant(content: &MessageContent, out: &mut Vec<ChatMessage>) {
    let blocks = match content {
        MessageContent::Text(s) => {
            if !s.is_empty() {
                out.push(ChatMessage {
                    role: "assistant",
                    content: Some(Value::String(s.clone())),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            return;
        }
        MessageContent::Blocks(b) => b,
    };

    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let flush = |text: &mut String, tool_calls: &mut Vec<ToolCall>, out: &mut Vec<ChatMessage>| {
        if text.is_empty() && tool_calls.is_empty() {
            return;
        }
        out.push(ChatMessage {
            role: "assistant",
            content: if text.is_empty() { None } else { Some(Value::String(std::mem::take(text))) },
            tool_calls: if tool_calls.is_empty() { None } else { Some(std::mem::take(tool_calls)) },
            tool_call_id: None,
        });
    };

    for b in blocks {
        match b {
            ContentBlock::Text { text: t } => {
                if !text.is_empty() {
                    text.push_str("\n\n");
                }
                text.push_str(t);
            }
            ContentBlock::ToolUse { id, name, input } | ContentBlock::ServerToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id: Some(id.clone()),
                    kind: "function".into(),
                    function: FunctionCall { name: name.clone(), arguments: input.to_string() },
                })
            }
            // Results of an earlier proxy-executed search: replay as a tool message right after
            // the assistant message that carries the matching call.
            ContentBlock::WebSearchToolResult { tool_use_id, content } => {
                flush(&mut text, &mut tool_calls, out);
                out.push(ChatMessage {
                    role: "tool",
                    content: Some(Value::String(crate::websearch::replay_result_text(content))),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                });
            }
            // Thinking blocks are model-bound; never replay them upstream.
            _ => {}
        }
    }
    flush(&mut text, &mut tool_calls, out);
}

// ---------------------------------------------------------------------------
// OpenAI -> Anthropic
// ---------------------------------------------------------------------------

pub fn map_stop_reason(finish_reason: Option<&str>, has_tool_use: bool) -> String {
    if has_tool_use {
        return "tool_use".into();
    }
    match finish_reason {
        Some("length") => "max_tokens",
        Some("tool_calls") | Some("function_call") => "tool_use",
        Some("content_filter") => "refusal",
        _ => "end_turn",
    }
    .into()
}

pub fn parse_tool_arguments(args: &str) -> Value {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return json!({});
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(v @ Value::Object(_)) => v,
        Ok(other) => json!({"value": other}),
        Err(e) => {
            warn!(error = %e, "tool call arguments are not valid JSON; wrapping raw string");
            json!({"_raw_arguments": trimmed})
        }
    }
}

pub fn map_usage(u: Option<&OaUsage>) -> Usage {
    let u = match u {
        Some(u) => u,
        None => return Usage::default(),
    };
    let cached = u.prompt_tokens_details.as_ref().and_then(|d| d.cached_tokens).filter(|c| *c > 0);
    // Anthropic reports input_tokens *excluding* cache reads.
    let prompt = u.prompt_tokens.unwrap_or(0);
    Usage {
        input_tokens: prompt.saturating_sub(cached.unwrap_or(0)),
        output_tokens: u.completion_tokens.unwrap_or(0),
        cache_read_input_tokens: cached,
        cache_creation_input_tokens: None,
        server_tool_use: None,
    }
}

pub fn from_chat_response(resp: ChatResponse, model: &str) -> MessagesResponse {
    let mut content: Vec<ResponseBlock> = Vec::new();
    let mut stop_reason = "end_turn".to_string();

    if let Some(choice) = resp.choices.into_iter().next() {
        let msg = choice.message;
        if let Some(r) = msg.reasoning_content.or(msg.reasoning) {
            if !r.is_empty() {
                content.push(ResponseBlock::Thinking { thinking: r, signature: String::new() });
            }
        }
        let text = value_to_text(msg.content);
        if !text.is_empty() {
            content.push(ResponseBlock::Text { text });
        }
        let calls = msg.tool_calls.unwrap_or_default();
        let has_tool_use = !calls.is_empty();
        for tc in calls {
            content.push(ResponseBlock::ToolUse {
                id: tc.id.unwrap_or_else(new_call_id),
                name: tc.function.name,
                input: parse_tool_arguments(&tc.function.arguments),
            });
        }
        stop_reason = map_stop_reason(choice.finish_reason.as_deref(), has_tool_use);
    }

    MessagesResponse {
        id: resp.id.map(|i| format!("msg_{i}")).unwrap_or_else(new_message_id),
        kind: "message",
        role: "assistant",
        model: model.to_string(),
        content,
        stop_reason: Some(stop_reason),
        stop_sequence: None,
        usage: map_usage(resp.usage.as_ref()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config::default()
    }

    #[test]
    fn converts_system_tools_and_tool_results() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-opus-5",
            "max_tokens": 100,
            "system": [{"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}],
            "tools": [
                {"name": "Read", "description": "read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
                {"type": "web_search_20260209", "name": "web_search"}
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "x"},
                    {"type": "text", "text": "reading"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.rs"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "fn main(){}"}]},
                    {"type": "text", "text": "thanks"}
                ]}
            ]
        }))
        .unwrap();
        let chat = build_chat_request(&req, &cfg(), "gpt-x");
        let v = serde_json::to_value(&chat).unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "reading");
        assert_eq!(msgs[2]["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["arguments"], "{\"path\":\"a.rs\"}");
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "toolu_1");
        assert_eq!(msgs[3]["content"], "fn main(){}");
        assert_eq!(msgs[4]["role"], "user");
        assert_eq!(msgs[4]["content"], "thanks");
        assert_eq!(v["tools"].as_array().unwrap().len(), 1);
        assert_eq!(v["tool_choice"], "auto");
        assert_eq!(v["parallel_tool_calls"], false);
        assert_eq!(v["max_tokens"], 100);
        assert_eq!(v["model"], "gpt-x");
    }

    #[test]
    fn converts_response_with_tool_calls() {
        let resp: ChatResponse = serde_json::from_value(json!({
            "id": "chatcmpl-1",
            "choices": [{"message": {"role": "assistant", "content": "ok", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": "{\"path\": \"x\"}"}}
            ]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 4}}
        }))
        .unwrap();
        let out = from_chat_response(resp, "claude-opus-5");
        let v = serde_json::to_value(&out).unwrap();
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][1]["type"], "tool_use");
        assert_eq!(v["content"][1]["input"]["path"], "x");
        assert_eq!(v["usage"]["input_tokens"], 6);
        assert_eq!(v["usage"]["cache_read_input_tokens"], 4);
        assert_eq!(v["usage"]["output_tokens"], 5);
    }

    #[test]
    fn mid_conversation_system_is_folded_into_user_turn() {
        let req: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 1,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": [{"type": "text", "text": "be brief"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
                {"role": "system", "content": "reminder"}
            ]
        }))
        .unwrap();
        let v = serde_json::to_value(build_chat_request(&req, &cfg(), "x")).unwrap();
        let roles: Vec<&str> = v["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool", "user"]);
        assert!(v["messages"][0]["content"].as_str().unwrap().contains("be brief"));
        assert!(v["messages"][3]["content"].as_str().unwrap().contains("reminder"));

        let strict = Config { mid_system_as_system: true, ..Default::default() };
        let v = serde_json::to_value(build_chat_request(&req, &strict, "x")).unwrap();
        let roles: Vec<&str> = v["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, vec!["user", "system", "assistant", "tool", "system"]);
    }

    #[test]
    fn bad_tool_arguments_are_wrapped_not_dropped() {
        assert_eq!(parse_tool_arguments("")["type"], Value::Null);
        assert_eq!(parse_tool_arguments("{oops")["_raw_arguments"], "{oops");
    }
}
