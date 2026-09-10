//! Anthropic Messages API wire types (the side Claude Code talks to).
//!
//! Deserialization is deliberately lenient: unknown fields (cache_control,
//! beta parameters, ...) are ignored so new Claude Code versions keep working.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub tools: Vec<ToolDef>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub thinking: Option<Value>,
    #[serde(default)]
    pub output_config: Option<OutputConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct OutputConfig {
    #[serde(default)]
    pub effort: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<ToolResultContent>,
        #[serde(default)]
        is_error: bool,
    },
    Thinking {},
    RedactedThinking {},
    Document {},
    /// Replayed server tool call (web_search) from a previous proxy turn.
    ServerToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    /// Replayed web search results from a previous proxy turn.
    WebSearchToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Value,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub media_type: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
}

impl ImageSource {
    /// Turn an Anthropic image source into something usable as an OpenAI `image_url`.
    pub fn to_url(&self) -> Option<String> {
        match self.kind.as_str() {
            "base64" => Some(format!(
                "data:{};base64,{}",
                self.media_type.as_deref().unwrap_or("image/png"),
                self.data.as_deref().unwrap_or("")
            )),
            "url" => self.url.clone(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<Value>,
    /// Present for Anthropic server/built-in tools (`web_search_20260209`, `bash_20250124`, ...).
    /// Absent or `custom` for ordinary function tools.
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    // web_search server tool options
    #[serde(default)]
    pub max_uses: Option<u32>,
    #[serde(default)]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(default)]
    pub blocked_domains: Option<Vec<String>>,
    #[serde(default)]
    pub user_location: Option<Value>,
}

impl ToolDef {
    pub fn is_web_search(&self) -> bool {
        self.kind.as_deref().map(|k| k.starts_with("web_search")).unwrap_or(false)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoice {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub disable_parallel_tool_use: bool,
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ResponseBlock>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<ServerToolUsage>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ServerToolUsage {
    pub web_search_requests: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseBlock {
    Text { text: String },
    Thinking { thinking: String, signature: String },
    ToolUse { id: String, name: String, input: Value },
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}

impl ErrorResponse {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: "error",
            error: ErrorBody {
                kind: kind.into(),
                message: message.into(),
            },
        }
    }
}
