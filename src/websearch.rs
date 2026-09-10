//! Emulation of the Anthropic `web_search` server tool.
//!
//! Anthropic runs a server-side loop: the model asks for a search, the API runs
//! it, feeds the results back and lets the model continue, until the turn ends
//! (`end_turn`), a client tool is requested (`tool_use`) or the iteration cap
//! is reached (`pause_turn`). This module does the same on top of an OpenAI
//! function tool, with a pluggable search backend (SearXNG, Brave, Tavily).

use std::sync::Arc;

use base64::Engine;
use bytes::Bytes;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::anthropic::{ToolChoice, ToolDef};
use crate::config::Config;
use crate::convert::{map_stop_reason, new_message_id};
use crate::openai::{ChatMessage, ChatRequest, FunctionCall, ToolCall};
use crate::stream::Translator;
use crate::upstream::{self, UpstreamError};

pub const TOOL_NAME: &str = "web_search";

#[derive(Debug, Clone, Default)]
pub struct WebSearchSpec {
    pub max_uses: Option<u32>,
    pub allowed_domains: Vec<String>,
    pub blocked_domains: Vec<String>,
    pub country: Option<String>,
}

/// Find the Anthropic web search tool in a request's tool list.
pub fn spec_from_tools(tools: &[ToolDef]) -> Option<WebSearchSpec> {
    let t = tools.iter().find(|t| t.is_web_search())?;
    Some(WebSearchSpec {
        max_uses: t.max_uses,
        allowed_domains: t.allowed_domains.clone().unwrap_or_default(),
        blocked_domains: t.blocked_domains.clone().unwrap_or_default(),
        country: t
            .user_location
            .as_ref()
            .and_then(|l| l.get("country"))
            .and_then(Value::as_str)
            .map(|c| c.to_ascii_uppercase()),
    })
}

/// The OpenAI function-tool definition shown to the upstream model.
pub fn function_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Search the web for current information. Returns a list of results with title, url, snippet and page age. Use it when the answer depends on recent or changing information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    })
}

/// Map the Anthropic `tool_choice` of a web-search request to the first upstream round.
pub fn first_tool_choice(tc: Option<&ToolChoice>) -> Option<Value> {
    match tc {
        None => None,
        Some(tc) => match tc.kind.as_str() {
            "any" => Some(json!("required")),
            "none" => Some(json!("none")),
            "tool" => tc.name.as_ref().map(|n| json!({"type": "function", "function": {"name": n}})),
            _ => Some(json!("auto")),
        },
    }
}

// ---------------------------------------------------------------------------
// Search backends
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,
}

#[derive(Debug, Clone)]
pub enum SearchError {
    Unavailable(String),
    TooManyRequests,
    InvalidInput,
    QueryTooLong,
}

impl SearchError {
    pub fn code(&self) -> &'static str {
        match self {
            SearchError::Unavailable(_) => "unavailable",
            SearchError::TooManyRequests => "too_many_requests",
            SearchError::InvalidInput => "invalid_tool_input",
            SearchError::QueryTooLong => "query_too_long",
        }
    }
}

const MAX_QUERY_CHARS: usize = 2048;

pub async fn search(
    client: &reqwest::Client,
    cfg: &Config,
    query: &str,
    spec: &WebSearchSpec,
) -> Result<Vec<SearchResult>, SearchError> {
    let query = query.trim();
    if query.is_empty() {
        return Err(SearchError::InvalidInput);
    }
    if query.chars().count() > MAX_QUERY_CHARS {
        return Err(SearchError::QueryTooLong);
    }
    let provider = cfg.search_provider.as_deref().unwrap_or("");
    let raw = match provider {
        "searxng" => searxng(client, cfg, query, spec).await?,
        "brave" => brave(client, cfg, query, spec).await?,
        "tavily" => tavily(client, cfg, query, spec).await?,
        other => return Err(SearchError::Unavailable(format!("unknown SEARCH_PROVIDER {other:?}"))),
    };
    let filtered: Vec<SearchResult> = raw
        .into_iter()
        .filter(|r| !r.url.is_empty())
        .filter(|r| spec.allowed_domains.is_empty() || spec.allowed_domains.iter().any(|d| domain_matches(&r.url, d)))
        .filter(|r| !spec.blocked_domains.iter().any(|d| domain_matches(&r.url, d)))
        .take(cfg.search_max_results)
        .collect();
    Ok(filtered)
}

/// Bias the query towards the allowed domains (post-filtering still applies).
fn query_with_site_hint(query: &str, spec: &WebSearchSpec) -> String {
    if spec.allowed_domains.is_empty() || spec.allowed_domains.len() > 5 {
        return query.to_string();
    }
    let sites: Vec<String> = spec.allowed_domains.iter().map(|d| format!("site:{d}")).collect();
    format!("{query} ({})", sites.join(" OR "))
}

/// `pattern` is a bare domain with an optional path (`example.com`, `example.com/blog`).
pub fn domain_matches(url: &str, pattern: &str) -> bool {
    let Ok(parsed) = url::Url::parse(url) else { return false };
    let Some(host) = parsed.host_str() else { return false };
    let pattern = pattern.trim().trim_start_matches("https://").trim_start_matches("http://");
    let (pat_host, pat_path) = match pattern.split_once('/') {
        Some((h, p)) => (h, Some(p.trim_end_matches('/'))),
        None => (pattern, None),
    };
    let pat_host = pat_host.trim_start_matches("*.").to_ascii_lowercase();
    let host = host.to_ascii_lowercase();
    let host_ok = host == pat_host || host.ends_with(&format!(".{pat_host}"));
    if !host_ok {
        return false;
    }
    match pat_path {
        None => true,
        Some(p) => parsed.path().trim_end_matches('/').starts_with(&format!("/{p}")),
    }
}

fn http_err(e: reqwest::Error) -> SearchError {
    SearchError::Unavailable(format!("search backend request failed: {e}"))
}

fn status_err(status: reqwest::StatusCode, body: &str) -> SearchError {
    if status.as_u16() == 429 {
        SearchError::TooManyRequests
    } else {
        SearchError::Unavailable(format!("search backend returned {}: {}", status.as_u16(), body.chars().take(300).collect::<String>()))
    }
}

fn str_field(v: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| v.get(k).and_then(Value::as_str)).map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

async fn searxng(client: &reqwest::Client, cfg: &Config, query: &str, spec: &WebSearchSpec) -> Result<Vec<SearchResult>, SearchError> {
    let base = cfg
        .search_url
        .clone()
        .ok_or_else(|| SearchError::Unavailable("SEARCH_URL is required for the searxng provider".into()))?;
    let resp = client
        .get(format!("{base}/search"))
        .query(&[("q", query_with_site_hint(query, spec).as_str()), ("format", "json"), ("categories", "general")])
        .send()
        .await
        .map_err(http_err)?;
    let status = resp.status();
    let body = resp.text().await.map_err(http_err)?;
    if !status.is_success() {
        return Err(status_err(status, &body));
    }
    let v: Value = serde_json::from_str(&body).map_err(|e| SearchError::Unavailable(format!("searxng JSON: {e}")))?;
    Ok(v["results"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|r| {
                    Some(SearchResult {
                        title: str_field(r, &["title"]).unwrap_or_default(),
                        url: str_field(r, &["url"])?,
                        snippet: str_field(r, &["content"]).unwrap_or_default(),
                        page_age: str_field(r, &["publishedDate"]),
                    })
                })
                .collect()
        })
        .unwrap_or_default())
}

async fn brave(client: &reqwest::Client, cfg: &Config, query: &str, spec: &WebSearchSpec) -> Result<Vec<SearchResult>, SearchError> {
    let key = cfg
        .search_api_key
        .clone()
        .ok_or_else(|| SearchError::Unavailable("SEARCH_API_KEY is required for the brave provider".into()))?;
    let base = cfg.search_url.clone().unwrap_or_else(|| "https://api.search.brave.com/res/v1".into());
    let count = cfg.search_max_results.clamp(1, 20).to_string();
    let mut params = vec![("q", query_with_site_hint(query, spec)), ("count", count)];
    if let Some(c) = &spec.country {
        params.push(("country", c.clone()));
    }
    let resp = client
        .get(format!("{base}/web/search"))
        .query(&params)
        .header("Accept", "application/json")
        .header("X-Subscription-Token", key)
        .send()
        .await
        .map_err(http_err)?;
    let status = resp.status();
    let body = resp.text().await.map_err(http_err)?;
    if !status.is_success() {
        return Err(status_err(status, &body));
    }
    let v: Value = serde_json::from_str(&body).map_err(|e| SearchError::Unavailable(format!("brave JSON: {e}")))?;
    Ok(v["web"]["results"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|r| {
                    Some(SearchResult {
                        title: str_field(r, &["title"]).unwrap_or_default(),
                        url: str_field(r, &["url"])?,
                        snippet: str_field(r, &["description"]).unwrap_or_default(),
                        page_age: str_field(r, &["page_age", "age"]),
                    })
                })
                .collect()
        })
        .unwrap_or_default())
}

async fn tavily(client: &reqwest::Client, cfg: &Config, query: &str, spec: &WebSearchSpec) -> Result<Vec<SearchResult>, SearchError> {
    let key = cfg
        .search_api_key
        .clone()
        .ok_or_else(|| SearchError::Unavailable("SEARCH_API_KEY is required for the tavily provider".into()))?;
    let base = cfg.search_url.clone().unwrap_or_else(|| "https://api.tavily.com".into());
    let mut body = json!({"query": query, "max_results": cfg.search_max_results.clamp(1, 20)});
    if !spec.allowed_domains.is_empty() {
        body["include_domains"] = json!(spec.allowed_domains);
    }
    if !spec.blocked_domains.is_empty() {
        body["exclude_domains"] = json!(spec.blocked_domains);
    }
    let resp = client
        .post(format!("{base}/search"))
        .bearer_auth(key)
        .json(&body)
        .send()
        .await
        .map_err(http_err)?;
    let status = resp.status();
    let text = resp.text().await.map_err(http_err)?;
    if !status.is_success() {
        return Err(status_err(status, &text));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| SearchError::Unavailable(format!("tavily JSON: {e}")))?;
    Ok(v["results"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|r| {
                    Some(SearchResult {
                        title: str_field(r, &["title"]).unwrap_or_default(),
                        url: str_field(r, &["url"])?,
                        snippet: str_field(r, &["content"]).unwrap_or_default(),
                        page_age: str_field(r, &["published_date"]),
                    })
                })
                .collect()
        })
        .unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Anthropic block shapes
// ---------------------------------------------------------------------------

fn b64() -> base64::engine::GeneralPurpose {
    base64::engine::general_purpose::STANDARD
}

/// `web_search_tool_result` block. The snippet travels in `encrypted_content`
/// (Anthropic's opaque field) so it can be restored when the client replays the turn.
pub fn result_block(tool_use_id: &str, outcome: &Result<Vec<SearchResult>, SearchError>) -> Value {
    let content = match outcome {
        Ok(results) => Value::Array(
            results
                .iter()
                .map(|r| {
                    json!({
                        "type": "web_search_result",
                        "url": r.url,
                        "title": r.title,
                        "encrypted_content": b64().encode(r.snippet.as_bytes()),
                        "page_age": r.page_age,
                    })
                })
                .collect(),
        ),
        Err(e) => json!({"type": "web_search_tool_result_error", "error_code": e.code()}),
    };
    json!({"type": "web_search_tool_result", "tool_use_id": tool_use_id, "content": content})
}

/// Text handed to the upstream model as the function result.
pub fn results_text(outcome: &Result<Vec<SearchResult>, SearchError>) -> String {
    match outcome {
        Ok(results) if results.is_empty() => "No results found.".to_string(),
        Ok(results) => serde_json::to_string_pretty(results).unwrap_or_default(),
        Err(SearchError::Unavailable(msg)) => format!("Web search error: unavailable ({msg})"),
        Err(e) => format!("Web search error: {}", e.code()),
    }
}

pub fn max_uses_text() -> String {
    "Web search error: max_uses_exceeded. The search budget for this turn is spent; answer with the information already gathered.".into()
}

/// Rebuild the function-result text from a replayed `web_search_tool_result` block.
pub fn replay_result_text(content: &Value) -> String {
    match content {
        Value::Array(items) => {
            let results: Vec<SearchResult> = items
                .iter()
                .map(|r| SearchResult {
                    title: str_field(r, &["title"]).unwrap_or_default(),
                    url: str_field(r, &["url"]).unwrap_or_default(),
                    snippet: str_field(r, &["encrypted_content"])
                        .and_then(|e| b64().decode(e).ok())
                        .map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_default(),
                    page_age: str_field(r, &["page_age"]),
                })
                .collect();
            results_text(&Ok(results))
        }
        other => {
            let code = other.get("error_code").and_then(Value::as_str).unwrap_or("unavailable");
            format!("Web search error: {code}")
        }
    }
}

// ---------------------------------------------------------------------------
// Server-side loop
// ---------------------------------------------------------------------------

pub struct TurnInput {
    pub client: reqwest::Client,
    pub cfg: Arc<Config>,
    pub auth: Option<String>,
    /// Model name as requested by the client (echoed back).
    pub model: String,
    /// Converted upstream request (messages + client tools). The `web_search`
    /// function tool is appended here.
    pub chat: ChatRequest,
    pub spec: WebSearchSpec,
    pub first_tool_choice: Option<Value>,
}

pub struct TurnOutcome {
    pub translator: Translator,
    pub stop_reason: Option<String>,
    pub error: Option<UpstreamError>,
}

/// Run the whole turn, streaming Anthropic events into `tx`.
pub async fn run_turn(input: TurnInput, tx: mpsc::Sender<Bytes>) -> TurnOutcome {
    let TurnInput { client, cfg, auth, model, mut chat, spec, first_tool_choice } = input;
    let mut tr = Translator::new(new_message_id(), model).with_server_tools([TOOL_NAME.to_string()]);
    let mut out: Vec<u8> = Vec::new();
    tr.start(&mut out);
    upstream::flush(&tx, &mut out).await;

    chat.stream = true;
    chat.stream_options = Some(json!({"include_usage": true}));
    let mut tools = chat.tools.take().unwrap_or_default();
    tools.push(function_tool());
    chat.tools = Some(tools);

    let mut uses: u32 = 0;
    let mut rounds: u32 = 0;
    let stop_reason: String;

    loop {
        rounds += 1;
        chat.tool_choice = if rounds == 1 { first_tool_choice.clone() } else { Some(json!("auto")) };

        let resp = match upstream::send_chat(&client, &cfg, auth.as_deref(), &chat).await {
            Ok(r) => r,
            Err(e) => {
                tr.error(&e.message, &mut out);
                upstream::flush(&tx, &mut out).await;
                return TurnOutcome { translator: tr, stop_reason: None, error: Some(e) };
            }
        };
        if let Err(msg) = upstream::pump(resp, &mut tr, &mut out, &tx, cfg.log_bodies).await {
            return TurnOutcome {
                translator: tr,
                stop_reason: None,
                error: Some(UpstreamError { status: 502, kind: "api_error", message: msg }),
            };
        }
        let summary = tr.end_round(&mut out);
        upstream::flush(&tx, &mut out).await;

        let (server_calls, client_calls): (Vec<_>, Vec<_>) = summary.calls.iter().partition(|c| c.is_server);
        if server_calls.is_empty() {
            stop_reason = if client_calls.is_empty() {
                map_stop_reason(summary.finish_reason.as_deref(), false)
            } else {
                "tool_use".to_string()
            };
            break;
        }

        // Replay this round into the upstream conversation.
        chat.messages.push(ChatMessage {
            role: "assistant",
            content: if summary.text.is_empty() { None } else { Some(Value::String(summary.text.clone())) },
            tool_calls: Some(
                summary
                    .calls
                    .iter()
                    .map(|c| ToolCall {
                        id: Some(c.oa_id.clone()),
                        kind: "function".into(),
                        function: FunctionCall { name: c.name.clone(), arguments: c.arguments.clone() },
                    })
                    .collect(),
            ),
            tool_call_id: None,
        });

        for call in &server_calls {
            let query = serde_json::from_str::<Value>(&call.arguments)
                .ok()
                .and_then(|v| v.get("query").and_then(Value::as_str).map(str::to_string))
                .unwrap_or_default();
            uses += 1;
            let over_budget = spec.max_uses.map(|m| uses > m).unwrap_or(false);
            let (block, text) = if over_budget {
                warn!(uses, "web_search max_uses exceeded");
                let outcome: Result<Vec<SearchResult>, SearchError> = Err(SearchError::InvalidInput);
                let mut block = result_block(&call.block_id, &outcome);
                block["content"] = json!({"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"});
                (block, max_uses_text())
            } else {
                info!(query = %query, round = rounds, "web_search");
                let outcome = search(&client, &cfg, &query, &spec).await;
                match &outcome {
                    Ok(r) => {
                        tr.note_web_search();
                        info!(results = r.len(), "web_search done");
                    }
                    Err(e) => warn!(error = ?e, "web_search failed"),
                }
                (result_block(&call.block_id, &outcome), results_text(&outcome))
            };
            tr.push_block(block, &mut out);
            chat.messages.push(ChatMessage {
                role: "tool",
                content: Some(Value::String(text)),
                tool_calls: None,
                tool_call_id: Some(call.oa_id.clone()),
            });
        }
        upstream::flush(&tx, &mut out).await;

        if !client_calls.is_empty() {
            // Client tools need the caller; the search results are already in place.
            stop_reason = "tool_use".to_string();
            break;
        }
        if rounds >= cfg.search_max_rounds {
            stop_reason = "pause_turn".to_string();
            break;
        }
    }

    tr.finish_with(&mut out, Some(&stop_reason));
    upstream::flush(&tx, &mut out).await;
    TurnOutcome { translator: tr, stop_reason: Some(stop_reason), error: None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_filter_semantics() {
        assert!(domain_matches("https://docs.example.com/a", "example.com"));
        assert!(domain_matches("https://example.com/blog/x", "example.com/blog"));
        assert!(!domain_matches("https://example.com/news", "example.com/blog"));
        assert!(!domain_matches("https://notexample.com", "example.com"));
        assert!(domain_matches("http://EXAMPLE.com", "example.com"));
    }

    #[test]
    fn result_block_roundtrip() {
        let ok = Ok(vec![SearchResult { title: "T".into(), url: "https://a.b".into(), snippet: "snip".into(), page_age: None }]);
        let block = result_block("srvtoolu_1", &ok);
        assert_eq!(block["type"], "web_search_tool_result");
        assert_eq!(block["content"][0]["type"], "web_search_result");
        assert!(replay_result_text(&block["content"]).contains("snip"));
        let err = result_block("srvtoolu_1", &Err(SearchError::TooManyRequests));
        assert_eq!(err["content"]["error_code"], "too_many_requests");
        assert_eq!(replay_result_text(&err["content"]), "Web search error: too_many_requests");
    }
}
