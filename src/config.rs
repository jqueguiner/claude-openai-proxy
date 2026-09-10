//! Environment-driven configuration.

use std::collections::HashMap;
use std::env;

#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Address to bind, e.g. `127.0.0.1:8082`.
    pub listen: String,
    /// Upstream OpenAI-compatible base URL including `/v1`, e.g. `http://localhost:8000/v1`.
    pub base_url: String,
    /// Upstream API key (sent as `Authorization: Bearer`).
    pub api_key: Option<String>,
    /// Exact Anthropic model name -> upstream model name.
    pub model_map: HashMap<String, String>,
    /// Family fallbacks (matched on substring of the requested model name).
    pub opus_model: Option<String>,
    pub sonnet_model: Option<String>,
    pub haiku_model: Option<String>,
    /// Used when nothing else matches. If unset the requested name is passed through.
    pub default_model: Option<String>,
    /// Clamp `max_tokens` (many local servers reject large values).
    pub max_tokens_cap: Option<u32>,
    /// `None` (don't send), `Some("auto")` (derive from effort/thinking) or a fixed value.
    pub reasoning_effort: Option<String>,
    /// When no upstream key is configured, forward the client's key/token as the bearer token.
    pub forward_client_key: bool,
    /// Log full upstream request/response bodies at DEBUG level.
    pub log_bodies: bool,
    /// Forward mid-conversation `role: system` messages as OpenAI system messages.
    /// Default false: fold them into the adjacent user turn (works with strict chat templates).
    pub mid_system_as_system: bool,
    /// Web search backend used to emulate the Anthropic `web_search` server tool:
    /// `searxng`, `brave` or `tavily`. Unset = web search tool rejected with 400 (like a disabled org).
    pub search_provider: Option<String>,
    /// Base URL for `searxng` (e.g. `http://localhost:8080`); optional override for the others.
    pub search_url: Option<String>,
    pub search_api_key: Option<String>,
    pub search_max_results: usize,
    /// Upper bound on upstream rounds inside one web-search turn (Anthropic: 10 -> `pause_turn`).
    pub search_max_rounds: u32,
}

fn env_opt(key: &str) -> Option<String> {
    env::var(key).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

fn env_bool(key: &str, default: bool) -> bool {
    match env_opt(key).as_deref().map(|v| v.to_ascii_lowercase()) {
        Some(v) => matches!(v.as_str(), "1" | "true" | "yes" | "on"),
        None => default,
    }
}

impl Config {
    pub fn from_env() -> Self {
        let model_map = env_opt("MODEL_MAP")
            .map(|raw| {
                raw.split(',')
                    .filter_map(|pair| {
                        let (k, v) = pair.split_once('=')?;
                        let (k, v) = (k.trim(), v.trim());
                        if k.is_empty() || v.is_empty() {
                            None
                        } else {
                            Some((k.to_string(), v.to_string()))
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        Self {
            listen: env_opt("PROXY_LISTEN").unwrap_or_else(|| "127.0.0.1:8082".into()),
            base_url: env_opt("OPENAI_BASE_URL")
                .unwrap_or_else(|| "http://127.0.0.1:8000/v1".into())
                .trim_end_matches('/')
                .to_string(),
            api_key: env_opt("OPENAI_API_KEY"),
            model_map,
            opus_model: env_opt("OPUS_MODEL"),
            sonnet_model: env_opt("SONNET_MODEL"),
            haiku_model: env_opt("HAIKU_MODEL"),
            default_model: env_opt("DEFAULT_MODEL"),
            max_tokens_cap: env_opt("MAX_TOKENS_CAP").and_then(|v| v.parse().ok()),
            reasoning_effort: env_opt("REASONING_EFFORT"),
            forward_client_key: env_bool("FORWARD_CLIENT_KEY", true),
            log_bodies: env_bool("LOG_BODIES", false),
            mid_system_as_system: env_bool("MID_SYSTEM_AS_SYSTEM", false),
            search_provider: env_opt("SEARCH_PROVIDER").map(|p| p.to_ascii_lowercase()),
            search_url: env_opt("SEARCH_URL").map(|u| u.trim_end_matches('/').to_string()),
            search_api_key: env_opt("SEARCH_API_KEY"),
            search_max_results: env_opt("SEARCH_MAX_RESULTS").and_then(|v| v.parse().ok()).unwrap_or(10),
            search_max_rounds: env_opt("SEARCH_MAX_ROUNDS").and_then(|v| v.parse().ok()).unwrap_or(10),
        }
    }

    /// Map a model name sent by Claude Code to the upstream model name.
    pub fn resolve_model(&self, requested: &str) -> String {
        if let Some(m) = self.model_map.get(requested) {
            return m.clone();
        }
        let lower = requested.to_ascii_lowercase();
        let family = if lower.contains("opus") || lower.contains("fable") || lower.contains("mythos") {
            self.opus_model.as_ref()
        } else if lower.contains("sonnet") {
            self.sonnet_model.as_ref()
        } else if lower.contains("haiku") {
            self.haiku_model.as_ref()
        } else {
            None
        };
        family
            .or(self.default_model.as_ref())
            .cloned()
            .unwrap_or_else(|| requested.to_string())
    }

    /// Model names Claude Code may request that this proxy knows how to map.
    pub fn known_models(&self) -> Vec<String> {
        let mut v: Vec<String> = self.model_map.keys().cloned().collect();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_by_exact_then_family_then_default() {
        let cfg = Config {
            model_map: HashMap::from([("claude-opus-5".to_string(), "big".to_string())]),
            sonnet_model: Some("mid".into()),
            default_model: Some("fallback".into()),
            ..Default::default()
        };
        assert_eq!(cfg.resolve_model("claude-opus-5"), "big");
        assert_eq!(cfg.resolve_model("claude-sonnet-5"), "mid");
        assert_eq!(cfg.resolve_model("claude-haiku-4-5"), "fallback");
        let passthrough = Config::default();
        assert_eq!(passthrough.resolve_model("gpt-x"), "gpt-x");
    }
}
