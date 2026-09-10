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

    // --- OpenAI-compatible side (`/v1/chat/completions` -> Anthropic Messages API) ---
    pub anthropic_base_url: String,
    pub anthropic_api_key: Option<String>,
    pub claude_code_oauth_token: Option<String>,
    pub claude_credentials_file: String,
    pub claude_keychain: bool,
    /// OpenAI-style model name -> Anthropic model id.
    pub chat_model_map: HashMap<String, String>,
    /// Used for model names that are not Claude models after mapping.
    pub chat_default_model: String,
    /// `max_tokens` when the client sends none.
    pub chat_max_tokens: u32,
    /// Claude Code version advertised in the OAuth user-agent (auto-detected when unset).
    pub claude_code_version: Option<String>,
}

fn parse_map(raw: Option<String>) -> HashMap<String, String> {
    raw.map(|raw| {
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
    .unwrap_or_default()
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
        let model_map = parse_map(env_opt("MODEL_MAP"));
        let home = env::var("HOME").unwrap_or_else(|_| ".".into());

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
            anthropic_base_url: env_opt("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|| "https://api.anthropic.com".into())
                .trim_end_matches('/')
                .to_string(),
            anthropic_api_key: env_opt("ANTHROPIC_API_KEY"),
            claude_code_oauth_token: env_opt("CLAUDE_CODE_OAUTH_TOKEN"),
            claude_credentials_file: env_opt("CLAUDE_CREDENTIALS_FILE")
                .unwrap_or_else(|| format!("{home}/.claude/.credentials.json")),
            claude_keychain: env_bool("CLAUDE_KEYCHAIN", cfg!(target_os = "macos")),
            chat_model_map: parse_map(env_opt("CHAT_MODEL_MAP")),
            chat_default_model: env_opt("CHAT_DEFAULT_MODEL").unwrap_or_else(|| "claude-opus-5".into()),
            chat_max_tokens: env_opt("CHAT_MAX_TOKENS").and_then(|v| v.parse().ok()).unwrap_or(16384),
            claude_code_version: env_opt("CLAUDE_CODE_VERSION"),
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

    /// Map an OpenAI-style model name to an Anthropic model id.
    pub fn resolve_chat_model(&self, requested: &str) -> String {
        let name = requested.trim();
        if let Some(m) = self.chat_model_map.get(name) {
            return m.clone();
        }
        let stripped = name.strip_prefix("anthropic/").unwrap_or(name);
        if let Some(m) = self.chat_model_map.get(stripped) {
            return m.clone();
        }
        if stripped.starts_with("claude-") {
            // `claude-fable-5.1` -> `claude-fable-5-1`
            return stripped.replace('.', "-");
        }
        self.chat_default_model.clone()
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

    #[test]
    fn resolves_chat_models() {
        let cfg = Config {
            chat_model_map: HashMap::from([("gpt-4o".to_string(), "claude-sonnet-5".to_string())]),
            chat_default_model: "claude-opus-5".into(),
            ..Default::default()
        };
        assert_eq!(cfg.resolve_chat_model("gpt-4o"), "claude-sonnet-5");
        assert_eq!(cfg.resolve_chat_model("anthropic/claude-fable-5.1"), "claude-fable-5-1");
        assert_eq!(cfg.resolve_chat_model("claude-haiku-4-5"), "claude-haiku-4-5");
        assert_eq!(cfg.resolve_chat_model("gpt-5"), "claude-opus-5");
    }
}
