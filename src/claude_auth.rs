//! Credentials for talking to the Anthropic API on behalf of an OpenAI-style
//! client: an API key, a Claude Code OAuth token from the environment, or the
//! token Claude Code itself stores (macOS keychain / `~/.claude/.credentials.json`).
//!
//! OAuth tokens are never refreshed here: refreshing rotates the refresh token
//! and would invalidate Claude Code's own login. Run `claude` to refresh.

use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tracing::{debug, warn};

use crate::config::Config;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Credential {
    ApiKey(String),
    OAuth(String),
}

impl Credential {
    pub fn is_oauth(&self) -> bool {
        matches!(self, Credential::OAuth(_))
    }

    /// Classify a raw Anthropic secret by its prefix.
    pub fn from_secret(secret: &str) -> Self {
        let s = secret.trim();
        if s.starts_with("sk-ant-oat") || s.starts_with("sk-ant-ort") {
            Credential::OAuth(s.to_string())
        } else {
            Credential::ApiKey(s.to_string())
        }
    }
}

static CACHE: Mutex<Option<(Instant, Credential)>> = Mutex::new(None);
const CACHE_TTL: Duration = Duration::from_secs(30);

/// Resolve the credential to use for one request.
///
/// Order: `ANTHROPIC_API_KEY` → `CLAUDE_CODE_OAUTH_TOKEN` → Claude Code's own
/// stored login → the client's bearer token when it is an Anthropic secret.
pub async fn resolve(cfg: &Config, client_bearer: Option<&str>) -> Result<Credential, String> {
    if let Some(k) = &cfg.anthropic_api_key {
        return Ok(Credential::from_secret(k));
    }
    if let Some(t) = &cfg.claude_code_oauth_token {
        return Ok(Credential::OAuth(t.clone()));
    }
    if let Some(c) = cached() {
        return Ok(c);
    }
    match load_claude_code_login(cfg).await {
        Ok(c) => {
            *CACHE.lock().unwrap() = Some((Instant::now(), c.clone()));
            Ok(c)
        }
        Err(e) => {
            if let Some(b) = client_bearer.filter(|b| b.starts_with("sk-ant-")) {
                return Ok(Credential::from_secret(b));
            }
            Err(e)
        }
    }
}

pub fn invalidate_cache() {
    *CACHE.lock().unwrap() = None;
}

fn cached() -> Option<Credential> {
    let guard = CACHE.lock().unwrap();
    match &*guard {
        Some((at, c)) if at.elapsed() < CACHE_TTL => Some(c.clone()),
        _ => None,
    }
}

async fn load_claude_code_login(cfg: &Config) -> Result<Credential, String> {
    let mut errors: Vec<String> = Vec::new();

    if cfg.claude_keychain {
        match read_keychain().await {
            Ok(json) => match parse_credentials(&json) {
                Ok(c) => {
                    debug!("using Claude Code OAuth token from macOS keychain");
                    return Ok(c);
                }
                Err(e) => errors.push(format!("keychain: {e}")),
            },
            Err(e) => errors.push(format!("keychain: {e}")),
        }
    }

    let path = cfg.claude_credentials_file.clone();
    match tokio::fs::read_to_string(&path).await {
        Ok(json) => match parse_credentials(&json) {
            Ok(c) => {
                debug!(path = %path, "using Claude Code OAuth token from credentials file");
                return Ok(c);
            }
            Err(e) => errors.push(format!("{path}: {e}")),
        },
        Err(e) => errors.push(format!("{path}: {e}")),
    }

    warn!(errors = ?errors, "no Anthropic credential available");
    Err(format!(
        "No Anthropic credential: set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN, or log in with `claude` ({})",
        errors.join("; ")
    ))
}

#[cfg(target_os = "macos")]
async fn read_keychain() -> Result<String, String> {
    let fut = tokio::process::Command::new("security")
        .args(["find-generic-password", "-s", "Claude Code-credentials", "-w"])
        .output();
    let out = tokio::time::timeout(Duration::from_secs(5), fut)
        .await
        .map_err(|_| "security(1) timed out".to_string())?
        .map_err(|e| format!("cannot run security(1): {e}"))?;
    if !out.status.success() {
        return Err(format!("security(1) exited with {}", out.status));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[cfg(not(target_os = "macos"))]
async fn read_keychain() -> Result<String, String> {
    Err("no keychain on this platform".into())
}

/// Parse the JSON Claude Code stores (`{"claudeAiOauth": {"accessToken", "expiresAt", ...}}`).
pub fn parse_credentials(json: &str) -> Result<Credential, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("invalid JSON: {e}"))?;
    let oauth = v.get("claudeAiOauth").ok_or("no claudeAiOauth entry")?;
    let token = oauth
        .get("accessToken")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
        .ok_or("no accessToken")?;
    if let Some(exp) = oauth.get("expiresAt").and_then(Value::as_u64) {
        let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
        if exp <= now_ms {
            return Err("Claude Code OAuth token expired — run `claude` once to refresh it".into());
        }
    }
    Ok(Credential::OAuth(token.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_checks_expiry() {
        let ok = r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-x","expiresAt":9999999999999}}"#;
        assert_eq!(parse_credentials(ok).unwrap(), Credential::OAuth("sk-ant-oat01-x".into()));
        let expired = r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-x","expiresAt":1}}"#;
        assert!(parse_credentials(expired).unwrap_err().contains("expired"));
        assert!(parse_credentials("{}").is_err());
        assert!(Credential::from_secret("sk-ant-api03-abc").is_oauth() == false);
        assert!(Credential::from_secret("sk-ant-oat01-abc").is_oauth());
    }
}
