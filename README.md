# claude-openai-proxy

A small Rust proxy that exposes the **Anthropic Messages API** (the protocol Claude Code
speaks) on top of **any OpenAI-compatible chat completions server** — vLLM, llama.cpp,
Ollama, LM Studio, SGLang, OpenRouter, LiteLLM, Groq, Mistral, OpenAI itself, ...

```
Claude Code ──(Anthropic /v1/messages)──▶ claude-openai-proxy ──(OpenAI /v1/chat/completions)──▶ your server
```

## What is translated

| Anthropic (Claude Code side)                 | OpenAI (upstream side)                                   |
| -------------------------------------------- | -------------------------------------------------------- |
| `system` (string or text blocks)             | first `system` message                                   |
| mid-conversation `role: system` messages     | folded into the adjacent user turn (or forwarded, see `MID_SYSTEM_AS_SYSTEM`) |
| `text` / `image` blocks                      | `content` string or `text` / `image_url` parts           |
| assistant `tool_use` blocks                  | `tool_calls` (function calling)                          |
| user `tool_result` blocks                    | `role: tool` messages with `tool_call_id`                |
| `thinking` / `redacted_thinking` blocks      | dropped on the way up (model-bound), `reasoning_content` mapped back to `thinking` |
| `tools` (custom)                             | `tools: [{type: function, ...}]` — other Anthropic server tools (bash, text editor, code execution, ...) are dropped with a warning |
| `web_search_*` server tool                   | emulated by the proxy (see below): `server_tool_use` + `web_search_tool_result` blocks, `usage.server_tool_use`, `pause_turn` |
| `tool_choice` auto / any / tool / none       | `auto` / `required` / `{type: function}` / `none`        |
| `disable_parallel_tool_use`                  | `parallel_tool_calls: false`                             |
| `max_tokens`, `temperature`, `top_p`, `stop_sequences` | `max_tokens`, `temperature`, `top_p`, `stop`   |
| `stream: true`                               | SSE stream, translated event-by-event                    |
| finish_reason `stop` / `length` / `tool_calls` | `stop_reason` `end_turn` / `max_tokens` / `tool_use`   |
| `usage.prompt_tokens` (+ `cached_tokens`)    | `input_tokens` / `cache_read_input_tokens` / `output_tokens` |
| upstream HTTP errors                         | Anthropic `{"type":"error","error":{...}}` with matching status |

Streaming emits the full Anthropic event sequence (`message_start`, `content_block_start`,
`content_block_delta` with `text_delta` / `input_json_delta` / `thinking_delta`,
`content_block_stop`, `message_delta`, `message_stop`) plus `ping` keep-alives every 15 s of
upstream silence, so Claude Code's tool-call streaming and usage accounting work unchanged.

### Web search (Anthropic server tool emulation)

Claude Code's `WebSearch` tool does not search by itself: it sends a side request to
`/v1/messages` with `tools: [{type: "web_search_20250305", name: "web_search", max_uses: 8, ...}]`
and `tool_choice: {type: "tool", name: "web_search"}`, then reads `server_tool_use`,
`web_search_tool_result` and `text` blocks from the answer. The proxy reproduces the
Anthropic server-side loop for such requests:

1. the upstream model is offered a `web_search(query)` function tool;
2. every call is executed by the proxy against the configured backend, emitted as
   `server_tool_use` (`srvtoolu_…`) + `web_search_tool_result` (`web_search_result` entries with
   `url`, `title`, `page_age`, `encrypted_content`), and fed back to the model;
3. the loop ends with `end_turn`, `tool_use` (a client tool was requested) or `pause_turn`
   (`SEARCH_MAX_ROUNDS` reached, like Anthropic's 10-iteration cap);
4. `max_uses`, `allowed_domains` / `blocked_domains` (bare domain, optional path prefix) and
   `user_location.country` are honoured; errors use `web_search_tool_result_error` with
   `max_uses_exceeded`, `too_many_requests`, `invalid_tool_input`, `query_too_long`, `unavailable`;
5. `usage.server_tool_use.web_search_requests` counts successful searches;
6. replaying a previous answer (multi-turn) restores the results for the upstream model —
   the snippet is carried in `encrypted_content`.

Streaming emits the same event sequence as Anthropic (`content_block_start` `server_tool_use`,
`input_json_delta` with the query, `content_block_start` `web_search_tool_result`, then text).
Without a configured backend, a request carrying the tool gets a `400 invalid_request_error`,
exactly like an organisation with web search disabled. `WebFetch` needs nothing: Claude Code
fetches the page itself and only asks the model to summarise it.

Backends:

| `SEARCH_PROVIDER` | Needs                                  | Notes |
| ----------------- | -------------------------------------- | ----- |
| `searxng`         | `SEARCH_URL=http://host:8080`          | self-hosted, free; JSON API must be enabled (`search.formats: [html, json]`) |
| `brave`           | `SEARCH_API_KEY`                       | https://api.search.brave.com — `country` from `user_location` is forwarded |
| `tavily`          | `SEARCH_API_KEY`                       | https://tavily.com — domain filters passed natively |

`SEARCH_MAX_RESULTS` (default 10) caps results per search, `SEARCH_MAX_ROUNDS` (default 10) caps
model/search iterations per turn.

Endpoints: `POST /v1/messages`, `POST /v1/messages/count_tokens` (rough estimate, ~4 chars/token),
`GET /v1/models`, `GET /health`.

## Build

```bash
cargo build --release
# binary: target/release/claude-openai-proxy
```

## Run

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY=sk-...            # optional; omit for local servers
DEFAULT_MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct \
./target/release/claude-openai-proxy
```

Then point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8082
export ANTHROPIC_API_KEY=dummy      # any non-empty value; forwarded upstream only if OPENAI_API_KEY is unset
claude
```

### Configuration (environment variables)

| Variable                 | Default                    | Meaning |
| ------------------------ | -------------------------- | ------- |
| `PROXY_LISTEN`           | `127.0.0.1:8082`           | Bind address |
| `OPENAI_BASE_URL`        | `http://127.0.0.1:8000/v1` | Upstream base URL, **including `/v1`** |
| `OPENAI_API_KEY`         | unset                      | Upstream bearer token |
| `FORWARD_CLIENT_KEY`     | `true`                     | If no `OPENAI_API_KEY`, forward Claude Code's `x-api-key` / bearer token upstream |
| `MODEL_MAP`              | unset                      | Exact mapping, e.g. `claude-opus-5=gpt-5,claude-haiku-4-5=gpt-5-mini` |
| `OPUS_MODEL` / `SONNET_MODEL` / `HAIKU_MODEL` | unset | Family fallback by substring of the requested model name (`fable`/`mythos` count as opus) |
| `DEFAULT_MODEL`          | unset                      | Used when nothing matches; if unset the requested name is passed through |
| `MAX_TOKENS_CAP`         | unset                      | Clamp `max_tokens` (Claude Code asks for 64k; some servers reject that) |
| `REASONING_EFFORT`       | unset                      | `auto` derives `reasoning_effort` from Claude Code's effort / thinking settings; or a fixed `low`/`medium`/`high`; unset sends nothing |
| `MID_SYSTEM_AS_SYSTEM`   | `false`                    | Forward mid-conversation system messages as OpenAI `system` messages instead of folding them into the user turn |
| `SEARCH_PROVIDER`        | unset                      | `searxng`, `brave` or `tavily` — enables the `web_search` server tool emulation |
| `SEARCH_URL`             | unset                      | SearXNG base URL (required for `searxng`); optional override for the others |
| `SEARCH_API_KEY`         | unset                      | Brave / Tavily API key |
| `SEARCH_MAX_RESULTS`     | `10`                       | Results returned per search |
| `SEARCH_MAX_ROUNDS`      | `10`                       | Model/search iterations per turn before `pause_turn` |
| `LOG_BODIES`             | `false`                    | Log full upstream request/response bodies at `debug` level |
| `RUST_LOG`               | `info`                     | Log filter (`debug` for chunk-level tracing) |

Model resolution order: `MODEL_MAP` exact match → family variable → `DEFAULT_MODEL` → pass-through.

### Examples

vLLM / SGLang on the same box:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 DEFAULT_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
SEARCH_PROVIDER=searxng SEARCH_URL=http://127.0.0.1:8080 ./claude-openai-proxy
```

Ollama:

```bash
OPENAI_BASE_URL=http://127.0.0.1:11434/v1 DEFAULT_MODEL=qwen3-coder MAX_TOKENS_CAP=16384 ./claude-openai-proxy
```

OpenRouter with per-family routing:

```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=sk-or-... \
OPUS_MODEL=anthropic/claude-opus-5 SONNET_MODEL=qwen/qwen3-coder HAIKU_MODEL=google/gemini-flash \
./claude-openai-proxy
```

## Notes and limitations

- Prompt caching, `cache_control`, beta headers and `output_config` are accepted and ignored
  (except `output_config.effort` when `REASONING_EFFORT=auto`).
- `web_search_*` is emulated (see above). Other Anthropic server-side tools (`web_fetch_*`,
  `bash_*`, `text_editor_*`, code execution, ...) have no OpenAI equivalent and are dropped.
  Claude Code's built-in tools, including `WebFetch`, are ordinary function tools and go through fine.
- Citations (`web_search_result_location`) are not generated; Claude Code does not use them.
- If the model requests a client tool and a web search in the same round, the search is still
  executed before returning `stop_reason: tool_use` (Anthropic defers it to the next request).
- Tool names longer than 64 characters (possible with MCP tools) may be rejected by strict
  upstreams; the proxy does not rename them.
- `count_tokens` is an estimate, not the upstream tokenizer.
- Thinking blocks Claude Code sends back are never replayed upstream; upstream
  `reasoning_content` / `reasoning` is surfaced to Claude Code as a `thinking` block with an
  empty signature.
- Malformed tool-call JSON from the upstream is wrapped as `{"_raw_arguments": "..."}` rather than
  silently replaced, so the failure is visible in Claude Code's tool error.

## Development

```bash
cargo test          # unit tests for conversion, SSE parsing and stream translation
RUST_LOG=debug LOG_BODIES=1 cargo run
```
