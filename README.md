# pollinations-mcp-server

FastMCP server wrapping the full [Pollinations AI](https://pollinations.ai) API — hosted on [Prefect Horizon](https://horizon.prefect.io).

## Tools

| Category | Tools |
|----------|-------|
| **Image** | `generate_image_url`, `generate_image`, `generate_image_batch`, `list_image_models` |
| **Video** | `generate_video`, `generate_video_url` |
| **Vision** | `describe_image`, `analyze_video` |
| **Text** | `generate_text`, `chat_completion`, `web_search`, `get_pricing`, `list_text_models` |
| **Audio** | `respond_audio`, `say_text`, `transcribe_audio`, `list_audio_voices` |
| **Auth** | `set_api_key`, `get_key_info`, `clear_api_key` |
| **Account** | `get_balance`, `get_usage` |

## Deploy to Prefect Horizon

1. Sign up at **[horizon.prefect.io](https://horizon.prefect.io)**
2. Connect your GitHub account → select this repo (`tomdacatto/pollinations-mcp-server`)
3. Configure:
   - **Entrypoint:** `server.py:mcp`
   - **Environment variable:** `POLLINATIONS_API_KEY=your_key`
4. Hit **Deploy** → get your live MCP URL in ~60 seconds

Your URL will be: `https://pollinations-mcp.<your-org>.horizon.prefect.io`

## Connect to Claude

Paste your Horizon URL into Claude's **Custom Connectors** (Settings → Connectors).  
Also works with Cursor, Windsurf, VS Code, OpenAI Agents, and any MCP-compatible client.

## Local dev

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your key
fastmcp dev server.py:mcp
```

## Get an API key

→ [enter.pollinations.ai](https://enter.pollinations.ai)

- `pk_` — publishable, rate-limited (1 pollen / IP / hour)
- `sk_` — secret, no rate limits, can spend Pollen
