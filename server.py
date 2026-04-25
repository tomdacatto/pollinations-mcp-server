"""
Pollinations AI — FastMCP Server
Hosted on Prefect Horizon

API docs: https://gen.pollinations.ai/api/docs
Get your key: https://enter.pollinations.ai
"""

import os
import io
import asyncio
import base64
from typing import Optional
from urllib.parse import quote, urlencode

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE   = "https://gen.pollinations.ai"
MEDIA_BASE = "https://media.pollinations.ai"

# Timeouts — all wrapped in asyncio.wait_for so a slow request NEVER blocks
T_DEFAULT   = 60
T_IMG2IMG   = 300   # editing is slow
T_VIDEO     = 240
T_MEDIA     = 45

_api_key: str = os.environ.get("POLLINATIONS_API_KEY", "")

mcp = FastMCP(
    "pollinations-mcp",
    instructions="""
Pollinations AI MCP Server. Base: https://gen.pollinations.ai
Auth: POLLINATIONS_API_KEY env var, or call set_api_key.

EDITING WORKFLOW (correct way):
  1. edit_image(prompt, image=URL, model="gpt-image-2", response_format="url")
  2. Download the returned imageUrl

Key tools:
  edit_image           — POST /v1/images/edits (use this for img2img, ALWAYS response_format=url)
  generate_image_url   — GET /image/{prompt}  (text-to-image, returns URL)
  generate_image       — GET /image/{prompt}  (text-to-image, returns base64)
  upload_media         — upload to media.pollinations.ai → stable URL
  generate_video_url   — returns video URL
  say_text             — TTS
  respond_audio        — AI spoken response
  transcribe_audio     — STT
  web_search           — search with citations
  generate_text        — simple text gen
  chat_completion      — full chat

1 pollen = $1 USD.
""",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key() -> str:
    return _api_key or os.environ.get("POLLINATIONS_API_KEY", "")

def _h(ct: Optional[str] = None) -> dict:
    h = {}
    k = _key()
    if k:
        h["Authorization"] = f"Bearer {k}"
    if ct:
        h["Content-Type"] = ct
    return h

def _require_key():
    if not _key():
        raise ToolError("API key required. Set POLLINATIONS_API_KEY or call set_api_key.")

def _url(path: str, params: dict) -> str:
    u = f"{API_BASE}{path}"
    clean = {k: v for k, v in params.items() if v is not None}
    if clean:
        u += "?" + urlencode(clean)
    return u

async def _get_bytes(url: str, timeout: int = T_DEFAULT) -> tuple[bytes, str]:
    async def _do():
        async with httpx.AsyncClient(timeout=timeout + 10) as c:
            r = await c.get(url, headers=_h(), follow_redirects=True)
            r.raise_for_status()
            return r.content, r.headers.get("content-type", "application/octet-stream")
    try:
        return await asyncio.wait_for(_do(), timeout=timeout)
    except asyncio.TimeoutError:
        raise ToolError(f"Request timed out after {timeout}s. Try a faster model or smaller size.")

async def _post(path: str, body: dict, timeout: int = T_DEFAULT) -> dict:
    async def _do():
        async with httpx.AsyncClient(timeout=timeout + 10) as c:
            r = await c.post(f"{API_BASE}{path}", json=body, headers=_h("application/json"))
            r.raise_for_status()
            return r.json()
    try:
        return await asyncio.wait_for(_do(), timeout=timeout)
    except asyncio.TimeoutError:
        raise ToolError(f"Request timed out after {timeout}s.")

def _iparams(model, width, height, seed, enhance, neg, quality,
             image, transparent, nologo, nofeed, safe, private) -> dict:
    return {k: v for k, v in {
        "model": model, "width": width, "height": height, "seed": seed,
        "enhance": enhance, "negative_prompt": neg, "quality": quality,
        "image": image, "transparent": transparent, "nologo": nologo,
        "nofeed": nofeed, "safe": safe, "private": private,
    }.items() if v is not None}


# ===========================================================================
# AUTH
# ===========================================================================

@mcp.tool()
async def set_api_key(key: str) -> dict:
    """Set Pollinations API key (pk_… or sk_…). Prefer POLLINATIONS_API_KEY env var."""
    global _api_key
    if not key.startswith(("pk_", "sk_")):
        raise ToolError("Key must start with pk_ or sk_")
    _api_key = key
    return {"success": True, "keyType": "publishable" if key.startswith("pk_") else "secret",
            "maskedKey": f"{key[:3]}...{key[-6:]}"}

@mcp.tool()
async def get_key_info() -> dict:
    """Get info about the active API key."""
    k = _key()
    if not k:
        return {"authenticated": False}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://enter.pollinations.ai/api/account/key", headers=_h())
            if r.status_code == 200:
                return {**r.json(), "maskedKey": f"{k[:3]}...{k[-6:]}"}
    except Exception:
        pass
    return {"authenticated": True, "maskedKey": f"{k[:3]}...{k[-6:]}"}

@mcp.tool()
async def clear_api_key() -> dict:
    """Clear the in-memory API key."""
    global _api_key
    had = bool(_api_key)
    _api_key = ""
    return {"cleared": had}


# ===========================================================================
# IMAGE EDITING  ← the important one
# ===========================================================================

@mcp.tool()
async def edit_image(
    prompt: str,
    image: str,
    model: str = "gpt-image-2",
    size: str = "1024x1024",
    quality: str = "hd",
    response_format: str = "url",
    seed: Optional[int] = None,
    nologo: Optional[bool] = None,
    enhance: Optional[bool] = None,
    safe: Optional[bool] = None,
) -> dict:
    """
    Edit an image using POST /v1/images/edits (OpenAI-compatible).
    ALWAYS use response_format="url" — b64_json is huge and slow.

    Best models:
      gpt-image-2    — best quality OpenAI (default)
      gptimage-large — GPT Image 1.5
      kontext        — FLUX Kontext, great for style/color edits
      nanobanana-2   — Gemini 3.1 Flash
      klein          — FLUX.2 Klein, fastest
      wan-image      — Alibaba up to 2K

    image: source image URL (wants-to.party or media.pollinations.ai)
    size: "1024x1024" | "2048x2048" | "1024x1792"
    quality: low | medium | high | hd
    response_format: always "url" unless you really need "b64_json"
    """
    _require_key()
    body: dict = {
        "prompt": prompt, "image": image, "model": model,
        "size": size, "quality": quality, "response_format": "b64_json",  # url format broken on most models
    }
    for k, v in {"seed": seed, "nologo": nologo, "enhance": enhance, "safe": safe}.items():
        if v is not None:
            body[k] = v

    async def _do():
        async with httpx.AsyncClient(timeout=T_IMG2IMG + 30) as c:
            r = await c.post(f"{API_BASE}/v1/images/edits", json=body, headers=_h("application/json"))
            r.raise_for_status()
            return r.json()

    try:
        result = await asyncio.wait_for(_do(), timeout=T_IMG2IMG)
    except asyncio.TimeoutError:
        raise ToolError(f"Image editing timed out after {T_IMG2IMG}s. Try model='klein' for faster results.")

    item = (result.get("data") or [{}])[0]
    out = {"prompt": prompt, "model": model, "size": size}

    # Try URL first, fall back to b64_json
    image_url = item.get("url", "")
    b64 = item.get("b64_json", "")

    if image_url:
        out["imageUrl"] = image_url
    elif b64:
        # Upload to Pollinations media storage to get a stable URL
        try:
            img_bytes = base64.b64decode(b64)
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(
                    f"{MEDIA_BASE}/upload",
                    headers=_h(),
                    files={"file": ("edited.jpg", io.BytesIO(img_bytes), "image/jpeg")},
                )
                if r.status_code == 200:
                    out["imageUrl"] = r.json().get("url", "")
                else:
                    # Return b64 directly as fallback
                    out["base64"] = b64
                    out["mimeType"] = "image/jpeg"
        except Exception:
            out["base64"] = b64
            out["mimeType"] = "image/jpeg"
    else:
        raise ToolError("Pollinations returned no image data. The model may not support editing.")

    return out


# ===========================================================================
# IMAGE GENERATION
# ===========================================================================

@mcp.tool()
async def generate_image_url(
    prompt: str,
    model: str = "flux",
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    enhance: Optional[bool] = None,
    negative_prompt: Optional[str] = None,
    quality: Optional[str] = None,
    image: Optional[str] = None,
    transparent: Optional[bool] = None,
    nologo: Optional[bool] = None,
    nofeed: Optional[bool] = None,
    safe: Optional[bool] = None,
    private: Optional[bool] = None,
) -> dict:
    """
    Generate an image URL (no download). For editing use edit_image.
    Models: flux, zimage, gptimage, gptimage-large, gpt-image-2,
            kontext, nanobanana, nanobanana-2, seedream5, klein, wan-image
    quality: low | medium | high | hd
    """
    _require_key()
    params = _iparams(model, width, height, seed, enhance, negative_prompt,
                      quality, image, transparent, nologo, nofeed, safe, private)
    u = _url(f"/image/{quote(prompt, safe='')}", params)
    return {"imageUrl": u, "prompt": prompt, "model": model, "width": width, "height": height}

@mcp.tool()
async def generate_image(
    prompt: str,
    model: str = "flux",
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    enhance: Optional[bool] = None,
    negative_prompt: Optional[str] = None,
    quality: Optional[str] = None,
    nologo: Optional[bool] = None,
    nofeed: Optional[bool] = None,
    safe: Optional[bool] = None,
) -> dict:
    """Generate image, return base64. For editing use edit_image."""
    _require_key()
    params = _iparams(model, width, height, seed, enhance, negative_prompt,
                      quality, None, None, nologo, nofeed, safe, None)
    u = _url(f"/image/{quote(prompt, safe='')}", params)
    data, ct = await _get_bytes(u)
    return {"base64": base64.b64encode(data).decode(), "mimeType": ct,
            "prompt": prompt, "model": model}

@mcp.tool()
async def list_image_models() -> dict:
    """List all image/video models live from the API."""
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{API_BASE}/image/models")
        r.raise_for_status()
        models = r.json()
    if isinstance(models, dict):
        models = models.get("models", [])
    imgs = [m for m in models if "video" not in (m.get("output_modalities") or [])]
    vids = [m for m in models if "video" in (m.get("output_modalities") or [])]
    return {
        "imageModels": [{"name": m["name"], "description": m.get("description"),
                         "imageInput": m.get("image_input", False)} for m in imgs],
        "videoModels": [{"name": m["name"], "description": m.get("description")} for m in vids],
        "total": len(models),
    }


# ===========================================================================
# MEDIA STORAGE
# ===========================================================================

@mcp.tool()
async def upload_media(file_url: str, filename: Optional[str] = None) -> dict:
    """
    Download file_url and re-host on media.pollinations.ai.
    Returns stable public URL. 14-day TTL, max 10MB.
    Use before edit_image if your source URL might be unstable.
    """
    _require_key()
    async def _do():
        async with httpx.AsyncClient(timeout=T_MEDIA + 10) as c:
            src = await c.get(file_url, follow_redirects=True)
            src.raise_for_status()
            ct = src.headers.get("content-type", "image/jpeg")
            fn = filename or file_url.split("/")[-1].split("?")[0] or "upload"
            r = await c.post(f"{MEDIA_BASE}/upload", headers=_h(),
                             files={"file": (fn, io.BytesIO(src.content), ct)})
            r.raise_for_status()
            return r.json()
    try:
        data = await asyncio.wait_for(_do(), timeout=T_MEDIA)
    except asyncio.TimeoutError:
        raise ToolError("Media upload timed out.")
    return {"url": data.get("url", ""), "id": data.get("id", ""),
            "size": data.get("size", 0), "duplicate": data.get("duplicate", False)}


# ===========================================================================
# VIDEO
# ===========================================================================

@mcp.tool()
async def generate_video_url(
    prompt: str,
    model: str = "ltx-2",
    duration: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    audio: Optional[bool] = None,
    image: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """Get a shareable video URL. Models: ltx-2 (free), veo, seedance, wan, nova-reel (paid)."""
    _require_key()
    params = {k: v for k, v in {"model": model, "duration": duration,
              "aspectRatio": aspect_ratio, "audio": audio,
              "image": image, "seed": seed}.items() if v is not None}
    u = _url(f"/image/{quote(prompt, safe='')}", params)
    return {"videoUrl": u, "prompt": prompt, "model": model}

@mcp.tool()
async def generate_video(
    prompt: str,
    model: str = "ltx-2",
    duration: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    audio: Optional[bool] = None,
    image: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """Generate video, return base64 mp4. Free model: ltx-2. Paid: veo, seedance, wan."""
    _require_key()
    params = {k: v for k, v in {"model": model, "duration": duration,
              "aspectRatio": aspect_ratio, "audio": audio,
              "image": image, "seed": seed}.items() if v is not None}
    u = _url(f"/image/{quote(prompt, safe='')}", params)
    data, ct = await _get_bytes(u, timeout=T_VIDEO)
    return {"base64": base64.b64encode(data).decode(), "mimeType": ct or "video/mp4",
            "prompt": prompt, "model": model}


# ===========================================================================
# VISION
# ===========================================================================

@mcp.tool()
async def describe_image(
    image_url: str,
    prompt: str = "Describe this image in detail.",
    model: str = "openai",
) -> dict:
    """Analyze an image. model: openai | gemini | claude | nanobanana"""
    _require_key()
    result = await _post("/v1/chat/completions", {"model": model, "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}]})
    return {"description": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "model": model}

@mcp.tool()
async def analyze_video(video_url: str, prompt: str = "Describe what happens.", model: str = "gemini-large") -> dict:
    """Analyze a video URL. Uses Gemini for native video understanding."""
    _require_key()
    result = await _post("/v1/chat/completions", {"model": model, "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": video_url}},
        ]}]})
    return {"analysis": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "model": model}


# ===========================================================================
# TEXT
# ===========================================================================

@mcp.tool()
async def generate_text(
    prompt: str,
    model: str = "openai",
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    json_mode: Optional[bool] = None,
) -> str:
    """Simple text gen. Models: openai, openai-fast, openai-large, claude, claude-fast,
    gemini, gemini-large, deepseek, grok, mistral, qwen-coder, kimi, perplexity-fast..."""
    _require_key()
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    body: dict = {"model": model, "messages": msgs}
    for k, v in {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}.items():
        if v is not None:
            body[k] = v
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    result = await _post("/v1/chat/completions", body)
    return result.get("choices", [{}])[0].get("message", {}).get("content", "")

@mcp.tool()
async def chat_completion(
    messages: list[dict],
    model: str = "openai",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    response_format: Optional[dict] = None,
    tools: Optional[list] = None,
    reasoning_effort: Optional[str] = None,
) -> dict:
    """Full OpenAI-compatible chat completions. Supports tools, reasoning, JSON mode."""
    _require_key()
    body: dict = {"model": model, "messages": messages}
    for k, v in {"temperature": temperature, "max_tokens": max_tokens, "seed": seed,
                 "response_format": response_format, "tools": tools,
                 "reasoning_effort": reasoning_effort}.items():
        if v is not None:
            body[k] = v
    result = await _post("/v1/chat/completions", body)
    choice = result.get("choices", [{}])[0]
    msg = choice.get("message", {})
    out = {"content": msg.get("content", ""), "model": result.get("model", model),
           "finish_reason": choice.get("finish_reason"), "usage": result.get("usage")}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    if msg.get("reasoning_content"):
        out["reasoning"] = msg["reasoning_content"]
    if result.get("citations"):
        out["citations"] = result["citations"]
    return out

@mcp.tool()
async def web_search(query: str, model: str = "perplexity-fast") -> dict:
    """Web search with citations. model: perplexity-fast | perplexity-reasoning | gemini-search"""
    _require_key()
    result = await _post("/v1/chat/completions", {"model": model, "messages": [
        {"role": "system", "content": "Answer accurately with source URLs."},
        {"role": "user", "content": query}]})
    return {"answer": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "query": query, "citations": result.get("citations", [])}

@mcp.tool()
async def list_text_models() -> dict:
    """List all text models live from the API."""
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{API_BASE}/v1/models")
        r.raise_for_status()
        data = r.json()
    models = data.get("data", data) if isinstance(data, dict) else data
    return {"models": models, "total": len(models)}


# ===========================================================================
# AUDIO
# ===========================================================================

@mcp.tool()
async def say_text(
    text: str,
    voice: str = "alloy",
    model: str = "elevenlabs",
    format: str = "mp3",
    seed: Optional[int] = None,
) -> dict:
    """TTS. model: elevenlabs | openai-audio | qwen-tts. Returns base64 audio.
    Voices: alloy, echo, fable, onyx, nova, shimmer, rachel, domi, bella, josh..."""
    _require_key()
    params: dict = {"model": model, "voice": voice, "format": format}
    if seed is not None:
        params["seed"] = seed
    u = _url(f"/audio/{quote(text, safe='')}", params)
    data, ct = await _get_bytes(u)
    return {"base64": base64.b64encode(data).decode(), "mimeType": ct or f"audio/{format}",
            "voice": voice}

@mcp.tool()
async def respond_audio(
    prompt: str,
    voice: str = "alloy",
    format: str = "mp3",
    voice_instructions: Optional[str] = None,
) -> dict:
    """AI spoken response. Returns base64 audio."""
    _require_key()
    full = f"{voice_instructions}\n\n{prompt}" if voice_instructions else prompt
    u = _url(f"/text/{quote(full, safe='')}", {"model": "openai-audio", "voice": voice, "format": format})
    data, ct = await _get_bytes(u)
    return {"base64": base64.b64encode(data).decode(), "mimeType": ct or f"audio/{format}",
            "voice": voice}

@mcp.tool()
async def transcribe_audio(audio_url: str, model: str = "scribe") -> dict:
    """STT from URL. model: scribe (90+ langs, diarization) | whisper"""
    _require_key()
    result = await _post("/v1/chat/completions", {"model": model, "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Transcribe this audio accurately."},
            {"type": "input_audio", "input_audio": {"url": audio_url}},
        ]}]})
    return {"transcription": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "model": model}

@mcp.tool()
async def list_audio_voices() -> dict:
    """List available TTS voices and audio models."""
    return {
        "voices": ["alloy","echo","fable","onyx","nova","shimmer","ash","ballad","coral",
                   "sage","verse","rachel","domi","bella","elli","charlotte","dorothy",
                   "sarah","emily","lily","matilda","adam","antoni","arnold","josh","sam",
                   "daniel","charlie","james","fin","callum","liam","george","brian","bill"],
        "formats": ["mp3","wav","flac","opus"],
        "models": ["elevenlabs","openai-audio","openai-audio-large","qwen-tts","qwen-tts-instruct"],
    }


# ===========================================================================
# ACCOUNT
# ===========================================================================

@mcp.tool()
async def get_balance() -> dict:
    """Get Pollen balance. 1 pollen = $1 USD."""
    _require_key()
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{API_BASE}/api/account/balance", headers=_h())
        r.raise_for_status()
    return {"pollen": r.json().get("balance"), "note": "1 pollen = $1 USD"}

@mcp.tool()
async def get_usage(daily: bool = False, days: Optional[int] = None, limit: Optional[int] = None) -> dict:
    """Usage history. daily=True for daily aggregated, False for per-request."""
    _require_key()
    path = "/api/account/usage/daily" if daily else "/api/account/usage"
    params: dict = {}
    if days:
        params["days"] = days
    if limit and not daily:
        params["limit"] = limit
    u = f"{API_BASE}{path}" + (("?" + urlencode(params)) if params else "")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(u, headers=_h())
        r.raise_for_status()
        data = r.json()
    return {"mode": "daily" if daily else "per-request",
            "records": data.get("usage", []), "count": data.get("count", 0)}


# ===========================================================================
# Entry
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(mcp.get_asgi_app(), host="0.0.0.0", port=8000)
