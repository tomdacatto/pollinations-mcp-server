"""
Pollinations AI — FastMCP Server
Hosted on Prefect Horizon

API docs: https://gen.pollinations.ai/api/docs
Get your key: https://enter.pollinations.ai
"""

import os
import io
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
MEDIA_BASE = "https://media.pollinations.ai"  # content-addressed file storage

TIMEOUT         = 120.0   # default
TIMEOUT_IMG2IMG = 360.0   # editing / img2img is slow (docs say 300s)
TIMEOUT_VIDEO   = 300.0   # video generation

_api_key: str = os.environ.get("POLLINATIONS_API_KEY", "")


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "pollinations-mcp",
    instructions="""
Pollinations AI MCP Server.
Base URL: https://gen.pollinations.ai

AUTH: Set POLLINATIONS_API_KEY env var on Horizon, or call set_api_key.

TOOLS:
  Image generation   → generate_image, generate_image_url, generate_image_batch
  Image editing      → edit_image  (POST /v1/images/edits — use this for img2img)
  Video              → generate_video, generate_video_url
  Vision             → describe_image, analyze_video
  Text               → generate_text, chat_completion, web_search
  Audio              → say_text, respond_audio, transcribe_audio
  Media storage      → upload_media (→ public URL for use with edit_image)
  Lists              → list_image_models, list_text_models, list_audio_voices
  Account            → get_balance, get_usage
  Auth               → set_api_key, get_key_info, clear_api_key

IMAGE EDITING WORKFLOW:
  1. upload_media(url) → get a stable media.pollinations.ai URL
  2. edit_image(prompt, image=that_url, model="gpt-image-2") → edited image

Key image models (with image editing support):
  gpt-image-2, gptimage-large, gptimage, kontext, nanobanana, nanobanana-2,
  nanobanana-pro, wan-image, wan-image-pro, qwen-image, klein, p-image-edit,
  nova-canvas, seedream5

1 pollen = $1 USD.
""",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key() -> str:
    return _api_key or os.environ.get("POLLINATIONS_API_KEY", "")


def _auth_headers(content_type: Optional[str] = None) -> dict:
    h = {}
    k = _key()
    if k:
        h["Authorization"] = f"Bearer {k}"
    if content_type:
        h["Content-Type"] = content_type
    return h


def _require_key():
    if not _key():
        raise ToolError(
            "API key required. Call set_api_key first, or set "
            "POLLINATIONS_API_KEY in Horizon env vars. "
            "Get yours: https://enter.pollinations.ai"
        )


def _build_url(path: str, params: dict) -> str:
    url = f"{API_BASE}{path}"
    clean = {k: v for k, v in params.items() if v is not None}
    if clean:
        url += "?" + urlencode(clean)
    return url


def _img_params(model, width, height, seed, enhance, negative_prompt,
                quality, image, transparent, nologo, nofeed, safe, private) -> dict:
    return {k: v for k, v in {
        "model": model, "width": width, "height": height, "seed": seed,
        "enhance": enhance, "negative_prompt": negative_prompt,
        "quality": quality, "image": image, "transparent": transparent,
        "nologo": nologo, "nofeed": nofeed, "safe": safe, "private": private,
    }.items() if v is not None}


async def _get(url: str, timeout: float = TIMEOUT) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(url, headers=_auth_headers(), follow_redirects=True)
        r.raise_for_status()
        return r.content, r.headers.get("content-type", "application/octet-stream")


async def _post(path: str, body: dict, timeout: float = TIMEOUT) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(
            f"{API_BASE}{path}",
            json=body,
            headers=_auth_headers("application/json"),
        )
        r.raise_for_status()
        return r.json()


# ===========================================================================
# AUTH
# ===========================================================================

@mcp.tool()
async def set_api_key(key: str) -> dict:
    """Set your Pollinations API key (pk_… or sk_…). Prefer POLLINATIONS_API_KEY env var for persistence."""
    global _api_key
    if not key.startswith(("pk_", "sk_")):
        raise ToolError("Key must start with pk_ or sk_. Get one at https://enter.pollinations.ai")
    _api_key = key
    return {"success": True, "keyType": "publishable" if key.startswith("pk_") else "secret",
            "maskedKey": f"{key[:3]}...{key[-6:]}"}


@mcp.tool()
async def get_key_info() -> dict:
    """Get info about the currently active API key (live from API)."""
    k = _key()
    if not k:
        return {"authenticated": False, "message": "No key. Call set_api_key or set POLLINATIONS_API_KEY."}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://enter.pollinations.ai/api/account/key", headers=_auth_headers())
            if r.status_code == 200:
                return {**r.json(), "maskedKey": f"{k[:3]}...{k[-6:]}"}
    except Exception:
        pass
    return {"authenticated": True, "keyType": "publishable" if k.startswith("pk_") else "secret",
            "maskedKey": f"{k[:3]}...{k[-6:]}"}


@mcp.tool()
async def clear_api_key() -> dict:
    """Clear the in-memory API key (does not affect env var)."""
    global _api_key
    had = bool(_api_key)
    _api_key = ""
    return {"success": True, "message": "Cleared" if had else "No key was set"}


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
    Generate an image and return a shareable URL (no binary download).
    For image EDITING use edit_image — it uses the dedicated POST endpoint.
    Models: flux, zimage, kontext, gptimage, gptimage-large, gpt-image-2,
            nanobanana, nanobanana-2, nanobanana-pro, seedream5, klein,
            wan-image, nova-canvas, qwen-image, p-image
    quality: low | medium | high | hd  (gptimage / gpt-image-2 only)
    """
    _require_key()
    params = _img_params(model, width, height, seed, enhance, negative_prompt,
                         quality, image, transparent, nologo, nofeed, safe, private)
    url = _build_url(f"/image/{quote(prompt, safe='')}", params)
    return {"imageUrl": url, "prompt": prompt, "model": model, "width": width, "height": height}


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
    private: Optional[bool] = None,
) -> dict:
    """
    Generate an image and return base64 data.
    For image EDITING (passing a source image) use edit_image instead.
    Models: flux, zimage, kontext, gptimage, gptimage-large, gpt-image-2,
            nanobanana, seedream5, klein, wan-image, nova-canvas, p-image
    quality: low | medium | high | hd
    """
    _require_key()
    params = _img_params(model, width, height, seed, enhance, negative_prompt,
                         quality, None, None, nologo, nofeed, safe, private)
    url = _build_url(f"/image/{quote(prompt, safe='')}", params)
    data, ct = await _get(url)
    return {"base64": base64.b64encode(data).decode(), "mimeType": ct,
            "prompt": prompt, "model": model, "width": width, "height": height}


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
    Edit / transform an image using a text prompt.
    Uses POST /v1/images/edits — the dedicated OpenAI-compatible editing endpoint.

    BEST MODELS for editing:
      gpt-image-2        — best quality (default)
      gptimage-large     — GPT Image 1.5
      gptimage           — GPT Image 1 Mini
      kontext            — FLUX Kontext (great for style transfer)
      nanobanana-2       — Gemini 3.1 Flash
      wan-image          — Alibaba, up to 2K
      nova-canvas        — Bedrock
      klein              — FLUX.2 Klein

    image: URL of the source image (use upload_media to get a stable URL first)
    size: "1024x1024" | "2048x2048" | "1024x1792" etc.
    quality: low | medium | high | hd
    response_format: "url" (returns imageUrl) | "b64_json" (returns base64)

    WORKFLOW:
      1. upload_media(your_image_url) → get stable URL
      2. edit_image(prompt, image=stable_url, model="gpt-image-2")
    """
    _require_key()
    body: dict = {
        "prompt": prompt,
        "image": image,
        "model": model,
        "size": size,
        "quality": quality,
        "response_format": response_format,
    }
    for k, v in {"seed": seed, "nologo": nologo, "enhance": enhance, "safe": safe}.items():
        if v is not None:
            body[k] = v

    async with httpx.AsyncClient(timeout=TIMEOUT_IMG2IMG) as c:
        r = await c.post(
            f"{API_BASE}/v1/images/edits",
            json=body,
            headers=_auth_headers("application/json"),
        )
        r.raise_for_status()
        result = r.json()

    item = (result.get("data") or [{}])[0]
    out = {"prompt": prompt, "model": model, "size": size}
    if response_format == "url":
        out["imageUrl"] = item.get("url", "")
    else:
        out["base64"] = item.get("b64_json", "")
        out["mimeType"] = "image/jpeg"
    return out


@mcp.tool()
async def generate_image_batch(
    prompts: list[str],
    model: str = "flux",
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    quality: Optional[str] = None,
    nologo: Optional[bool] = None,
    safe: Optional[bool] = None,
) -> dict:
    """Generate up to 10 images in parallel. Best with sk_ keys."""
    _require_key()
    if not prompts or len(prompts) > 10:
        raise ToolError("Provide 1–10 prompts")
    import asyncio

    async def _one(p: str, i: int):
        params = _img_params(model, width, height,
                             (seed + i) if seed is not None else None,
                             None, None, quality, None, None, nologo, None, safe, None)
        url = _build_url(f"/image/{quote(p, safe='')}", params)
        data, ct = await _get(url)
        return {"index": i, "prompt": p, "base64": base64.b64encode(data).decode(), "mimeType": ct}

    results = await asyncio.gather(*[_one(p, i) for i, p in enumerate(prompts)], return_exceptions=True)
    ok = [r for r in results if not isinstance(r, Exception)]
    err = [str(r) for r in results if isinstance(r, Exception)]
    return {"total": len(prompts), "successful": len(ok), "failed": len(err), "images": ok, "errors": err}


@mcp.tool()
async def list_image_models() -> dict:
    """List all image/video models live from the API."""
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{API_BASE}/image/models")
        r.raise_for_status()
        models = r.json()
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
    Download a file from file_url and upload it to Pollinations media storage.
    Returns a stable public URL (https://media.pollinations.ai/<hash>).
    Files are content-addressed and expire after 14 days (re-upload resets TTL).
    Max 10 MB.

    Use this to get a stable URL for any image before passing it to edit_image.
    """
    _require_key()
    async with httpx.AsyncClient(timeout=60) as c:
        src = await c.get(file_url, follow_redirects=True)
        src.raise_for_status()
        ct = src.headers.get("content-type", "image/jpeg")
        fbytes = src.content

    fn = filename or file_url.split("/")[-1].split("?")[0] or "upload"
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"{MEDIA_BASE}/upload",
            headers=_auth_headers(),
            files={"file": (fn, io.BytesIO(fbytes), ct)},
        )
        r.raise_for_status()
        data = r.json()

    return {
        "url": data.get("url", ""),
        "id": data.get("id", ""),
        "size": data.get("size", len(fbytes)),
        "contentType": ct,
        "duplicate": data.get("duplicate", False),
    }


# ===========================================================================
# VIDEO
# ===========================================================================

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
    """
    Generate a video. Returns base64 mp4.
    Models (free): ltx-2
    Models (paid): veo, seedance, seedance-pro, wan, wan-fast, grok-video-pro, nova-reel
    duration: 1-15s (model dependent)
    aspect_ratio: '16:9' | '9:16'
    audio: True to enable audio track (veo, wan)
    image: reference image URL for image-to-video
    """
    _require_key()
    params = {k: v for k, v in {
        "model": model, "duration": duration, "aspectRatio": aspect_ratio,
        "audio": audio, "image": image, "seed": seed,
    }.items() if v is not None}
    url = _build_url(f"/image/{quote(prompt, safe='')}", params)
    data, ct = await _get(url, timeout=TIMEOUT_VIDEO)
    return {"base64": base64.b64encode(data).decode(), "mimeType": ct or "video/mp4",
            "prompt": prompt, "model": model}


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
    """Get a shareable video URL without downloading binary."""
    _require_key()
    params = {k: v for k, v in {
        "model": model, "duration": duration, "aspectRatio": aspect_ratio,
        "audio": audio, "image": image, "seed": seed,
    }.items() if v is not None}
    url = _build_url(f"/image/{quote(prompt, safe='')}", params)
    return {"videoUrl": url, "prompt": prompt, "model": model}


# ===========================================================================
# VISION
# ===========================================================================

@mcp.tool()
async def describe_image(
    image_url: str,
    prompt: str = "Describe this image in detail.",
    model: str = "openai",
) -> dict:
    """Analyze an image with a vision model. model: openai | gemini | claude | nanobanana"""
    _require_key()
    result = await _post("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
    })
    return {"description": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "imageUrl": image_url, "model": model}


@mcp.tool()
async def analyze_video(
    video_url: str,
    prompt: str = "Describe what happens in this video.",
    model: str = "gemini-large",
) -> dict:
    """Analyze a video (YouTube or direct URL). Uses Gemini for native video understanding."""
    _require_key()
    result = await _post("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": video_url}},
        ]}],
    })
    return {"analysis": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "videoUrl": video_url, "model": model}


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
    """
    Simple text generation.
    Models: openai, openai-fast, openai-large, claude, claude-fast, claude-large,
            gemini, gemini-large, deepseek, grok, grok-large, mistral, qwen-coder,
            kimi, perplexity-fast, perplexity-reasoning, nova-fast, nova, glm, ...
    """
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
    """
    Full OpenAI-compatible chat completions. Supports tool calling, reasoning, JSON mode.
    messages: [{"role":"user"|"assistant"|"system","content":"..."}]
    reasoning_effort: low | medium | high
    """
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
async def web_search(
    query: str,
    model: str = "perplexity-fast",
    detailed: bool = False,
) -> dict:
    """Web search with citations. model: perplexity-fast | perplexity-reasoning | gemini-search"""
    _require_key()
    result = await _post("/v1/chat/completions", {
        "model": model,
        "messages": [
            {"role": "system", "content": "Search the web and provide an accurate answer with source URLs."},
            {"role": "user", "content": query},
        ],
    })
    return {
        "answer": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
        "query": query, "model": result.get("model", model),
        "citations": result.get("citations", []),
    }


@mcp.tool()
async def list_text_models() -> dict:
    """List all text models live from the API."""
    async with httpx.AsyncClient(timeout=30) as c:
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
    """
    Text-to-speech. Returns base64 audio.
    model: elevenlabs | openai-audio | openai-audio-large | qwen-tts | qwen-tts-instruct
    voice: alloy, echo, fable, onyx, nova, shimmer, rachel, domi, bella, elli,
           josh, sam, daniel, charlie, james, fin, callum, liam, george, brian, bill ...
    format: mp3 | wav | flac | opus
    """
    _require_key()
    params = {"model": model, "voice": voice, "format": format}
    if seed is not None:
        params["seed"] = seed
    url = _build_url(f"/audio/{quote(text, safe='')}", params)
    data, ct = await _get(url)
    return {"base64": base64.b64encode(data).decode(), "mimeType": ct or f"audio/{format}",
            "text": text, "voice": voice, "format": format}


@mcp.tool()
async def respond_audio(
    prompt: str,
    voice: str = "alloy",
    format: str = "mp3",
    voice_instructions: Optional[str] = None,
) -> dict:
    """
    Generate a spoken AI response to a prompt. Returns base64 audio.
    voice: alloy | echo | fable | onyx | nova | shimmer | coral | verse | ballad | ash | sage
    """
    _require_key()
    full = f"{voice_instructions}\n\n{prompt}" if voice_instructions else prompt
    url = _build_url(f"/text/{quote(full, safe='')}", {"model": "openai-audio", "voice": voice, "format": format})
    data, ct = await _get(url)
    return {"base64": base64.b64encode(data).decode(), "mimeType": ct or f"audio/{format}",
            "prompt": prompt, "voice": voice, "format": format}


@mcp.tool()
async def transcribe_audio(
    audio_url: str,
    model: str = "scribe",
    prompt: str = "Transcribe this audio accurately.",
) -> dict:
    """
    Speech-to-text from a URL.
    model: scribe (90+ languages, diarization) | whisper (fast)
    """
    _require_key()
    result = await _post("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"url": audio_url}},
        ]}],
    })
    return {"transcription": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "audioUrl": audio_url, "model": model}


@mcp.tool()
async def list_audio_voices() -> dict:
    """List available TTS voices, formats, and audio models."""
    return {
        "voices": [
            "alloy", "echo", "fable", "onyx", "nova", "shimmer",
            "ash", "ballad", "coral", "sage", "verse",
            "rachel", "domi", "bella", "elli", "charlotte", "dorothy",
            "sarah", "emily", "lily", "matilda",
            "adam", "antoni", "arnold", "josh", "sam",
            "daniel", "charlie", "james", "fin", "callum", "liam", "george", "brian", "bill",
        ],
        "formats": ["mp3", "wav", "flac", "opus"],
        "models": ["elevenlabs", "openai-audio", "openai-audio-large", "qwen-tts", "qwen-tts-instruct", "elevenmusic", "acestep"],
    }


# ===========================================================================
# ACCOUNT
# ===========================================================================

@mcp.tool()
async def get_balance() -> dict:
    """Get your current Pollen balance. 1 pollen = $1 USD. Requires account:usage permission."""
    _require_key()
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{API_BASE}/api/account/balance", headers=_auth_headers())
        r.raise_for_status()
        data = r.json()
    return {"pollen": data.get("balance"), "note": "1 pollen = $1 USD"}


@mcp.tool()
async def get_usage(
    daily: bool = False,
    days: Optional[int] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Get API usage history.
    daily=True → daily aggregated (date + model)
    daily=False → per-request rows
    Requires account:usage permission.
    """
    _require_key()
    path = "/api/account/usage/daily" if daily else "/api/account/usage"
    params: dict = {}
    if days:
        params["days"] = days
    if limit and not daily:
        params["limit"] = limit
    url = f"{API_BASE}{path}" + (("?" + urlencode(params)) if params else "")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url, headers=_auth_headers())
        r.raise_for_status()
        data = r.json()
    return {"mode": "daily" if daily else "per-request",
            "records": data.get("usage", []), "count": data.get("count", 0)}


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(mcp.get_asgi_app(), host="0.0.0.0", port=8000)
