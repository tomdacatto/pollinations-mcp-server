"""
Pollinations AI — FastMCP Server
Hosted on Prefect Horizon

All tools: image, video, text, audio, auth, account
API: https://gen.pollinations.ai
Get your key: https://enter.pollinations.ai
"""

import os
import base64
from typing import Optional
from urllib.parse import quote, urlencode

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "https://gen.pollinations.ai"
IMAGE_BASE = "https://image.pollinations.ai"  # legacy endpoint, better for img2img
TIMEOUT = 120.0        # default
TIMEOUT_IMG2IMG = 360.0  # image-to-image is slow — official docs use 300s, we add buffer
TIMEOUT_VIDEO = 300.0  # video generation

# Module-level key store — populated from env or via setApiKey tool
_api_key: str = os.environ.get("POLLINATIONS_API_KEY", "")


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "pollinations-mcp",
    instructions="""
Pollinations AI MCP Server — image, video, text, audio generation.

Authentication:
  Set POLLINATIONS_API_KEY in Horizon env vars (recommended), or call setApiKey first.
  pk_ keys: rate-limited (1 pollen / IP / hour)
  sk_ keys: no rate limits, can spend Pollen

Get your key: https://enter.pollinations.ai

Quick reference:
  Images  → generateImage / generateImageUrl / generateImageBatch
  Video   → generateVideo / generateVideoUrl
  Vision  → describeImage / analyzeVideo
  Text    → generateText / chatCompletion / webSearch
  Audio   → respondAudio / sayText / transcribeAudio
  Lists   → listImageModels / listTextModels / listAudioVoices
  Account → getBalance / getUsage
""",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key() -> str:
    return _api_key or os.environ.get("POLLINATIONS_API_KEY", "")


def _headers() -> dict:
    k = _key()
    return {"Authorization": f"Bearer {k}"} if k else {}


def _require_key():
    if not _key():
        raise ToolError(
            "API key required. Call setApiKey first, or set POLLINATIONS_API_KEY "
            "in Horizon's environment variables. Get a key at https://enter.pollinations.ai"
        )


def _image_params(
    model, width, height, seed, enhance, negative_prompt,
    guidance_scale, quality, image, transparent, nologo, nofeed, safe, private
) -> dict:
    raw = {
        "model": model, "width": width, "height": height,
        "seed": seed, "enhance": enhance, "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale, "quality": quality,
        "image": image, "transparent": transparent,
        "nologo": nologo, "nofeed": nofeed, "safe": safe, "private": private,
    }
    return {k: v for k, v in raw.items() if v is not None}


def _build_url(path: str, params: dict) -> str:
    base = f"{API_BASE}{path}"
    if params:
        base += "?" + urlencode(params)
    return base


def _build_image_url(prompt: str, params: dict, use_legacy: bool = False) -> str:
    """Build image generation URL. Uses legacy image.pollinations.ai for img2img (more reliable)."""
    encoded = quote(prompt, safe="")
    base_host = IMAGE_BASE if use_legacy else API_BASE
    url = f"{base_host}/image/{encoded}" if not use_legacy else f"{base_host}/prompt/{encoded}"
    if params:
        url += "?" + urlencode({k: v for k, v in params.items() if v is not None})
    return url


def _shareable_url(path: str, params: dict) -> str:
    clean = {k: v for k, v in params.items() if k not in ("key", "token")}
    return _build_url(path, clean)


async def _get_json(path: str) -> list:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{API_BASE}{path}", headers=_headers())
        r.raise_for_status()
        return r.json()


async def _fetch_binary(url: str, timeout: float = TIMEOUT) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(url, headers=_headers(), follow_redirects=True)
        r.raise_for_status()
        return r.content, r.headers.get("content-type", "application/octet-stream")


async def _post_json(path: str, body: dict, timeout: float = TIMEOUT) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(
            f"{API_BASE}{path}",
            json=body,
            headers={**_headers(), "Content-Type": "application/json"},
        )
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# AUTH TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
async def set_api_key(key: str) -> dict:
    """
    Set your Pollinations API key for this session.
    Prefer setting POLLINATIONS_API_KEY in Horizon env vars for persistence.
    pk_ = publishable (rate-limited), sk_ = secret (no limits).
    Get yours at https://enter.pollinations.ai
    """
    global _api_key
    if not key.startswith(("pk_", "sk_")):
        raise ToolError("Key must start with pk_ (publishable) or sk_ (secret)")
    _api_key = key
    kind = "publishable" if key.startswith("pk_") else "secret"
    masked = f"{key[:3]}...{key[-6:]}"
    return {"success": True, "keyType": kind, "maskedKey": masked}


@mcp.tool()
async def get_key_info() -> dict:
    """Check the status of the currently loaded API key."""
    k = _key()
    if not k:
        return {"authenticated": False, "message": "No key set. Call setApiKey or set POLLINATIONS_API_KEY env var."}
    kind = "publishable" if k.startswith("pk_") else "secret" if k.startswith("sk_") else "unknown"
    return {"authenticated": True, "keyType": kind, "maskedKey": f"{k[:3]}...{k[-6:]}"}


@mcp.tool()
async def clear_api_key() -> dict:
    """Clear the in-memory API key (does not affect env var)."""
    global _api_key
    had = bool(_api_key)
    _api_key = ""
    return {"success": True, "message": "Key cleared" if had else "No key was set"}


# ---------------------------------------------------------------------------
# IMAGE TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_image_url(
    prompt: str,
    model: str = "flux",
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    enhance: Optional[bool] = None,
    negative_prompt: Optional[str] = None,
    guidance_scale: Optional[float] = None,
    quality: Optional[str] = None,
    image: Optional[str] = None,
    transparent: Optional[bool] = None,
    nologo: Optional[bool] = None,
    nofeed: Optional[bool] = None,
    safe: Optional[bool] = None,
    private: Optional[bool] = None,
) -> dict:
    """
    Generate an image URL from a text prompt.
    Returns a shareable URL — no binary download.
    Models: flux (fast), turbo (ultra-fast), gptimage, kontext, seedream,
            nanobanana (Gemini), zimage. Use listImageModels for the full list.
    quality: low | medium | high | hd
    image: URL for image-to-image (kontext, seedream, nanobanana support this)
    """
    _require_key()
    params = _image_params(model, width, height, seed, enhance, negative_prompt,
                           guidance_scale, quality, image, transparent, nologo, nofeed, safe, private)
    is_img2img = image is not None
    encoded = quote(prompt, safe="")
    if is_img2img:
        # Use legacy image.pollinations.ai endpoint — more reliable for img2img
        url = f"{IMAGE_BASE}/prompt/{encoded}"
    else:
        url = f"{API_BASE}/image/{encoded}"
    if params:
        url += "?" + urlencode({k: v for k, v in params.items() if v is not None})
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
    guidance_scale: Optional[float] = None,
    quality: Optional[str] = None,
    image: Optional[str] = None,
    transparent: Optional[bool] = None,
    nologo: Optional[bool] = None,
    nofeed: Optional[bool] = None,
    safe: Optional[bool] = None,
    private: Optional[bool] = None,
) -> dict:
    """
    Generate an image and return base64-encoded data.
    Full parameter control. Supports image-to-image with kontext/seedream/nanobanana.
    quality: low | medium | high | hd
    """
    _require_key()
    is_img2img = image is not None
    params = _image_params(model, width, height, seed, enhance, negative_prompt,
                           guidance_scale, quality, image, transparent, nologo, nofeed, safe, private)
    encoded = quote(prompt, safe="")
    if is_img2img:
        # Use legacy endpoint — more reliable, and increase timeout significantly
        fetch_url = f"{IMAGE_BASE}/prompt/{encoded}"
        timeout = TIMEOUT_IMG2IMG
    else:
        fetch_url = f"{API_BASE}/image/{encoded}"
        timeout = TIMEOUT
    if params:
        fetch_url += "?" + urlencode({k: v for k, v in params.items() if v is not None})
    data, content_type = await _fetch_binary(fetch_url, timeout=timeout)
    b64 = base64.b64encode(data).decode()
    return {
        "base64": b64,
        "mimeType": content_type,
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
    }


@mcp.tool()
async def generate_image_batch(
    prompts: list[str],
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
    Generate up to 10 images in parallel from a list of prompts.
    Best with sk_ keys (no rate limits). Returns base64 for each image.
    """
    _require_key()
    if not prompts or len(prompts) > 10:
        raise ToolError("Provide 1–10 prompts")

    import asyncio

    async def _gen_one(prompt: str, idx: int):
        params = _image_params(model, width, height,
                               (seed + idx) if seed is not None else None,
                               enhance, negative_prompt, None, quality,
                               None, None, nologo, nofeed, safe, private)
        encoded = quote(prompt, safe="")
        url = _build_url(f"/image/{encoded}", params)
        data, ct = await _fetch_binary(url)
        return {"index": idx, "prompt": prompt, "base64": base64.b64encode(data).decode(), "mimeType": ct}

    results = await asyncio.gather(*[_gen_one(p, i) for i, p in enumerate(prompts)], return_exceptions=True)
    successful, failed = [], []
    for r in results:
        if isinstance(r, Exception):
            failed.append(str(r))
        else:
            successful.append(r)
    return {"total": len(prompts), "successful": len(successful), "failed": len(failed), "images": successful, "errors": failed}


@mcp.tool()
async def generate_video(
    prompt: str,
    model: str = "veo",
    duration: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    audio: Optional[bool] = None,
    image: Optional[str] = None,
    seed: Optional[int] = None,
    nologo: Optional[bool] = None,
    nofeed: Optional[bool] = None,
    safe: Optional[bool] = None,
    private: Optional[bool] = None,
) -> dict:
    """
    Generate a video from a text prompt. Returns base64 mp4.
    Models: veo (4/6/8s, supports audio), seedance (2-10s), seedance-pro (best quality)
    duration: veo=4/6/8, seedance=2-10
    audio: veo only
    image: reference image URL for image-to-video
    aspect_ratio: '16:9' | '9:16' | '1:1'
    """
    _require_key()
    params = {k: v for k, v in {
        "model": model, "duration": duration, "aspectRatio": aspect_ratio,
        "audio": audio, "image": image, "seed": seed,
        "nologo": nologo, "nofeed": nofeed, "safe": safe, "private": private,
    }.items() if v is not None}
    encoded = quote(prompt, safe="")
    url = _build_url(f"/image/{encoded}", params)
    data, content_type = await _fetch_binary(url, timeout=TIMEOUT_VIDEO)
    b64 = base64.b64encode(data).decode()
    return {"base64": b64, "mimeType": content_type or "video/mp4", "prompt": prompt, "model": model, "duration": duration}


@mcp.tool()
async def generate_video_url(
    prompt: str,
    model: str = "veo",
    duration: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    audio: Optional[bool] = None,
    image: Optional[str] = None,
    seed: Optional[int] = None,
    nologo: Optional[bool] = None,
    nofeed: Optional[bool] = None,
    safe: Optional[bool] = None,
    private: Optional[bool] = None,
) -> dict:
    """
    Generate a shareable video URL (no binary download).
    Models: veo, seedance, seedance-pro
    """
    _require_key()
    params = {k: v for k, v in {
        "model": model, "duration": duration, "aspectRatio": aspect_ratio,
        "audio": audio, "image": image, "seed": seed,
        "nologo": nologo, "nofeed": nofeed, "safe": safe, "private": private,
    }.items() if v is not None}
    encoded = quote(prompt, safe="")
    url = _shareable_url(f"/image/{encoded}", params)
    return {"videoUrl": url, "prompt": prompt, "model": model, "duration": duration}


@mcp.tool()
async def describe_image(
    image_url: str,
    prompt: str = "Describe this image in detail.",
    model: str = "openai",
) -> dict:
    """
    Analyze / describe an image using a vision AI model.
    Great for captioning, OCR, object detection, visual Q&A.
    model: openai | gemini | claude | grok
    """
    _require_key()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
    }
    result = await _post_json("/v1/chat/completions", body)
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"description": content, "imageUrl": image_url, "model": model}


@mcp.tool()
async def analyze_video(
    video_url: str,
    prompt: str = "Describe what happens in this video in detail.",
    model: str = "gemini-large",
) -> dict:
    """
    Analyze a video (YouTube or direct URL) using AI.
    Uses gemini-large for native video + audio understanding.
    """
    _require_key()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": video_url}},
        ]}],
    }
    result = await _post_json("/v1/chat/completions", body)
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"analysis": content, "videoUrl": video_url, "model": model}


@mcp.tool()
async def list_image_models() -> dict:
    """
    List all available image and video generation models with their capabilities.
    Fetched live from the API — always up to date.
    """
    models = await _get_json("/image/models")
    image_models = [m for m in models if "video" not in (m.get("output_modalities") or [])]
    video_models = [m for m in models if "video" in (m.get("output_modalities") or [])]
    return {
        "imageModels": [{"name": m["name"], "description": m.get("description"), "aliases": m.get("aliases", [])} for m in image_models],
        "videoModels": [{"name": m["name"], "description": m.get("description"), "aliases": m.get("aliases", [])} for m in video_models],
        "total": len(models),
    }


# ---------------------------------------------------------------------------
# TEXT TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_text(
    prompt: str,
    model: str = "openai",
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    json_mode: Optional[bool] = None,
    private: Optional[bool] = None,
) -> str:
    """
    Simple text generation from a prompt.
    Models: openai, openai-fast, openai-large, claude, claude-large,
            gemini, gemini-large, deepseek, grok, mistral, qwen-coder, and more.
    Use listTextModels for the full list.
    """
    _require_key()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body: dict = {"model": model, "messages": messages, "stream": False}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if seed is not None:
        body["seed"] = seed
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    result = await _post_json("/v1/chat/completions", body)
    return result.get("choices", [{}])[0].get("message", {}).get("content", "")


@mcp.tool()
async def chat_completion(
    messages: list[dict],
    model: str = "openai",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    response_format: Optional[dict] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    modalities: Optional[list[str]] = None,
    audio: Optional[dict] = None,
) -> dict:
    """
    Full OpenAI-compatible chat completions.
    Supports multi-turn, function/tool calling, reasoning, audio output, JSON mode.
    messages: [{"role": "user"|"assistant"|"system", "content": "..."}]
    reasoning_effort: low | medium | high  (for reasoning models)
    modalities: ["text"] or ["text", "audio"] for voice output
    """
    _require_key()
    body: dict = {"model": model, "messages": messages, "stream": False}
    optional = {
        "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p,
        "frequency_penalty": frequency_penalty, "presence_penalty": presence_penalty,
        "seed": seed, "response_format": response_format, "tools": tools,
        "tool_choice": tool_choice, "reasoning_effort": reasoning_effort,
        "modalities": modalities, "audio": audio,
    }
    body.update({k: v for k, v in optional.items() if v is not None})
    result = await _post_json("/v1/chat/completions", body)
    choice = result.get("choices", [{}])[0]
    msg = choice.get("message", {})
    out = {
        "content": msg.get("content", ""),
        "model": result.get("model", model),
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage"),
    }
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    if msg.get("reasoning_content"):
        out["reasoning"] = msg["reasoning_content"]
    if result.get("citations"):
        out["citations"] = result["citations"]
    return out


@mcp.tool()
async def list_text_models() -> dict:
    """
    List all available text generation models with capabilities.
    Shows reasoning, vision, audio, and tool-calling support.
    Fetched live from the API.
    """
    models = await _get_json("/text/models")
    return {
        "models": [
            {
                "name": m["name"],
                "description": m.get("description"),
                "aliases": m.get("aliases", []),
                "reasoning": m.get("reasoning", False),
                "tools": m.get("tools", False),
                "vision": m.get("vision", False),
            }
            for m in models
        ],
        "total": len(models),
    }


@mcp.tool()
async def web_search(
    query: str,
    model: str = "perplexity-fast",
    detailed: bool = False,
) -> dict:
    """
    Search the web for real-time information with citations.
    model: perplexity-fast | perplexity-reasoning | gemini-search
    detailed: True for comprehensive answer with more sources.
    """
    _require_key()
    system = (
        "Search the web and provide a comprehensive answer with sources. Cite your sources."
        if detailed else
        "Search the web and provide a concise, accurate answer. Include source URLs."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
    }
    result = await _post_json("/v1/chat/completions", body)
    return {
        "answer": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
        "query": query,
        "model": result.get("model", model),
        "citations": result.get("citations", []),
    }


@mcp.tool()
async def get_pricing(type: str = "all") -> dict:
    """
    Get pricing info for all models. type: all | text | image
    Prices in pollen (1 pollen = $0.001 USD).
    """
    result: dict = {"currency": "pollen", "note": "1 pollen = $0.001 USD"}
    if type in ("all", "text"):
        models = await _get_json("/text/models")
        result["textModels"] = [
            {"name": m["name"], "pricing": m.get("pricing")}
            for m in models if m.get("pricing")
        ]
    if type in ("all", "image"):
        models = await _get_json("/image/models")
        result["imageModels"] = [
            {"name": m["name"], "pricing": m.get("pricing")}
            for m in models if m.get("pricing")
        ]
    return result


# ---------------------------------------------------------------------------
# AUDIO TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
async def respond_audio(
    prompt: str,
    voice: str = "alloy",
    format: str = "mp3",
    voice_instructions: Optional[str] = None,
) -> dict:
    """
    Generate an audio response to a text prompt (AI speaks back to you).
    voice: alloy | echo | fable | onyx | nova | shimmer | coral | verse | ballad | ash | sage
    format: mp3 | wav | flac | opus | pcm16
    voice_instructions: e.g. 'Speak with enthusiasm and energy'
    Returns base64-encoded audio.
    """
    _require_key()
    full_prompt = f"{voice_instructions}\n\n{prompt}" if voice_instructions else prompt
    params = {"model": "openai-audio", "voice": voice, "format": format}
    url = _build_url(f"/text/{quote(full_prompt, safe='')}", params)
    data, content_type = await _fetch_binary(url)
    return {
        "base64": base64.b64encode(data).decode(),
        "mimeType": content_type or f"audio/{format}",
        "prompt": prompt,
        "voice": voice,
        "format": format,
    }


@mcp.tool()
async def say_text(
    text: str,
    voice: str = "alloy",
    format: str = "mp3",
    voice_instructions: Optional[str] = None,
) -> dict:
    """
    Convert text to speech verbatim (TTS).
    voice: alloy | echo | fable | onyx | nova | shimmer | coral | verse | ballad | ash | sage
    format: mp3 | wav | flac | opus | pcm16
    Returns base64-encoded audio.
    """
    _require_key()
    full_prompt = f"{voice_instructions}\n\nSay verbatim: {text}" if voice_instructions else f"Say verbatim: {text}"
    params = {"model": "openai-audio", "voice": voice, "format": format}
    url = _build_url(f"/text/{quote(full_prompt, safe='')}", params)
    data, content_type = await _fetch_binary(url)
    return {
        "base64": base64.b64encode(data).decode(),
        "mimeType": content_type or f"audio/{format}",
        "text": text,
        "voice": voice,
        "format": format,
    }


@mcp.tool()
async def list_audio_voices() -> dict:
    """
    List all available TTS voices and audio formats.
    Fetched live from the API.
    """
    try:
        models = await _get_json("/audio/models")
        voices: list = []
        by_model = []
        for m in models:
            if m.get("voices"):
                by_model.append({"model": m["name"], "voices": m["voices"]})
                for v in m["voices"]:
                    if v not in voices:
                        voices.append(v)
        return {"voices": voices, "byModel": by_model, "formats": ["mp3", "wav", "flac", "opus", "pcm16"], "total": len(voices)}
    except Exception:
        fallback = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "verse", "ballad", "ash", "sage"]
        return {"voices": fallback, "formats": ["mp3", "wav", "flac", "opus", "pcm16"], "total": len(fallback), "note": "fallback list"}


@mcp.tool()
async def transcribe_audio(
    audio_url: str,
    prompt: str = "Transcribe this audio accurately. Include timestamps if multiple speakers.",
    model: str = "gemini-large",
) -> dict:
    """
    Transcribe audio from a URL using AI (speech-to-text).
    model: gemini-large | gemini | openai-audio
    Supports most audio formats.
    """
    _require_key()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"url": audio_url}},
        ]}],
    }
    result = await _post_json("/v1/chat/completions", body)
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"transcription": content, "audioUrl": audio_url, "model": model}


# ---------------------------------------------------------------------------
# ACCOUNT TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_balance() -> dict:
    """
    Get your current Pollen balance.
    Requires an API key with account:usage permission.
    """
    _require_key()
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{API_BASE}/account/balance", headers=_headers())
        r.raise_for_status()
        data = r.json()
    return {"pollen": data.get("balance"), "note": "1 pollen = $0.001 USD"}


@mcp.tool()
async def get_usage(
    daily: bool = False,
    days: Optional[int] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Get API usage history.
    daily=True for daily aggregated summary grouped by date and model.
    daily=False (default) for per-request rows.
    days: time window 1–90 (default: 30 per-request, 90 daily)
    limit: max rows to return when daily=False (default: 100)
    Requires account:usage permission.
    """
    _require_key()
    path = "/account/usage/daily" if daily else "/account/usage"
    params: dict = {}
    if days is not None:
        params["days"] = days
    if not daily and limit is not None:
        params["limit"] = limit
    url = f"{API_BASE}{path}" + (("?" + urlencode(params)) if params else "")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url, headers=_headers())
        r.raise_for_status()
        data = r.json()
    return {"mode": "daily" if daily else "per-request", "records": data.get("usage", []), "count": data.get("count", 0)}


# ---------------------------------------------------------------------------
# Entry point (ignored by Horizon, used for local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    app = mcp.get_asgi_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
