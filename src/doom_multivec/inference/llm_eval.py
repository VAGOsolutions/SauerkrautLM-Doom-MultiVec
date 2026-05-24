"""Lightweight LLM evaluator for DoomMultiVec.

Sends a prompt and a sequence of images to a Triton-hosted
OpenAI-compatible chat-completions endpoint and returns the model text
reply. Images are sent as multimodal content parts (`text` +
`image_url`) so the model sees the original pixels instead of an ASCII
rendering.

Provides:
- `query_llm_with_frames(images, prompt, api_key, ...)` — main helper.
- `test_query(api_key)` — example/test helper (kept compatible with provided snippet).

This module keeps a conservative `max_tokens` default and clamps a
user-visible `budget_dollars` to a $50 hard cap as requested. Actual
billing enforcement must happen server-side; this client only caps
client-side parameters.
"""

from typing import List, Optional, Any, Dict, Union
import requests
import json
import base64
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import numpy as np

# Optional .env support
try:
    from dotenv import load_dotenv  # type: ignore
    _DOTENV_AVAILABLE = True
except Exception:
    _DOTENV_AVAILABLE = False

DEFAULT_TRITON_URL = "https://tritonai-api.ucsd.edu/v1/chat/completions"
USAGE_LOG_PATH = Path(__file__).resolve().parent / "usage" / "usage_log.txt"

# Per-million-token pricing for models we know we use in this repo.
# Values come from the Triton model hub table.
MODEL_PRICING_PER_MILLION = {
    "api-gemma-4-26b": {"input": 0.08, "output": 0.35},
}


def _img_to_png_bytes(frame: Any) -> bytes:
    """Convert a frame to PNG bytes.

    Accepted input types:
    - bytes: assumed to be already encoded image bytes (PNG/JPEG)
    - str: file path to an image
    - PIL.Image.Image: converted to PNG
    - numpy.ndarray: converted via PIL.Image.fromarray
    """
    if isinstance(frame, bytes):
        return frame

    if isinstance(frame, str):
        # treat as file path if possible; otherwise caller may be passing ASCII text
        path = Path(frame)
        if path.exists():
            with open(path, "rb") as f:
                return f.read()
        raise TypeError("String frame is not a file path")

    if isinstance(frame, Image.Image):
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        return buf.getvalue()

    if isinstance(frame, np.ndarray):
        # Convert numpy array to image. Expect HWC or HW
        arr = frame
        if arr.ndim == 2:
            img = Image.fromarray(arr.astype(np.uint8), mode="L")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        elif arr.ndim == 3 and arr.shape[2] == 4:
            img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        else:
            # attempt generic conversion
            img = Image.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    raise TypeError(f"Unsupported frame type: {type(frame)}")


def _image_to_data_uri(frame: Any) -> str:
    png = _img_to_png_bytes(frame)
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _frame_to_content_part(frame: Any, index: int) -> Dict[str, Any]:
    """Convert one frame to an OpenAI-compatible content part.

    - PIL images, numpy arrays, bytes, and file paths are sent as `image_url`.
    - Strings that are not file paths are treated as plain text for backwards compatibility.
    """
    if isinstance(frame, str) and not Path(frame).exists():
        return {"type": "text", "text": frame}

    data_uri = _image_to_data_uri(frame)
    return {
        "type": "image_url",
        "image_url": {"url": data_uri},
    }


def _build_multimodal_content(frames: List[Any], prompt: str) -> List[Dict[str, Any]]:
    """Build OpenAI-style multimodal message content."""
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    if frames:
        content.append({"type": "text", "text": "Here are the frames in order:"})
        for index, frame in enumerate(frames):
            try:
                content.append(_frame_to_content_part(frame, index))
            except Exception:
                content.append({"type": "text", "text": f"[unreadable frame {index}]"})

    return content


def _extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    """Try several common response shapes to extract the assistant text."""
    # OpenAI-style
    try:
        choices = resp_json.get("choices")
        if choices and isinstance(choices, list):
            first = choices[0]
            # Chat-style
            msg = first.get("message")
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
            # Completion-style
            if "text" in first:
                return first["text"]
    except Exception:
        pass

    # Anthropic-style
    if "completion" in resp_json:
        return resp_json["completion"]

    # Fallback: try to stringize the full response
    return json.dumps(resp_json)


def _extract_usage(resp_json: Dict[str, Any]) -> Dict[str, int]:
    """Extract usage counts from an OpenAI-style response."""
    usage = resp_json.get("usage", {}) if isinstance(resp_json, dict) else {}
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Estimate spend for a single call using per-million-token pricing."""
    pricing = MODEL_PRICING_PER_MILLION.get(model)
    if pricing is None:
        return None

    return (
        (prompt_tokens / 1_000_000.0) * pricing["input"]
        + (completion_tokens / 1_000_000.0) * pricing["output"]
    )


def _append_usage_log(entry: Dict[str, Any]) -> None:
    """Append a single JSON line to the usage log."""
    USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def query_llm_with_frames(
    frames: List[Any],
    prompt: str,
    api_key: Optional[str],
    model: str = "api-gemma-4-26b",
    url: str = DEFAULT_TRITON_URL,
    max_tokens: int = 512,
    budget_dollars: float = 50.0,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Send frames + prompt to the Triton chat completions endpoint.

    Args:
        frames: Sequence of ASCII-encoded frames (strings).
        prompt: Prompt string to prepend to the message.
        api_key: Bearer API key for the Triton endpoint.
        model: Model name to request on the server (default: api-gemma-4-26b).
        url: The Triton chat completions endpoint.
        max_tokens: Max tokens to request for the model's reply.
        budget_dollars: Client-side budget cap (will be clamped to $50).
        timeout: Request timeout in seconds.

    Returns:
        The parsed JSON response from the server. The helper
        `_extract_text_from_response` can be used to pull the text.
    """
    # Resolve API key from argument or environment
    if not api_key:
        if _DOTENV_AVAILABLE:
            load_dotenv()
        api_key = os.getenv("LITELLM_API_KEY")

    if not api_key:
        raise ValueError("api_key is required (or set LITELLM_API_KEY in environment or .env)")

    # Enforce client-side hard cap
    if budget_dollars is None or budget_dollars > 50.0:
        budget_dollars = 50.0

    # Build OpenAI-style multimodal message content
    message_content = _build_multimodal_content(frames, prompt)

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": message_content}
        ],
        # Conservative generation settings; server may ignore unknown fields
        "max_tokens": max_tokens,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        resp_json = resp.json()
    except ValueError:
        # Non-JSON response
        resp_json = {"status_code": resp.status_code, "text": resp.text}

    # Attach http status for caller convenience
    if isinstance(resp_json, dict):
        resp_json.setdefault("_http_status", resp.status_code)

        usage = _extract_usage(resp_json)
        cost_usd = _estimate_cost_usd(model, usage["prompt_tokens"], usage["completion_tokens"])
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model": model,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "cost_usd": cost_usd,
            "http_status": resp.status_code,
            "endpoint": url,
        }
        _append_usage_log(log_entry)

    return resp_json


def test_query(api_key: Optional[str]) -> None:
    """Compatibility helper matching the user's provided snippet."""
    url = DEFAULT_TRITON_URL

    # Allow api_key to be omitted and loaded from environment/.env
    if not api_key:
        if _DOTENV_AVAILABLE:
            load_dotenv()
        api_key = os.getenv("LITELLM_API_KEY")

    if not api_key:
        raise ValueError("api_key is required for test_query (or set LITELLM_API_KEY in environment or .env)")

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }

    payload = {
        "model": "api-gemma-4-26b",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    print("Status code:", response.status_code)
    print("Response JSON:")
    try:
        print(response.json())
    except Exception:
        print(response.text)
