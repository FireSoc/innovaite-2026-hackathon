"""LLM gateway — CommonStack (Anthropic-style) or Gemini."""

import base64
import json
import logging
from typing import Any, Type, TypeVar

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# CommonStack vision supports these MIME types; PDFs are not sent as image parts.
COMMONSTACK_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _get_client() -> genai.Client:
    """Create a Gemini client using the configured API key."""
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)


def _build_commonstack_content(
    prompt: str,
    images: list[tuple[bytes, str]] | None = None,
) -> list[dict[str, Any]]:
    """Build user message content: text part + optional base64 image parts (images only)."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if images:
        for img_bytes, mime_type in images:
            if mime_type in COMMONSTACK_IMAGE_MIME_TYPES:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64.standard_b64encode(img_bytes).decode("ascii"),
                    },
                })
    return content


def _extract_text_from_commonstack_response(body: dict) -> str:
    """Get assistant text from CommonStack Messages API response."""
    content = body.get("content") or []
    for block in content:
        if block.get("type") == "text" and "text" in block:
            return block["text"]
    raise ValueError("No text content block in CommonStack response")


async def _commonstack_complete_json(
    schema: Type[T],
    prompt: str,
    images: list[tuple[bytes, str]] | None = None,
    max_retries: int = 1,
) -> T:
    """Call CommonStack Messages API with JSON schema; return validated Pydantic model."""
    settings = get_settings()
    url = settings.commonstack_base_url.rstrip("/") + "/messages"
    headers = {
        "Authorization": f"Bearer {settings.commonstack_api_key}",
        "Content-Type": "application/json",
    }
    json_schema = schema.model_json_schema()
    last_error: Exception | None = None

    for attempt in range(1 + max_retries):
        try:
            retry_prompt = prompt
            if attempt > 0 and last_error:
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: Your previous response failed validation with this error:\n"
                    f"{str(last_error)}\n"
                    f"Please fix the JSON output to conform to the schema."
                )
            content = _build_commonstack_content(retry_prompt, images)
            payload: dict[str, Any] = {
                "model": settings.commonstack_model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": content}],
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": json_schema,
                    },
                },
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=payload)
            if response.status_code >= 400:
                logger.error(
                    "CommonStack API error: status=%s body=%s",
                    response.status_code,
                    response.text[:500],
                )
                raise ValueError(
                    f"CommonStack API error: {response.status_code} - {response.text[:200]}"
                )
            raw_text = _extract_text_from_commonstack_response(response.json())
            if not raw_text:
                raise ValueError("Empty text in CommonStack response")
            parsed = json.loads(raw_text)
            return schema.model_validate(parsed)
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(
                "CommonStack JSON response validation failed (attempt %s): %s",
                attempt + 1,
                e,
            )
            continue
    raise last_error  # type: ignore[misc]


async def _commonstack_complete_text(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str:
    """Call CommonStack Messages API for plain text completion."""
    settings = get_settings()
    url = settings.commonstack_base_url.rstrip("/") + "/messages"
    headers = {
        "Authorization": f"Bearer {settings.commonstack_api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": settings.commonstack_model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
    if response.status_code >= 400:
        logger.error(
            "CommonStack API error: status=%s body=%s",
            response.status_code,
            response.text[:500],
        )
        raise ValueError(
            f"CommonStack API error: {response.status_code} - {response.text[:200]}"
        )
    text = _extract_text_from_commonstack_response(response.json())
    return text or ""


async def complete_json(
    schema: Type[T],
    prompt: str,
    images: list[tuple[bytes, str]] | None = None,
    max_retries: int = 1,
) -> T:
    """
    Send a prompt (with optional images) to the configured LLM and parse the response
    into a Pydantic model. Uses CommonStack if llm_provider is "commonstack" and key
    is set; otherwise Gemini.
    """
    settings = get_settings()
    if settings.llm_provider == "commonstack" and settings.commonstack_api_key:
        return await _commonstack_complete_json(
            schema=schema,
            prompt=prompt,
            images=images,
            max_retries=max_retries,
        )

    # Gemini path
    client = _get_client()
    contents: list[types.Part | str] = [prompt]
    if images:
        for img_bytes, mime_type in images:
            contents.append(
                types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
            )
    json_schema = schema.model_json_schema()
    last_error: Exception | None = None

    for attempt in range(1 + max_retries):
        try:
            retry_prompt = prompt
            if attempt > 0 and last_error:
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: Your previous response failed validation with this error:\n"
                    f"{str(last_error)}\n"
                    f"Please fix the JSON output to conform to the schema."
                )
                contents[0] = retry_prompt

            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=json_schema,
                    temperature=0.1,
                ),
            )
            raw_text = response.text
            if not raw_text:
                raise ValueError("Empty response from Gemini")
            parsed = json.loads(raw_text)
            result = schema.model_validate(parsed)
            return result
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(
                f"Gemini response validation failed (attempt {attempt + 1}): {e}"
            )
            continue
    raise last_error  # type: ignore[misc]


async def complete_text(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str:
    """
    Simple text completion (used for letter hardship paragraphs, etc.).
    Uses CommonStack if llm_provider is "commonstack" and key is set; otherwise Gemini.
    """
    settings = get_settings()
    if settings.llm_provider == "commonstack" and settings.commonstack_api_key:
        return await _commonstack_complete_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    client = _get_client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text or ""
