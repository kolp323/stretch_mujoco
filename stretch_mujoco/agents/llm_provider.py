"""HTTP provider for event-driven OpenAI-compatible LLM requests."""

from __future__ import annotations

import json
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .actions import ActionType
from .llm import LLMRequest
from .llm_config import LLMProviderConfig


class LLMProviderError(RuntimeError):
    """Raised when a configured LLM request or response is invalid."""


Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class OpenAICompatibleProvider:
    """Convert queued office events into validated JSON response candidates."""

    def __init__(
        self,
        config: LLMProviderConfig,
        transport: Transport | None = None,
    ) -> None:
        if not config.enabled:
            raise LLMProviderError("LLM provider is disabled in the local config")
        self.config = config
        self._transport = transport or self._http_transport

    def __call__(self, request: LLMRequest) -> dict[str, Any]:
        url, payload = self._build_request(request)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        if self.config.project:
            headers["OpenAI-Project"] = self.config.project

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                raw_response = self._transport(
                    url,
                    headers,
                    payload,
                    self.config.timeout_seconds,
                )
                return self._parse_response(raw_response)
            except (HTTPError, URLError, TimeoutError, LLMProviderError) as error:
                last_error = error
                if attempt >= self.config.max_retries:
                    break
                time.sleep(min(0.25 * (2**attempt), 2.0))
        raise LLMProviderError(f"LLM request failed: {last_error}") from last_error

    def _build_request(self, request: LLMRequest) -> tuple[str, dict[str, Any]]:
        prompt = self._prompt(request)
        if self.config.api_mode == "responses":
            return (
                f"{self.config.base_url}/responses",
                {
                    "model": self.config.model,
                    "input": prompt,
                },
            )
        return (
            f"{self.config.base_url}/chat/completions",
            {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return one JSON object only. Do not invent action names, "
                            "objects, locations, or permissions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            },
        )

    @staticmethod
    def _prompt(request: LLMRequest) -> str:
        allowed_actions = ", ".join(action.value for action in ActionType)
        if request.trigger.value == "day_start":
            contract = (
                'Return exactly {"schedule": [{"id": "...", "start": "HH:MM", '
                '"end": "HH:MM", "activity": "work|rest|meeting", '
                '"location": "valid_object_id", "variation_minutes": 0}]}. '
                "Cover the 09:00-18:00 workday with realistic non-overlapping items. "
                "Use only object IDs from valid_objects. Keep rest periods brief and include "
                "a lunch/rest window."
            )
        elif request.trigger.value == "dialogue":
            allowed_intents = request.context.get("allowed_intents", [])
            contract = (
                'Return exactly {"dialogue": {"session_id": "...", "turn_id": "...", '
                '"speaker": "...", "listener": "...", "act": "acknowledge", '
                '"text": "brief office reply"}}. The IDs must echo context; do not add actions. '
                "The runtime will filter sensitive content, constrain length, and may replace text. "
                f"Allowed intents: {allowed_intents}."
            )
        elif request.trigger.value == "new_task":
            contract = (
                'Return {"action": {"action": "allowed_action", '
                '"target": "valid_object_id", "parameters": {}}}. The action must '
                "directly address context.target. For robot delivery, use action "
                '"request_robot", target "stretch_3", parameters.task "deliver", '
                "parameters.object equal to context.target, and a valid destination."
            )
        else:
            contract = (
                'Return either {"action": {"action": "allowed_action", '
                '"target": "valid_object_id", "parameters": {}}} or '
                '{"dialogue": "brief explanation"}. For robot delivery, use action '
                '"request_robot", target "stretch_3", and parameters with task, object, '
                "and destination."
            )
        request_payload = {
            "trigger": request.trigger.value,
            "agent_id": request.agent_id,
            "day": request.day,
            "minute_of_day": request.minute_of_day,
            "context": request.context,
        }
        return (
            "You provide high-level office NPC guidance. The deterministic runtime "
            "will validate every result and execute all low-level behavior. Return one "
            "json object only, with no markdown. Do not invent objects or locations. "
            f"Response contract: {contract} Allowed actions: {allowed_actions}. Event: "
            f"{json.dumps(request_payload, ensure_ascii=False)}"
        )

    def _parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        if self.config.api_mode == "chat_completions":
            try:
                content = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise LLMProviderError("Chat response does not contain message content") from error
        else:
            content = response.get("output_text")
            if content is None:
                content = self._responses_output_text(response)
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("LLM response contains no text")
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise LLMProviderError("LLM response is not valid JSON") from error
        if not isinstance(payload, dict):
            raise LLMProviderError("LLM response must be a JSON object")
        return payload

    @staticmethod
    def _responses_output_text(response: dict[str, Any]) -> str | None:
        for output in response.get("output", []):
            for content in output.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return content["text"]
        return None

    @staticmethod
    def _http_transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            raise LLMProviderError(f"LLM HTTP error {error.code}") from error
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as error:
            raise LLMProviderError("LLM endpoint returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise LLMProviderError("LLM endpoint returned a non-object response")
        return parsed
