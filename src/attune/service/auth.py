"""Optional Basic Authentication for temporary Attune demo deployments."""

from __future__ import annotations

import base64
import binascii
import secrets
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BasicAuthCredentials:
    username: str
    password: str

    def __post_init__(self) -> None:
        if not self.username or not self.password:
            raise ValueError("demo username and password must both be non-empty")


class BasicAuthMiddleware:
    """Protect every HTTP and WebSocket route when credentials are configured."""

    def __init__(self, app: Any, *, credentials: BasicAuthCredentials) -> None:
        self.app = app
        self.credentials = credentials

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in {"http", "websocket"} or self._authorized(scope):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send(
                {"type": "websocket.close", "code": 1008, "reason": "authentication required"}
            )
            return
        error_body = b'{"detail":"authentication required"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(error_body)).encode()),
                    (b"www-authenticate", b'Basic realm="Attune Cadence", charset="UTF-8"'),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": error_body})

    def _authorized(self, scope: dict[str, Any]) -> bool:
        headers = dict(scope.get("headers", ()))
        authorization = headers.get(b"authorization", b"")
        try:
            scheme, encoded = authorization.split(b" ", 1)
            if scheme.lower() != b"basic":
                return False
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError, binascii.Error):
            return False
        username_matches = secrets.compare_digest(username, self.credentials.username)
        password_matches = secrets.compare_digest(password, self.credentials.password)
        return username_matches and password_matches
