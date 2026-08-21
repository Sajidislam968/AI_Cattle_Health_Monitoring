"""Small, explicit client for the local FastAPI service."""

from __future__ import annotations

from typing import Any

import requests


class APIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = requests.request(method, f"{self.base_url}{path}", timeout=30, **kwargs)
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(str(detail))
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/")

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/get-status")

    def final_results(self) -> dict[str, Any]:
        return self._request("GET", "/get-final-results")

    def summary(self) -> dict[str, Any]:
        return self._request("GET", "/get-summary")

    def upload_video(self, filename: str, content: bytes) -> dict[str, Any]:
        return self._request(
            "POST",
            "/upload-video",
            files={"video": (filename, content, "video/mp4")},
        )

    def start_pipeline(self) -> dict[str, Any]:
        return self._request("POST", "/run-pipeline")

    def processed_video(self) -> bytes:
        response = requests.get(f"{self.base_url}/get-video", timeout=30)
        if not response.ok:
            raise RuntimeError("Processed video is not available yet.")
        return response.content
