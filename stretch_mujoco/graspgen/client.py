"""Lightweight client for persistent GraspGen RGB-D inference."""

from __future__ import annotations

from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from .protocol import control_request, encode_rgbd_request


msgpack_numpy.patch()


class RGBDGraspClient:
    def __init__(self, host: str, port: int = 5557, *, timeout_ms: int = 180_000) -> None:
        self.address = f"tcp://{host}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.address)
        self.last_request_bytes = 0

    def _request(self, frames: list[bytes]) -> dict[str, Any]:
        self.last_request_bytes = sum(map(len, frames))
        try:
            self.socket.send_multipart(frames)
            packed = self.socket.recv()
        except zmq.error.Again as exc:
            raise TimeoutError(f"Timed out waiting for GraspGen at {self.address}") from exc
        response = msgpack.unpackb(packed, raw=False)
        if response.get("status") == "error" or "error" in response:
            raise RuntimeError(f"GraspGen server error: {response.get('error')}")
        return response

    def health(self) -> dict[str, Any]:
        return self._request(control_request("health"))

    def metadata(self) -> dict[str, Any]:
        return self._request(control_request("metadata"))

    def infer(
        self,
        rgb: np.ndarray,
        depth_metres: np.ndarray,
        camera_k: np.ndarray,
        text_prompt: str,
        *,
        return_top_k: int = 10,
        num_grasps: int = 500,
        min_grasps: int = 20,
        max_tries: int = 6,
        candidate_top_k: int = 200,
        desired_approach_direction: tuple[float, float, float] | None = None,
        max_approach_angle_deg: float = 180.0,
        approach_allow_opposite: bool = True,
        min_depth: float = 0.05,
        max_depth: float = 3.0,
    ) -> dict[str, Any]:
        inference: dict[str, Any] = {
            "return_top_k": return_top_k,
            "candidate_top_k": candidate_top_k,
            "num_grasps": num_grasps,
            "min_grasps": min_grasps,
            "max_tries": max_tries,
            "remove_outliers": True,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "max_object_points": 20_000,
            "max_approach_angle_deg": max_approach_angle_deg,
            "approach_allow_opposite": approach_allow_opposite,
        }
        if desired_approach_direction is not None:
            inference["desired_approach_direction"] = desired_approach_direction
        return self._request(
            encode_rgbd_request(
                rgb,
                depth_metres,
                camera_k,
                text_prompt,
                inference=inference,
            )
        )

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None

    def __enter__(self) -> "RGBDGraspClient":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
