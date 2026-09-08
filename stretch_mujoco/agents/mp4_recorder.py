"""Readable MP4 recorder for multi-agent office simulations.

The recorder visualizes logical state (locations, actions, targets, events,
and robot tasks). It is not presented as a physics-camera recording of MuJoCo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

AGENT_COLORS = [(67, 134, 246), (79, 190, 110), (210, 123, 81), (174, 92, 193)]
STATUS_COLORS = {
    "work": (188, 129, 51),
    "use_computer": (180, 113, 47),
    "rest": (96, 168, 65),
    "eat": (83, 147, 218),
    "drink": (83, 147, 218),
    "attend_meeting": (174, 92, 193),
    "move_to": (105, 105, 105),
    "idle": (140, 140, 140),
    "request_robot": (62, 77, 205),
    "conversation": (182, 89, 190),
}
ACTION_LABELS = {
    "work": "Working",
    "use_computer": "Using computer",
    "rest": "Resting",
    "eat": "Eating",
    "drink": "Drinking",
    "attend_meeting": "In meeting",
    "move_to": "Walking",
    "sit": "Sitting",
    "pick_up": "Picking up item",
    "put_down": "Putting down item",
    "request_robot": "Requesting robot",
    "conversation": "In conversation",
    "idle": "Idle",
}
ZONE_COLORS = {
    "work": (238, 242, 246),
    "meeting": (239, 232, 247),
    "lounge": (232, 244, 235),
    "snack": (244, 238, 224),
}
ZONE_LABELS = {
    "work": "WORK AREA",
    "meeting": "MEETING",
    "lounge": "LOUNGE",
    "snack": "SNACK BAR",
}


class OfficeMp4Recorder:
    """Record an inspection-oriented 2-D office-day video."""

    def __init__(
        self,
        output_path: str | Path,
        scene_manifest: str | Path | None = None,
        *,
        width_m: float = 18.0,
        depth_m: float = 12.0,
        fps: int = 10,
        pixels_per_meter: int = 80,
        title: str = "OFFICE NPC DAY",
        subtitle: str = "Logical simulation playback",
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps
        self.title = title
        self.subtitle = subtitle
        self.pixels_per_meter = min(pixels_per_meter, 60)
        self._width_m = width_m
        self._depth_m = depth_m
        self._zones: list[dict[str, Any]] = []
        if scene_manifest is not None:
            self._load_zones(Path(scene_manifest))

        self._header_h = 82
        self._footer_h = 174
        self._map_margin = 32
        self._sidebar_w = 410
        self._layout_canvas()
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0
        self._base_canvas: np.ndarray | None = None

    def start(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (self.canvas_w, self.canvas_h)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for '{self.output_path}'")
        self._base_canvas = self._draw_floor_plan()

    def record_frame(
        self,
        minute_of_day: float,
        agents: dict[str, dict[str, Any]],
        *,
        events: list[str] | None = None,
        robot_tasks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Render a map, live state cards, robot tasks, and recent events."""
        if self._writer is None or self._base_canvas is None:
            raise RuntimeError("Call start() before record_frame()")
        canvas = self._base_canvas.copy()
        self._draw_header(canvas, minute_of_day, len(agents), robot_tasks or [])
        self._draw_agents(canvas, agents)
        self._draw_sidebar(canvas, agents, robot_tasks or [])
        self._draw_event_feed(canvas, events or [])
        self._writer.write(canvas)
        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _layout_canvas(self) -> None:
        self._map_w = int(self._width_m * self.pixels_per_meter)
        self._map_h = int(self._depth_m * self.pixels_per_meter)
        self._origin_x = self._map_margin
        self._origin_y = self._header_h + 16
        self._x_offset = self._width_m / 2
        self._y_offset = self._depth_m / 2
        self.canvas_w = self._map_w + self._map_margin * 2 + 20 + self._sidebar_w
        self.canvas_h = self._header_h + 16 + self._map_h + self._footer_h
        self._sidebar_x = self._origin_x + self._map_w + 20
        self._footer_y = self._origin_y + self._map_h + 18

    def _world_to_pixel(self, x_m: float, y_m: float) -> tuple[int, int]:
        return (
            int(self._origin_x + (x_m + self._x_offset) * self.pixels_per_meter),
            int(self._origin_y + (self._y_offset - y_m) * self.pixels_per_meter),
        )

    def _draw_floor_plan(self) -> np.ndarray:
        canvas = np.full((self.canvas_h, self.canvas_w, 3), (248, 249, 251), dtype=np.uint8)
        cv2.rectangle(
            canvas,
            (self._origin_x, self._origin_y),
            (self._origin_x + self._map_w, self._origin_y + self._map_h),
            (246, 244, 240),
            -1,
        )
        for zone in self._zones:
            xmin, xmax, ymin, ymax = zone["bounds"]
            px1, py1 = self._world_to_pixel(xmin, ymax)
            px2, py2 = self._world_to_pixel(xmax, ymin)
            zone_type = zone.get("type", "")
            cv2.rectangle(
                canvas, (px1, py1), (px2, py2), ZONE_COLORS.get(zone_type, (235, 235, 235)), -1
            )
            cx, cy = self._world_to_pixel((xmin + xmax) / 2, (ymin + ymax) / 2)
            self._text(
                canvas,
                ZONE_LABELS.get(zone_type, zone_type.upper()),
                (cx, cy),
                0.45,
                (150, 150, 150),
                center=True,
            )
        cv2.rectangle(
            canvas,
            (self._origin_x, self._origin_y),
            (self._origin_x + self._map_w, self._origin_y + self._map_h),
            (84, 91, 102),
            2,
        )
        self._draw_furniture(canvas)
        return canvas

    def _draw_header(
        self,
        canvas: np.ndarray,
        minute_of_day: float,
        agent_count: int,
        robot_tasks: list[dict[str, Any]],
    ) -> None:
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, self._header_h), (34, 43, 58), -1)
        self._text(canvas, self.title, (32, 34), 0.78, (255, 255, 255), thickness=2)
        self._text(canvas, self.subtitle, (32, 61), 0.43, (190, 201, 216))
        hour, minute = divmod(int(minute_of_day) % (24 * 60), 60)
        self._text(
            canvas,
            f"{hour:02d}:{minute:02d}",
            (self.canvas_w - 34, 38),
            0.9,
            (255, 255, 255),
            thickness=2,
            right=True,
        )
        active_tasks = sum(task.get("status") in {"pending", "running"} for task in robot_tasks)
        self._text(
            canvas,
            f"{agent_count} NPCs  |  Robot tasks: {active_tasks} active",
            (self.canvas_w - 34, 64),
            0.42,
            (190, 201, 216),
            right=True,
        )

    def _draw_agents(self, canvas: np.ndarray, agents: dict[str, dict[str, Any]]) -> None:
        for index, (agent_id, state) in enumerate(sorted(agents.items())):
            color = AGENT_COLORS[index % len(AGENT_COLORS)]
            px, py = self._world_to_pixel(*state.get("position", (0.0, 0.0))[:2])
            target_position = state.get("target_position")
            if target_position is not None:
                tx, ty = self._world_to_pixel(*target_position[:2])
                cv2.arrowedLine(canvas, (px, py), (tx, ty), color, 2, cv2.LINE_AA, tipLength=0.06)
                cv2.circle(canvas, (tx, ty), 8, color, 1, cv2.LINE_AA)
            action = state.get("action", "idle")
            status_color = STATUS_COLORS.get(action, STATUS_COLORS["idle"])
            cv2.circle(canvas, (px, py), 19, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 16, status_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 11, color, -1, cv2.LINE_AA)
            self._text(
                canvas,
                str(index + 1),
                (px, py + 5),
                0.42,
                (255, 255, 255),
                center=True,
                thickness=2,
            )
            self._text(
                canvas, state.get("label", agent_id), (px, py + 33), 0.39, (45, 52, 64), center=True
            )

    def _draw_sidebar(
        self,
        canvas: np.ndarray,
        agents: dict[str, dict[str, Any]],
        robot_tasks: list[dict[str, Any]],
    ) -> None:
        x, y = self._sidebar_x, self._origin_y
        cv2.rectangle(
            canvas, (x, y), (self.canvas_w - 24, self._footer_y - 14), (255, 255, 255), -1
        )
        cv2.rectangle(canvas, (x, y), (self.canvas_w - 24, self._footer_y - 14), (220, 224, 230), 1)
        self._text(canvas, "LIVE NPC STATUS", (x + 18, y + 30), 0.56, (42, 49, 61), thickness=2)
        y += 48
        for index, (agent_id, state) in enumerate(sorted(agents.items())):
            color = AGENT_COLORS[index % len(AGENT_COLORS)]
            card_h = 132
            cv2.rectangle(
                canvas, (x + 12, y), (self.canvas_w - 36, y + card_h), (247, 249, 252), -1
            )
            cv2.rectangle(canvas, (x + 12, y), (self.canvas_w - 36, y + card_h), (226, 230, 236), 1)
            cv2.rectangle(canvas, (x + 12, y), (x + 18, y + card_h), color, -1)
            name = state.get("display_name", agent_id.replace("_", " ").title())
            self._text(canvas, name, (x + 32, y + 27), 0.5, (35, 43, 54), thickness=2)
            self._text(canvas, state.get("role", ""), (x + 32, y + 49), 0.39, (101, 111, 126))
            action = state.get("action", "idle")
            action_label = ACTION_LABELS.get(action, action.replace("_", " ").title())
            self._chip(
                canvas,
                action_label,
                (x + 32, y + 63),
                STATUS_COLORS.get(action, STATUS_COLORS["idle"]),
            )
            location = state.get("location", "unknown").replace("_", " ")
            self._text(canvas, f"At: {location}", (x + 32, y + 100), 0.4, (67, 75, 88))
            target = state.get("target")
            detail = f"Target: {str(target).replace('_', ' ')}" if target else "No active target"
            self._text(canvas, detail, (x + 32, y + 120), 0.36, (101, 111, 126))
            y += card_h + 12
        self._text(canvas, "ROBOT TASKS", (x + 18, y + 28), 0.52, (42, 49, 61), thickness=2)
        y += 43
        if not robot_tasks:
            self._text(
                canvas, "No robot task in this interval", (x + 18, y + 16), 0.39, (110, 119, 133)
            )
            return
        for task in robot_tasks[-2:]:
            status = str(task.get("status", "unknown"))
            color = (
                (80, 169, 96)
                if status == "succeeded"
                else (62, 77, 205) if status in {"pending", "running"} else (70, 70, 210)
            )
            self._chip(canvas, status.upper(), (x + 18, y), color)
            object_id = str(task.get("object", "item")).replace("_", " ")
            destination = str(task.get("destination", "destination")).replace("_", " ")
            self._text(
                canvas, f"{object_id} -> {destination}", (x + 18, y + 35), 0.38, (67, 75, 88)
            )
            y += 54

    def _draw_event_feed(self, canvas: np.ndarray, events: list[str]) -> None:
        x, y = self._origin_x, self._footer_y
        cv2.rectangle(canvas, (x, y), (self.canvas_w - 24, self.canvas_h - 20), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x, y), (self.canvas_w - 24, self.canvas_h - 20), (220, 224, 230), 1)
        self._text(canvas, "RECENT EVENTS", (x + 16, y + 28), 0.5, (42, 49, 61), thickness=2)
        if not events:
            self._text(
                canvas,
                "Waiting for the next runtime event",
                (x + 16, y + 58),
                0.42,
                (101, 111, 126),
            )
            return
        line_y = y + 54
        for event in events[-4:]:
            self._text(canvas, event, (x + 18, line_y), 0.39, (67, 75, 88))
            line_y += 24

    def _draw_furniture(self, canvas: np.ndarray) -> None:
        furniture = {
            "DESK 1": (-3.0, -3.0),
            "DESK 2": (3.0, -3.0),
            "TABLE": (-5.5, 3.0),
            "SNACK": (6.8, 3.0),
            "STORAGE": (5.2, 5.5),
        }
        for label, (x, y) in furniture.items():
            px, py = self._world_to_pixel(x, y)
            cv2.rectangle(canvas, (px - 13, py - 9), (px + 13, py + 9), (155, 165, 180), -1)
            self._text(canvas, label, (px, py - 15), 0.3, (98, 107, 120), center=True)
        rx, ry = self._world_to_pixel(-7.7, -0.7)
        cv2.circle(canvas, (rx, ry), 10, (61, 75, 204), -1)
        self._text(canvas, "ROBOT", (rx, ry + 24), 0.32, (61, 75, 204), center=True, thickness=2)

    @staticmethod
    def _text(
        canvas: np.ndarray,
        text: str,
        point: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        *,
        center: bool = False,
        right: bool = False,
        thickness: int = 1,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (width, _), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = point
        if center:
            x -= width // 2
        elif right:
            x -= width
        cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def _chip(
        self,
        canvas: np.ndarray,
        text: str,
        point: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (width, height), _ = cv2.getTextSize(text, font, 0.34, 1)
        x, y = point
        cv2.rectangle(canvas, (x, y), (x + width + 18, y + height + 12), color, -1)
        self._text(canvas, text, (x + 9, y + height + 5), 0.34, (255, 255, 255))

    def _load_zones(self, manifest_path: Path) -> None:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._zones = data.get("zones", [])
        dimensions = data.get("dimensions_m")
        if dimensions:
            self._width_m = float(dimensions[0])
            self._depth_m = float(dimensions[1])
