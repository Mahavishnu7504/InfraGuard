# =========================================================
# INFRA GUARD — ENTERPRISE WEBSOCKET MANAGER
# =========================================================
#
# Features:
#   - Multiple client support per camera
#   - Camera-room subscriptions
#   - Typed broadcast helpers (frame, alert, analytics, status, system)
#   - Heartbeat every 25 seconds to detect dead connections
#   - Auto-cleanup on disconnect
#   - Connection statistics
#   - Thread-safe async locking
#   - Error isolation (one bad client never blocks others)
#   - Authentication hook (ready for future integration)
# =========================================================

from fastapi import WebSocket
from typing import List, Dict, Any, Optional, Callable, Awaitable

import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger("infraguard.websocket")

# =========================================================
# TYPES
# =========================================================

AuthHook = Optional[Callable[[WebSocket, int], Awaitable[bool]]]

# =========================================================
# ENTERPRISE WEBSOCKET MANAGER
# =========================================================

class ConnectionManager:

    def __init__(
        self,
        heartbeat_interval: int = 25,
        auth_hook: AuthHook = None
    ):
        # =============================================
        # CONFIGURATION
        # =============================================

        self.heartbeat_interval = heartbeat_interval

        # Optional async callable: (websocket, cam_id) -> bool
        # Return False to reject the connection.
        self._auth_hook: AuthHook = auth_hook

        # =============================================
        # ACTIVE CONNECTIONS (global set)
        # =============================================

        self._all_connections: List[WebSocket] = []

        # =============================================
        # PER-CAMERA ROOMS
        # cam_id -> list[WebSocket]
        # =============================================

        self._camera_rooms: Dict[int, List[WebSocket]] = {}

        # =============================================
        # ASYNC LOCK — protects all mutable state
        # =============================================

        self._lock = asyncio.Lock()

        # =============================================
        # STATISTICS
        # =============================================

        self.stats: Dict[str, Any] = {
            "total_connections_ever":   0,
            "active_connections":       0,
            "messages_sent":            0,
            "failed_sends":             0,
            "connections_per_camera":   {},   # cam_id -> current count
        }

        # =============================================
        # HEARTBEAT TASK HANDLE
        # =============================================

        self._heartbeat_task: Optional[asyncio.Task] = None

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    async def _safe_send(
        self,
        websocket: WebSocket,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Send JSON to a single client.
        Returns True on success, False on any error.
        Errors are isolated so one bad socket never stops a broadcast.
        """
        try:
            await websocket.send_json(payload)
            async with self._lock:
                self.stats["messages_sent"] += 1
            return True
        except Exception as exc:
            logger.debug("send failed (%s): %s", id(websocket), exc)
            async with self._lock:
                self.stats["failed_sends"] += 1
            return False

    async def _remove_socket(self, websocket: WebSocket) -> None:
        """
        Remove websocket from every data structure.
        Must be called while holding self._lock OR the caller must hold it.
        """
        if websocket in self._all_connections:
            self._all_connections.remove(websocket)
            self.stats["active_connections"] -= 1

        for cam_id, sockets in list(self._camera_rooms.items()):
            if websocket in sockets:
                sockets.remove(websocket)
                cam_key = str(cam_id)
                if cam_key in self.stats["connections_per_camera"]:
                    self.stats["connections_per_camera"][cam_key] = max(
                        0,
                        self.stats["connections_per_camera"][cam_key] - 1
                    )

    # =========================================================
    # PUBLIC API — LIFECYCLE
    # =========================================================

    async def connect(
        self,
        cam_id: int,
        websocket: WebSocket
    ) -> bool:
        """
        Accept and register a new WebSocket client.

        Authentication hook:
            If an auth_hook was provided at construction, it is called here.
            Return False → connection is rejected (socket closed, returns False).
            Raise    → same: connection rejected.

        Returns True on success, False if rejected.
        """
        await websocket.accept()

        # ── Authentication hook (future-ready) ──────────────────────
        if self._auth_hook is not None:
            try:
                allowed = await self._auth_hook(websocket, cam_id)
            except Exception as exc:
                logger.warning("auth_hook raised: %s", exc)
                allowed = False

            if not allowed:
                await websocket.close(code=4003)
                logger.info("WebSocket rejected by auth_hook (cam=%s)", cam_id)
                return False
        # ─────────────────────────────────────────────────────────────

        async with self._lock:
            self._all_connections.append(websocket)
            self._camera_rooms.setdefault(cam_id, []).append(websocket)

            self.stats["total_connections_ever"] += 1
            self.stats["active_connections"]     += 1
            cam_key = str(cam_id)
            self.stats["connections_per_camera"][cam_key] = (
                self.stats["connections_per_camera"].get(cam_key, 0) + 1
            )

        logger.info(
            "WebSocket connected cam=%s total_active=%s",
            cam_id,
            self.stats["active_connections"]
        )

        # ── Send initial handshake ───────────────────────────────────
        await self._safe_send(websocket, {
            "type":      "system",
            "event":     "connected",
            "message":   "InfraGuard realtime intelligence connected.",
            "timestamp": self._now(),
            "stats":     self.get_stats(),
        })

        # ── Start heartbeat loop once ────────────────────────────────
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(),
                name="ws_heartbeat"
            )

        return True

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Cleanly remove a client from all rooms and free memory.
        Safe to call even if the socket is already gone.
        """
        async with self._lock:
            await self._remove_socket(websocket)

        logger.info(
            "WebSocket disconnected. active=%s",
            self.stats["active_connections"]
        )

    # =========================================================
    # PUBLIC API — BROADCAST HELPERS
    # =========================================================

    async def broadcast_frame(
        self,
        cam_id: int,
        frame_data: Dict[str, Any]
    ) -> None:
        """
        Push a processed frame result to all subscribers of cam_id.
        """
        await self._broadcast_to_camera(cam_id, {
            "type":   "frame_result",
            "cam_id": cam_id,
            **frame_data,
        })

    async def broadcast_alert(
        self,
        cam_id: int,
        alert: Dict[str, Any]
    ) -> None:
        """
        Push an alert to all subscribers of cam_id AND all global clients.
        """
        payload = {
            "type":   "alert",
            "cam_id": cam_id,
            **alert,
        }
        await self._broadcast_to_camera(cam_id, payload)

        # Also fan out to clients watching global/dashboard feeds
        await self._broadcast_global(payload, skip_cam_id=cam_id)

    async def broadcast_analytics(
        self,
        cam_id: int,
        analytics: Dict[str, Any]
    ) -> None:
        """
        Push analytics snapshot to cam_id subscribers.
        """
        await self._broadcast_to_camera(cam_id, {
            "type":   "analytics",
            "cam_id": cam_id,
            **analytics,
        })

    async def broadcast_status(
        self,
        cam_id: int,
        status: Dict[str, Any]
    ) -> None:
        """
        Push camera status update to cam_id subscribers.
        """
        await self._broadcast_to_camera(cam_id, {
            "type":   "camera_status",
            "cam_id": cam_id,
            **status,
        })

    async def broadcast_telemetry(
        self,
        cam_id: int,
        fps: float,
        active_detections: int,
        cpu_usage: float = 0.0,
        memory_usage: float = 0.0,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Push telemetry metrics for a camera.
        """
        payload: Dict[str, Any] = {
            "type":              "telemetry",
            "cam_id":            cam_id,
            "fps":               round(fps, 1),
            "active_detections": active_detections,
            "cpu_usage":         round(cpu_usage, 1),
            "memory_usage":      round(memory_usage, 1),
        }
        if extra:
            payload.update(extra)
        await self._broadcast_to_camera(cam_id, payload)

    async def broadcast_system(
        self,
        message: str,
        level: str = "info",
        cam_id: Optional[int] = None
    ) -> None:
        """
        Push a system message.
        If cam_id is given, sends to that camera's room only.
        Otherwise broadcasts to all connected clients.
        """
        payload = {
            "type":    "system",
            "level":   level,
            "message": message,
        }
        if cam_id is not None:
            await self._broadcast_to_camera(cam_id, payload)
        else:
            await self._broadcast_global(payload)

    # =========================================================
    # INTERNAL BROADCAST WORKERS
    # =========================================================

    async def _broadcast_to_camera(
        self,
        cam_id: int,
        payload: Dict[str, Any]
    ) -> None:
        payload.setdefault("timestamp", self._now())

        async with self._lock:
            sockets = list(self._camera_rooms.get(cam_id, []))

        if not sockets:
            return

        dead: List[WebSocket] = []

        for ws in sockets:
            ok = await self._safe_send(ws, payload)
            if not ok:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    await self._remove_socket(ws)

    async def _broadcast_global(
        self,
        payload: Dict[str, Any],
        skip_cam_id: Optional[int] = None
    ) -> None:
        """
        Broadcast to all connected clients, optionally skipping those
        already covered by a per-camera send.
        """
        payload.setdefault("timestamp", self._now())

        async with self._lock:
            skip_set: set = set()
            if skip_cam_id is not None:
                skip_set = set(self._camera_rooms.get(skip_cam_id, []))
            targets = [ws for ws in self._all_connections if ws not in skip_set]

        dead: List[WebSocket] = []

        for ws in targets:
            ok = await self._safe_send(ws, payload)
            if not ok:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    await self._remove_socket(ws)

    # =========================================================
    # HEARTBEAT LOOP
    # =========================================================

    async def _heartbeat_loop(self) -> None:
        """
        Runs as a background asyncio Task.
        Sends a heartbeat to all clients every `heartbeat_interval` seconds.
        Dead connections are pruned automatically.
        """
        logger.info(
            "Heartbeat loop started (interval=%ss)", self.heartbeat_interval
        )
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                async with self._lock:
                    active = len(self._all_connections)

                if active == 0:
                    continue

                await self._broadcast_global({
                    "type":              "heartbeat",
                    "status":            "alive",
                    "active_connections": active,
                    "timestamp":         self._now(),
                })

            except asyncio.CancelledError:
                logger.info("Heartbeat loop cancelled.")
                break
            except Exception as exc:
                logger.error("Heartbeat error: %s", exc)

    # =========================================================
    # STATISTICS
    # =========================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Return a snapshot of connection statistics.
        Safe to call from sync context (no await needed).
        """
        return {
            "total_connections_ever":   self.stats["total_connections_ever"],
            "active_connections":       self.stats["active_connections"],
            "messages_sent":            self.stats["messages_sent"],
            "failed_sends":             self.stats["failed_sends"],
            "connections_per_camera":   dict(self.stats["connections_per_camera"]),
        }

    # =========================================================
    # AUTHENTICATION HOOK SETTER
    # =========================================================

    def set_auth_hook(self, hook: AuthHook) -> None:
        """
        Register (or replace) an async authentication callable.

        Signature:
            async def my_auth(websocket: WebSocket, cam_id: int) -> bool: ...

        Return True  → allow connection
        Return False → reject connection (close with 4003)
        """
        self._auth_hook = hook

    # =========================================================
    # GRACEFUL SHUTDOWN
    # =========================================================

    async def shutdown(self) -> None:
        """
        Cancel the heartbeat task and close all active connections.
        Call this from your application lifespan shutdown handler.
        """
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            all_ws = list(self._all_connections)

        for ws in all_ws:
            try:
                await ws.close(code=1001)
            except Exception:
                pass

        async with self._lock:
            self._all_connections.clear()
            self._camera_rooms.clear()
            self.stats["active_connections"] = 0
            self.stats["connections_per_camera"].clear()

        logger.info("WebSocket manager shut down cleanly.")


# =========================================================
# GLOBAL SINGLETON
# =========================================================

manager = ConnectionManager(heartbeat_interval=25)