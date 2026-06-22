# =========================================================
# INFRA GUARD — ENTERPRISE WEBSOCKET MANAGER
# =========================================================

from fastapi import WebSocket
from typing import List, Dict, Any

import asyncio
import json
from datetime import datetime

# =========================================================
# ENTERPRISE WEBSOCKET MANAGER
# =========================================================

class ConnectionManager:

    def __init__(self):

        # =============================================
        # ACTIVE CONNECTIONS
        # =============================================

        self.active_connections: List[
            WebSocket
        ] = []

        # =============================================
        # PER-CAMERA CONNECTIONS
        # cam_id -> list of websockets subscribed to it
        # =============================================

        self.camera_connections: Dict[
            int, List[WebSocket]
        ] = {}

        # =============================================
        # TELEMETRY
        # =============================================

        self.total_connections = 0

        self.total_events_sent = 0

    # =====================================================
    # CONNECT
    # =====================================================

    async def connect(
        self,
        cam_id: int,
        websocket: WebSocket
    ):

        await websocket.accept()

        self.active_connections.append(
            websocket
        )

        self.camera_connections.setdefault(
            cam_id, []
        ).append(
            websocket
        )

        self.total_connections += 1

        # =============================================
        # INITIAL PAYLOAD
        # =============================================

        await websocket.send_json({

            "type":
                "system",

            "status":
                "connected",

            "message":
                "InfraGuard realtime intelligence connected.",

            "timestamp":
                datetime.utcnow().isoformat(),

            "telemetry": {

                "active_connections":
                    len(
                        self.active_connections
                    ),

                "events_sent":
                    self.total_events_sent
            }
        })

    # =====================================================
    # DISCONNECT
    # =====================================================

    def disconnect(
        self,
        websocket: WebSocket
    ):

        if websocket in self.active_connections:

            self.active_connections.remove(
                websocket
            )

        for cam_id, sockets in self.camera_connections.items():

            if websocket in sockets:

                sockets.remove(
                    websocket
                )

    # =====================================================
    # SAFE SEND
    # =====================================================

    async def safe_send(

        self,

        websocket: WebSocket,

        payload: Dict[str, Any]
    ):

        try:

            await websocket.send_json(
                payload
            )

            return True

        except Exception:

            return False

    # =====================================================
    # BROADCAST TO ONE CAMERA'S SUBSCRIBERS
    # =====================================================

    async def broadcast_to_camera(

        self,

        cam_id: int,

        payload: Dict[str, Any]
    ):

        sockets = self.camera_connections.get(
            cam_id, []
        )

        if not sockets:
            return

        disconnected = []

        payload["timestamp"] = (

            datetime.utcnow().isoformat()
        )

        for connection in list(sockets):

            success = await self.safe_send(

                connection,
                payload
            )

            if not success:

                disconnected.append(
                    connection
                )

        for dead in disconnected:

            self.disconnect(dead)

        self.total_events_sent += 1

    # =====================================================
    # BROADCAST
    # =====================================================

    async def broadcast(

        self,

        payload: Dict[str, Any]
    ):

        disconnected = []

        payload["timestamp"] = (

            datetime.utcnow().isoformat()
        )

        for connection in self.active_connections:

            success = await self.safe_send(

                connection,
                payload
            )

            if not success:

                disconnected.append(
                    connection
                )

        # =============================================
        # CLEANUP
        # =============================================

        for dead in disconnected:

            self.disconnect(dead)

        self.total_events_sent += 1

    # =====================================================
    # DETECTION EVENT
    # =====================================================

    async def broadcast_detection(

        self,

        detection_type: str,

        severity: str,

        confidence: float,

        metadata: Dict[str, Any] = None
    ):

        await self.broadcast({

            "type":
                "detection",

            "detection_type":
                detection_type,

            "severity":
                severity,

            "confidence":
                round(confidence, 2),

            "metadata":
                metadata or {}
        })

    # =====================================================
    # SYSTEM EVENT
    # =====================================================

    async def broadcast_system(

        self,

        message: str,

        level: str = "info"
    ):

        await self.broadcast({

            "type":
                "system",

            "level":
                level,

            "message":
                message
        })

    # =====================================================
    # TELEMETRY
    # =====================================================

    async def broadcast_telemetry(

        self,

        fps: int,

        active_detections: int,

        cpu_usage: float = 0,

        memory_usage: float = 0
    ):

        await self.broadcast({

            "type":
                "telemetry",

            "fps":
                fps,

            "active_detections":
                active_detections,

            "cpu_usage":
                cpu_usage,

            "memory_usage":
                memory_usage
        })

    # =====================================================
    # HEARTBEAT
    # =====================================================

    async def heartbeat(self):

        while True:

            try:

                await self.broadcast({

                    "type":
                        "heartbeat",

                    "status":
                        "alive",

                    "active_connections":
                        len(
                            self.active_connections
                        )
                })

                await asyncio.sleep(10)

            except Exception:

                await asyncio.sleep(10)

# =========================================================
# GLOBAL INSTANCE
# =========================================================

manager = ConnectionManager()