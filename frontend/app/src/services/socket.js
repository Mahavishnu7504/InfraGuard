const BASE_URL =
    import.meta.env.VITE_API_URL ||
    "http://127.0.0.1:8000";

const WS_BASE = BASE_URL
    .replace("https", "wss")
    .replace("http", "ws");

/* =========================================
   ENTERPRISE SOCKET ENGINE
========================================= */

class InfraGuardSocket {

    constructor(
        camId,
        handlers = {}
    ) {

        this.camId = camId;

        this.handlers = handlers;

        this.socket = null;

        this.connected = false;

        this.manualClose = false;

        this.reconnectAttempts = 0;

        this.maxReconnects = 20;

        this.reconnectDelay = 2500;

        this.reconnectTimer = null;

        this.heartbeatTimer = null;

        this.connect();
    }

    /* =====================================
       CONNECT
    ===================================== */

    connect() {

        if (
            this.socket &&
            (
                this.socket.readyState === WebSocket.OPEN ||
                this.socket.readyState === WebSocket.CONNECTING
            )
        ) {
            return;
        }

        try {

            this.socket = new WebSocket(
                `${WS_BASE}/safety/ws/${this.camId}`
            );

            this.socket.onopen = () => {

                this.connected = true;

                this.reconnectAttempts = 0;

                console.log(
                    `[WS CONNECTED] CAM ${this.camId}`
                );

                this.startHeartbeat();

                this.handlers.onOpen?.({
                    camId: this.camId
                });
            };

            this.socket.onmessage = (
                event
            ) => {

                try {

                    const data = JSON.parse(
                        event.data
                    );

                    this.handlers.onMessage?.(
                        data
                    );

                } catch (err) {

                    console.error(
                        "[WS MESSAGE ERROR]",
                        err
                    );
                }
            };

            this.socket.onerror = (
                err
            ) => {

                console.error(
                    `[WS ERROR][CAM ${this.camId}]`,
                    err
                );

                this.handlers.onError?.(
                    err
                );
            };

            this.socket.onclose = () => {

                console.warn(
                    `[WS CLOSED][CAM ${this.camId}]`
                );

                this.connected = false;

                this.stopHeartbeat();

                this.handlers.onClose?.();

                if (!this.manualClose) {

                    this.reconnect();
                }
            };

        } catch (err) {

            console.error(
                "[WS INIT ERROR]",
                err
            );

            this.reconnect();
        }
    }

    /* =====================================
       HEARTBEAT
    ===================================== */

    startHeartbeat() {

        this.stopHeartbeat();

        this.heartbeatTimer =
            setInterval(() => {

                if (
                    this.socket &&
                    this.connected
                ) {

                    try {

                        this.socket.send(
                            JSON.stringify({
                                type: "ping"
                            })
                        );

                    } catch { }
                }

            }, 30000);
    }

    stopHeartbeat() {

        if (
            this.heartbeatTimer
        ) {

            clearInterval(
                this.heartbeatTimer
            );

            this.heartbeatTimer = null;
        }
    }

    /* =====================================
       SEND
    ===================================== */

    send(payload) {

        if (
            this.socket &&
            this.connected
        ) {

            this.socket.send(
                JSON.stringify(
                    payload
                )
            );
        }
    }

    /* =====================================
       RECONNECT
    ===================================== */

    reconnect() {

        if (
            this.reconnectAttempts >=
            this.maxReconnects
        ) {

            console.error(
                `[WS FAILED][CAM ${this.camId}]`
            );

            return;
        }

        this.reconnectAttempts++;

        console.log(
            `[WS RECONNECT ${this.reconnectAttempts}] CAM ${this.camId}`
        );

        this.reconnectTimer =
            setTimeout(() => {

                this.connect();

            }, this.reconnectDelay);
    }

    /* =====================================
       CLOSE
    ===================================== */

    close() {

        this.manualClose = true;

        this.stopHeartbeat();

        if (
            this.reconnectTimer
        ) {

            clearTimeout(
                this.reconnectTimer
            );
        }

        if (this.socket) {

            this.socket.close();

            this.socket = null;
        }

        this.connected = false;
    }
}

/* =========================================
   FACTORY
========================================= */

export const createSocket = (
    camId,
    handlers = {}
) => {

    return new InfraGuardSocket(
        camId,
        handlers
    );
};