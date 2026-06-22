from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.safety_routes import router as safety_router
from backend.api.activity_routes import router as activity_router
from backend.api.quality_routes import router as quality_router

from backend.core.database import Base, engine


# -----------------------------
# MODEL LOAD
# -----------------------------
try:
    from backend.models import db_models
    print("[MODELS LOADED]")
except Exception as e:
    print("[MODEL WARNING]", e)


# -----------------------------
# APP LIFECYCLE
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
        print("[DATABASE READY]")
    except Exception as e:
        print("[DATABASE ERROR]", e)

    print("[SYSTEM ONLINE]")
    yield

    try:
        from backend.services.safety.stream_service import stop_all_cameras
        stop_all_cameras()
        print("[CAMERAS STOPPED]")
    except Exception as e:
        print("[SHUTDOWN WARNING]", e)

    print("[SYSTEM CLOSED]")


# -----------------------------
# APP
# -----------------------------
app = FastAPI(
    title="InfraGuard AI Platform",
    version="5.3 Final",
    lifespan=lifespan
)


# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# ROUTERS
# -----------------------------
app.include_router(
    safety_router,
    prefix="/safety",
    tags=["Safety"]
)

app.include_router(
    activity_router,
    prefix="/activity",
    tags=["Activity"]
)

app.include_router(
    quality_router,
    prefix="/quality",
    tags=["Quality"]
)


# -----------------------------
# HEALTH
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "app": "InfraGuard",
        "version": "5.3"
    }


@app.get("/health")
def health():
    try:
        conn = engine.connect()
        conn.close()
        db = "connected"
    except Exception:
        db = "error"

    return {
        "healthy": True,
        "database": db
    }


@app.get("/ping")
def ping():
    return {"message": "pong"}