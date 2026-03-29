"""
QuantSwarm v3 — FastAPI Dashboard Backend
Real-time signal feed, portfolio, risk, and prediction endpoints.
Run: uvicorn dashboard.api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio

logger = logging.getLogger("quantswarm.dashboard")

app = FastAPI(
    title="QuantSwarm v3 Dashboard",
    description="Real-time quant trading signal engine",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state (replace with DB in production)
_state: Dict[str, Any] = {
    "status": "initializing",
    "portfolio_value": 0,
    "initial_capital": 0,
    "regime": "unknown",
    "drift_tier": "none",
    "signals": [],
    "positions": [],
    "risk_summary": {},
    "predictions": [],
    "last_cycle": None,
}

_ws_clients: List[WebSocket] = []


# === REST Endpoints ===

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/status")
async def status():
    return {
        "agent_status": _state["status"],
        "portfolio_value": _state["portfolio_value"],
        "regime": _state["regime"],
        "drift_tier": _state["drift_tier"],
        "last_cycle": _state["last_cycle"],
    }


@app.get("/signals")
async def get_signals(limit: int = 50):
    return {"signals": _state["signals"][-limit:]}


@app.get("/predictions")
async def get_predictions(limit: int = 20):
    return {"predictions": _state["predictions"][-limit:]}


@app.get("/positions")
async def get_positions():
    return {"positions": _state["positions"]}


@app.get("/risk")
async def get_risk():
    return _state["risk_summary"]


@app.get("/instruments")
async def get_instruments():
    """Return all 100 tracked instruments with latest data."""
    from config_loader import get_config
    config = get_config()
    return {
        "stocks": config["instruments"]["stocks"],
        "crypto": config["instruments"]["crypto"],
        "total": len(config["instruments"]["stocks"]) + len(config["instruments"]["crypto"]),
    }


@app.post("/pause")
async def pause_agent():
    """Manually pause the agent."""
    _state["status"] = "paused_manual"
    await broadcast({"type": "status_change", "status": "paused_manual"})
    return {"message": "Agent paused"}


@app.post("/resume")
async def resume_agent():
    """Resume the agent."""
    _state["status"] = "running"
    await broadcast({"type": "status_change", "status": "running"})
    return {"message": "Agent resumed"}


# === WebSocket for real-time updates ===

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        # Send current state on connect
        await ws.send_json({"type": "init", "state": _state})
        while True:
            # Keep connection alive, handle client messages
            data = await asyncio.wait_for(ws.receive_text(), timeout=30)
            if data == "ping":
                await ws.send_text("pong")
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


async def broadcast(message: dict):
    """Broadcast message to all connected WebSocket clients."""
    disconnected = []
    for client in _ws_clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.append(client)
    for c in disconnected:
        if c in _ws_clients:
            _ws_clients.remove(c)


def update_state(cycle_result: dict):
    """Called by agent after each cycle to update dashboard state."""
    _state["last_cycle"] = datetime.utcnow().isoformat()
    _state["status"] = "running"

    if "risk" in cycle_result:
        _state["risk_summary"] = cycle_result["risk"]
        _state["portfolio_value"] = cycle_result["risk"].get("portfolio_value", 0)

    if "thoughts" in cycle_result:
        _state["regime"] = cycle_result["thoughts"].get("regime", "unknown")
        _state["drift_tier"] = cycle_result["thoughts"].get("drift", "none")

    if "result" in cycle_result and "actions" in cycle_result["result"]:
        for action in cycle_result["result"]["actions"]:
            _state["signals"].append({
                "timestamp": datetime.utcnow().isoformat(),
                **action
            })
        if len(_state["signals"]) > 500:
            _state["signals"] = _state["signals"][-500:]


# === Config loader helper ===

def get_config():
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "base.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Serve React frontend ──────────────────────────────────────────────────────
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os as _os

_DASH_DIR = _os.path.dirname(_os.path.abspath(__file__))

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the React dashboard."""
    return FileResponse(_os.path.join(_DASH_DIR, "index.html"))

# Full state endpoint matching the frontend's expected schema
@app.get("/state")
async def get_full_state():
    """Return the full dashboard state for initial load."""
    return JSONResponse(_state)
