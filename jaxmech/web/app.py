"""FastAPI application — jaxmech web frontend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from jaxmech.web.routers import status, models, config, analysis, docs, visualization

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="jaxmech", version="0.1.0")

# Register API routers
app.include_router(status.router)
app.include_router(models.router)
app.include_router(config.router)
app.include_router(analysis.router)
app.include_router(docs.router)
app.include_router(visualization.router)

# Serve static files (CSS / JS / images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    """Disable browser caching for static JS/CSS files during development."""
    response: Response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/")
async def index():
    """Serve the SPA entry page."""
    return FileResponse(str(TEMPLATES_DIR / "index.html"))
