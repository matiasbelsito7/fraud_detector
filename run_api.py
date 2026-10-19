from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from src.api.app import CONFIG_ENV, DEFAULT_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Servidor FastAPI de inferencia.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
        help="Ruta a deployment.json (por defecto config/deployment.json).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Recarga en desarrollo.")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    args = parser.parse_args()

    os.environ[CONFIG_ENV] = str(args.config)
    uvicorn.run(
        "src.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
