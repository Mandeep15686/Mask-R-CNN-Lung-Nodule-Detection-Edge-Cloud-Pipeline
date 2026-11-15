#!/usr/bin/env python3
"""
Entry-point for the FastAPI backend server.
Validates environment, initialises the DB, and starts uvicorn.

Usage:
  python backend/run.py
  python backend/run.py --host 0.0.0.0 --port 8000 --reload

FIXES APPLIED
=============
FIX-10  RELOAD env var treats "false" as True:
  BEFORE: default=bool(os.getenv("RELOAD", ""))
          bool("false") == True because any non-empty string is truthy.
  AFTER:  Parse the env var as a case-insensitive string; only "1", "true",
          "yes", "on" enable reload.  Everything else (including "false",
          "0", "") is False.
"""
import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _str_to_bool(value: str) -> bool:
    """Parse a string environment variable as a boolean.

    Truthy strings: "1", "true", "yes", "on"  (case-insensitive)
    Everything else (including "false", "0", "") → False.
    """
    return value.strip().lower() in {"1", "true", "yes", "on"}


def check_environment():
    """Warn about missing optional dependencies or config."""
    warnings = []

    try:
        import torch
        if not torch.cuda.is_available():
            warnings.append("CUDA not available — inference will run on CPU (slower)")
    except ImportError:
        warnings.append("PyTorch not installed — inference engine disabled")

    model_path = os.getenv("MODEL_PATH", "./experiments/best_model.pth")
    if not os.path.exists(model_path):
        warnings.append(
            f"No checkpoint at {model_path} — running in dev mode (random predictions). "
            f"Run 'make train-p1' to train a model."
        )

    data_root = os.getenv("DATA_ROOT", "./data/luna16")
    if not os.path.exists(data_root):
        warnings.append(
            f"LUNA16 data not found at {data_root}. "
            f"Run 'make data' for synthetic data or 'make data-real' for instructions."
        )

    for w in warnings:
        logger.warning("⚠  %s", w)

    return len(warnings)


def main():
    parser = argparse.ArgumentParser(description="Lung Nodule Detection API Server")
    parser.add_argument("--host",    default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port",    type=int, default=int(os.getenv("PORT", "8000")))
    # FIX-10: use _str_to_bool instead of bool() so RELOAD=false → False
    parser.add_argument(
        "--reload",
        action="store_true",
        default=_str_to_bool(os.getenv("RELOAD", "")),
    )
    parser.add_argument("--workers",   type=int, default=int(os.getenv("WORKERS", "1")))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"))
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, project_root)

    n_warnings = check_environment()
    if n_warnings:
        logger.info("Starting with %d warning(s) — see above", n_warnings)

    logger.info("Starting server on http://%s:%d", args.host, args.port)
    logger.info("API docs: http://%s:%d/docs", args.host, args.port)

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        app_dir=os.path.dirname(os.path.abspath(__file__)),
    )


if __name__ == "__main__":
    main()
