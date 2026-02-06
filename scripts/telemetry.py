import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RUNS_DIR = Path("runs")
_CONTEXT: "TelemetryContext | None" = None


@dataclass(frozen=True)
class TelemetryContext:
    run_id: str
    run_dir: Path
    events_path: Path
    log_level: int


def init_run(run_id: str | None = None, runs_dir: Path = DEFAULT_RUNS_DIR) -> TelemetryContext:
    global _CONTEXT
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    log_level = _parse_log_level(os.getenv("LOG_LEVEL", "INFO"))
    _CONTEXT = TelemetryContext(
        run_id=run_id,
        run_dir=run_dir,
        events_path=events_path,
        log_level=log_level,
    )
    return _CONTEXT


def get_context() -> TelemetryContext:
    if _CONTEXT is None:
        return init_run()
    return _CONTEXT


def setup_logging(context: TelemetryContext | None = None, level: int | None = None) -> None:
    if context is None:
        context = get_context()
    log_level = context.log_level if level is None else level
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(context.run_dir / "console.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def log_event(
    event: str,
    *,
    stage: str,
    level: str = "INFO",
    **fields: Any,
) -> None:
    context = get_context()
    event_level = _parse_log_level(level)
    if event_level < context.log_level:
        return
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": context.run_id,
        "stage": stage,
        "event": event,
        **fields,
    }
    context.events_path.parent.mkdir(parents=True, exist_ok=True)
    with context.events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_json(name: str, payload: dict) -> Path:
    context = get_context()
    path = context.run_dir / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_csv(name: str, headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> Path:
    import csv

    context = get_context()
    path = context.run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))
        for row in rows:
            writer.writerow(list(row))
    return path


def _parse_log_level(level: str) -> int:
    value = level.strip().upper()
    if value == "DEBUG":
        return logging.DEBUG
    if value == "WARNING":
        return logging.WARNING
    if value == "ERROR":
        return logging.ERROR
    if value == "CRITICAL":
        return logging.CRITICAL
    return logging.INFO
