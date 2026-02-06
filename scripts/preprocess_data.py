import csv
import json
import logging
import math
from collections import Counter
from pathlib import Path
from statistics import median

from config import DATA_PATH, PROCESSED_DATA_PATH
from scripts.telemetry import log_event, setup_logging


logger = logging.getLogger("loteca_preprocess")
logger.setLevel(logging.INFO)


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader)


def parse_target(row: dict) -> tuple[int | None, bool]:
    try:
        values = {
            "[1]": int(row["[1]"]),
            "[x]": int(row["[x]"]),
            "[2]": int(row["[2]"]),
        }
    except (KeyError, ValueError):
        return None, False

    if sum(values.values()) != 1:
        return None, False

    target_key = max(values, key=values.get)
    target = {"[1]": 0, "[x]": 1, "[2]": 2}[target_key]
    return target, True


def parse_odds(row: dict) -> tuple[list[float] | None, bool]:
    try:
        odds_1 = float(row["Odds_1"])
        odds_x = float(row["Odds_X"])
        odds_2 = float(row["Odds_2"])
    except (KeyError, ValueError):
        return None, False

    if odds_1 <= 0 or odds_x <= 0 or odds_2 <= 0:
        return None, False

    log_odds = [math.log(odds_1), math.log(odds_x), math.log(odds_2)]
    implied = [1.0 / odds_1, 1.0 / odds_x, 1.0 / odds_2]
    total_implied = sum(implied)
    if total_implied <= 0:
        return None, False
    normalized = [value / total_implied for value in implied]
    overround = total_implied
    market_entropy = 0.0
    for prob in normalized:
        if prob > 0:
            market_entropy -= prob * math.log(prob)
    sorted_probs = sorted(normalized, reverse=True)
    favorite_gap = sorted_probs[0] - sorted_probs[1]
    return [*log_odds, overround, market_entropy, favorite_gap], True


def build_training_set(rows: list[dict]) -> dict:
    features = []
    targets = []
    invalid_target = 0
    invalid_odds = 0
    suspicious_overround = 0
    target_counter: Counter[int] = Counter()
    concursos = []

    for row in rows:
        if row.get("Concurso"):
            try:
                concursos.append(int(row["Concurso"]))
            except ValueError:
                pass
        target, target_ok = parse_target(row)
        if not target_ok:
            invalid_target += 1
            continue

        odds, odds_ok = parse_odds(row)
        if not odds_ok:
            invalid_odds += 1
            continue
        if odds[3] < 0.95:
            suspicious_overround += 1
            continue

        features.append(odds)
        targets.append(target)
        target_counter[target] += 1

    logger.info("linhas_total=%s", len(rows))
    logger.info("linhas_invalidas_target=%s", invalid_target)
    logger.info("linhas_invalidas_odds=%s", invalid_odds)
    logger.info("linhas_overround_suspeito=%s", suspicious_overround)
    logger.info(
        "distribuicao_target={1: %s, X: %s, 2: %s}",
        target_counter.get(0, 0),
        target_counter.get(1, 0),
        target_counter.get(2, 0),
    )
    contest_range = {
        "min": min(concursos) if concursos else None,
        "max": max(concursos) if concursos else None,
    }
    log_event(
        "data_loaded",
        stage="preprocess",
        linhas_total=len(rows),
        invalid_target=invalid_target,
        invalid_odds=invalid_odds,
        contest_range=contest_range,
        target_distribution={
            "1": target_counter.get(0, 0),
            "X": target_counter.get(1, 0),
            "2": target_counter.get(2, 0),
        },
        linhas_overround_suspeito=suspicious_overround,
    )
    if features:
        feature_stats = []
        for idx in range(len(features[0])):
            values = [row[idx] for row in features]
            feature_stats.append(
                {
                    "feature_index": idx,
                    "min": min(values),
                    "median": median(values),
                    "max": max(values),
                }
            )
        log_event(
            "feature_stats",
            stage="preprocess",
            feature_stats=feature_stats,
        )
        overrounds = [row[3] for row in features]
        log_event(
            "market_quality",
            stage="preprocess",
            overround_min=min(overrounds),
            overround_median=median(overrounds),
            overround_max=max(overrounds),
            buckets={
                "<0.98": sum(1 for value in overrounds if value < 0.98),
                "<1.0": sum(1 for value in overrounds if value < 1.0),
                ">=1.0": sum(1 for value in overrounds if value >= 1.0),
            },
        )
        if min(overrounds) < 0.98:
            logger.warning(
                "overround_min=%.3f -> odds suspeitas: normalizacao aplicada",
                min(overrounds),
            )
            log_event(
                "warning_overround_low",
                stage="preprocess",
                level="WARNING",
                overround_min=min(overrounds),
                action="odds suspeitas: normalizacao aplicada",
            )

    return {"features": features, "targets": targets}


def save_processed(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def preprocess() -> None:
    setup_logging()
    rows = load_rows(DATA_PATH)
    processed = build_training_set(rows)
    save_processed(processed, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    preprocess()
