import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from config import (
    DUPLO_COUNT,
    L1_DELTA_TH,
    L1_DELTA_TH_HIGH,
    L1_DELTA_TH_LOW,
    MARKET_FAVORITE_FLIP_TH,
    MARKET_FAVORITE_HIGH_TH,
    MARKET_FAVORITE_MID_HIGH,
    MARKET_FAVORITE_MID_LOW,
    MODEL_PATH,
    NEXT_CONTEST_PATH,
    OBJECTIVE_ALPHA,
    OBJECTIVE_MODE,
    OPTIMIZATION_MODE,
    CONTRARIAN_MIN_P,
    DUPLO_RISK_TOPK_GAMES,
    DUPLO_VALUE_ALPHA,
    ENTROPY_COIN_TOSS_TH,
    MARGIN_COIN_TOSS_TH,
    ANTI_PULVERIZATION_BIAS,
    OUTPUT_CARD_PATH,
    PULVERIZATION_LAMBDA,
    PULVERIZATION_CONTRARIAN_BONUS_WEIGHT,
    PULVERIZATION_MARKET_PENALTY_WEIGHT,
    PULVERIZATION_PENALTY_BUDGET_AGRESSIVE,
    PULVERIZATION_PENALTY_BUDGET_ROBUSTO,
    PULVERIZATION_PENALTY_WEIGHTS,
    RISK_WEIGHTS,
    ROBUST_MIN_P13_PLUS,
    SECO_MIN_ACERTO_TH,
    SECO_FLIP_MARGIN_TH,
    SECO_FRACO_TH,
    SECO_FRACO_TOP2_TH,
    DUPLO_TOPK_PAIRS,
    SENSITIVITY_SHIFTS,
    SEED,
    SECO_FLIP_GAP_MIN,
)
from scripts.card_builder import (
    choose_duplo_pick,
    compute_pulverization_metrics,
    compute_risk,
    entropy,
    log_card_summary,
    log_card_table,
    score_game,
    select_best_duplos,
    select_duplos_heuristic,
    summarize_card,
    summarize_sensitivity,
)
from scripts.telemetry import get_context, log_event, save_csv, save_json, setup_logging


logger = logging.getLogger("loteca_antipulverizacao")
logger.setLevel(logging.INFO)


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_next_contest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader)


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_scores = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


def build_features(rows: list[dict]) -> tuple[list[list[float]], list[list[float]]]:
    features = []
    market_probs = []
    for row in rows:
        odds_1 = float(row["Odds_1"])
        odds_x = float(row["Odds_X"])
        odds_2 = float(row["Odds_2"])
        log_odds = [math.log(odds_1), math.log(odds_x), math.log(odds_2)]
        implied = [1.0 / odds_1, 1.0 / odds_x, 1.0 / odds_2]
        total_implied = sum(implied)
        normalized = [value / total_implied for value in implied]
        overround = total_implied
        market_probs.append(normalized)
        market_entropy = 0.0
        for prob in normalized:
            if prob > 0:
                market_entropy -= prob * math.log(prob)
        sorted_probs = sorted(normalized, reverse=True)
        favorite_gap = sorted_probs[0] - sorted_probs[1]
        features.append([*log_odds, overround, market_entropy, favorite_gap])
    return features, market_probs


def predict_probabilities(model: dict, features: list[list[float]]) -> list[list[float]]:
    weights = model["weights"]
    num_classes = len(model["classes"])
    temperature = float(model.get("temperature", 1.0))
    probabilities = []
    for row in features:
        row_with_bias = row + [1.0]
        logits = [
            sum(row_with_bias[i] * weights[i][cls] for i in range(len(row_with_bias)))
            for cls in range(num_classes)
        ]
        scaled_logits = [value / temperature for value in logits]
        probabilities.append(softmax(scaled_logits))
    return probabilities


def apply_antipulverization_bias(
    probabilities: list[list[float]],
    market_probabilities: list[list[float]],
    bias_strength: float,
    min_prob: float = 0.01,
) -> list[list[float]]:
    if bias_strength <= 0:
        return probabilities
    adjusted: list[list[float]] = []
    for probs, market in zip(probabilities, market_probabilities):
        shifted = [max(min_prob, p + bias_strength * (p - m)) for p, m in zip(probs, market)]
        total = sum(shifted)
        if total <= 0:
            adjusted.append(probs)
        else:
            adjusted.append([value / total for value in shifted])
    return adjusted


def format_pick(probabilities: list[float], is_duplo: bool) -> str:
    outcomes = ["1", "X", "2"]
    sorted_indices = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)
    if is_duplo:
        pick = "/".join(outcomes[idx] for idx in sorted_indices[:2])
    else:
        pick = outcomes[sorted_indices[0]]
    return pick


def build_card(
    rows: list[dict],
    probabilities: list[list[float]],
    market_probabilities: list[list[float]],
    objective: str,
    objective_alpha: float,
    lambda_penalty: float,
    penalty_weights: tuple[float, float, float],
    seco_fraco_th: float,
    seco_top2_th: float,
    l1_delta_th: float,
    l1_delta_th_high: float,
    l1_delta_th_low: float,
    seco_flip_margin_th: float,
    seco_flip_gap_min: float,
    top_k_pairs: int,
    optimization_mode: str,
    penalty_budget: float | None,
    min_p13_plus: float | None,
    risk_weights: tuple[float, float, float],
    duplo_value_alpha: float,
    market_penalty_weight: float,
    contrarian_bonus_weight: float,
    risk_top_k_games: int,
    market_favorite_mid_low: float,
    market_favorite_mid_high: float,
    market_favorite_high_th: float,
    market_favorite_flip_th: float,
    contrarian_min_p: float,
    entropy_coin_toss_th: float,
    margin_coin_toss_th: float,
    seco_min_acerto_th: float,
) -> tuple[List[str], dict]:
    if l1_delta_th_high is None:
        l1_delta_th_high = l1_delta_th
    if l1_delta_th_low is None:
        l1_delta_th_low = l1_delta_th
    l1_high_games = [
        idx
        for idx, (probs, market_probs) in enumerate(zip(probabilities, market_probabilities))
        if sum(abs(model - market) for model, market in zip(probs, market_probs)) >= l1_delta_th_high
    ]
    selection_info = select_best_duplos(
        probabilities,
        market_probabilities,
        must_include_games=l1_high_games,
        duplo_count=DUPLO_COUNT,
        lambda_penalty=lambda_penalty,
        penalty_weights=penalty_weights,
        market_penalty_weight=market_penalty_weight,
        contrarian_bonus_weight=contrarian_bonus_weight,
        objective=objective,
        objective_alpha=objective_alpha,
        seco_pmax_threshold=seco_fraco_th,
        seco_top2_threshold=seco_top2_th,
        top_k_pairs=top_k_pairs,
        risk_weights=risk_weights,
        risk_top_k_games=risk_top_k_games,
        duplo_value_alpha=duplo_value_alpha,
        market_favorite_mid_low=market_favorite_mid_low,
        market_favorite_mid_high=market_favorite_mid_high,
        market_favorite_high_th=market_favorite_high_th,
        contrarian_min_p=contrarian_min_p,
        contrarian_count_l1_th=l1_delta_th_low,
        entropy_coin_toss_th=entropy_coin_toss_th,
        margin_coin_toss_th=margin_coin_toss_th,
        penalty_budget=penalty_budget,
        mode=optimization_mode,
        min_p13_plus=min_p13_plus,
    )
    duplo_indices = selection_info["duplo_indices"]
    if len(duplo_indices) != DUPLO_COUNT:
        logger.warning(
            "duplos_incompletos=%s esperado=%s -> usando heuristica de fallback",
            len(duplo_indices),
            DUPLO_COUNT,
        )
        duplo_indices = select_duplos_heuristic(probabilities, DUPLO_COUNT)
        selection_info["duplo_indices"] = duplo_indices
    picks = []
    per_game_scores = []
    p_acertos = []
    pick_sources = []
    source_counts = {"mercado": 0, "mercado_min": 0, "flip_contrarian": 0}
    secos_fracos_detalhes = []
    decision_table = []
    warnings = []
    for idx, row in enumerate(rows):
        probs = probabilities[idx]
        market_probs = market_probabilities[idx]
        market_index = max(range(len(market_probs)), key=market_probs.__getitem__)
        market_pick = ["1", "X", "2"][market_index]
        market_max = market_probs[market_index]
        sorted_probs = sorted(probs, reverse=True)
        pmax, second, third = sorted_probs[0], sorted_probs[1], sorted_probs[2]
        p_top2 = pmax + second
        margin = pmax - second
        score = score_game(probs)
        per_game_scores.append(score)
        is_duplo = idx in duplo_indices
        pick_source = "modelo"
        pick_probs = probs
        pick = format_pick(probs, is_duplo)
        l1_delta = sum(abs(model - market) for model, market in zip(probs, market_probs))
        duplo_meta = None
        if is_duplo:
            duplo_meta = choose_duplo_pick(probs, market_probs, value_alpha=duplo_value_alpha)
            pick = duplo_meta["pick"]
            pick_source = "duplo_auto"
        else:
            sorted_indices = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
            pick_index = sorted_indices[0]
            alt_index = sorted_indices[1]
            alt_prob = probs[alt_index]
            market_gap = market_probs[market_index] - market_probs[alt_index]
            coin_toss = entropy(probs) > entropy_coin_toss_th and margin < margin_coin_toss_th
            allow_flip = (
                alt_prob >= probs[pick_index] - seco_flip_margin_th
                and market_probs[pick_index] >= market_favorite_flip_th
                and market_gap >= seco_flip_gap_min
                and l1_delta >= l1_delta_th_low
            )
            if coin_toss and alt_prob >= probs[pick_index] - seco_flip_margin_th:
                allow_flip = True
            if pmax < seco_min_acerto_th and market_probs[market_index] > probs[pick_index]:
                pick_index = market_index
                pick_source = "mercado_min"
                pick_probs = market_probs
            elif pmax < seco_fraco_th and l1_delta > l1_delta_th_high:
                pick_index = market_index
                pick_source = "mercado"
                pick_probs = market_probs
            elif allow_flip:
                pick_index = alt_index
                pick_source = "flip_contrarian"
                pick_probs = probs
            pick = ["1", "X", "2"][pick_index]
        if pick_source in source_counts:
            source_counts[pick_source] += 1
        pick_sources.append(pick_source)
        picks.append(pick)
        if is_duplo:
            p_acerto = duplo_meta["p_cover"] if duplo_meta is not None else p_top2
        else:
            pick_index = ["1", "X", "2"].index(pick)
            p_acerto = pick_probs[pick_index]
        p_acertos.append(p_acerto)
        risk = compute_risk(p_acerto, margin, entropy(probs), risk_weights)
        if is_duplo:
            coin_toss = entropy(probs) > entropy_coin_toss_th and margin < margin_coin_toss_th
        if not is_duplo and p_acerto < seco_fraco_th:
            secos_fracos_detalhes.append(
                (
                    row["Jogo"],
                    p_acerto,
                    pmax,
                    margin,
                    entropy(probs),
                    l1_delta,
                    pick_source,
                )
            )
            log_event(
                "seco_flagged",
                stage="predict",
                level="DEBUG",
                game=int(row["Jogo"]),
                p_acerto=p_acerto,
                pmax=pmax,
                margin=margin,
                entropy=entropy(probs),
                l1_delta=l1_delta,
                source=pick_source,
                seco_fraco=True,
                seco_fraco_top2=p_top2 >= seco_top2_th,
                flip_risk=margin < seco_flip_margin_th,
            )
        log_event(
            "match_probs",
            stage="predict",
            level="DEBUG",
            game=int(row["Jogo"]),
            home=row["Mandante"],
            away=row["Visitante"],
            p1=probs[0],
            pX=probs[1],
            p2=probs[2],
            pick=pick,
            source=pick_source,
            entropy=entropy(probs),
            margin=margin,
            l1_delta=l1_delta,
            risk=risk,
        )
        log_event(
            "market_vs_model",
            stage="predict",
            level="DEBUG",
            game=int(row["Jogo"]),
            delta_p1=probs[0] - market_probs[0],
            delta_pX=probs[1] - market_probs[1],
            delta_p2=probs[2] - market_probs[2],
            l1_delta=l1_delta,
        )
        if l1_delta > L1_DELTA_TH_HIGH:
            warnings.append(
                {
                    "game": int(row["Jogo"]),
                    "l1_delta": l1_delta,
                    "message": "modelo muito divergente do mercado (possivel ruido / feature drift)",
                }
            )
            logger.warning(
                "l1_delta=%.3f jogo=%s -> modelo muito divergente do mercado (possivel ruido / feature drift)",
                l1_delta,
                row["Jogo"],
            )
            log_event(
                "warning_l1_delta_high",
                stage="predict",
                level="WARNING",
                game=int(row["Jogo"]),
                l1_delta=l1_delta,
                action="modelo muito divergente do mercado (possivel ruido / feature drift)",
            )
        decision_table.append(
            {
                "game": row["Jogo"],
                "home": row["Mandante"],
                "away": row["Visitante"],
                "p1": probs[0],
                "pX": probs[1],
                "p2": probs[2],
                "market_p1": market_probs[0],
                "market_pX": market_probs[1],
                "market_p2": market_probs[2],
                "pmax": pmax,
                "second": second,
                "third": third,
                "p_top2": p_top2,
                "margin": margin,
                "entropy": entropy(probs),
                "l1_delta": l1_delta,
                "risk": risk,
                "p_acerto": p_acerto,
                "market_max": market_max,
                "market_pick": market_pick,
                "source": pick_source,
                "pick": pick,
                "is_duplo": is_duplo,
                "seco_fraco": (not is_duplo and p_acerto < seco_fraco_th),
                "seco_fraco_top2": (not is_duplo and p_top2 >= seco_top2_th and p_acerto < seco_fraco_th),
                "flip_risk": (not is_duplo and margin < seco_flip_margin_th),
                "coin_toss": coin_toss,
                "l1_delta_high": l1_delta >= l1_delta_th_high,
            }
        )

    contrarian_count = sum(
        1
        for row in decision_table
        if (not row["is_duplo"])
        and row["pick"] != row["market_pick"]
        and row["p_acerto"] >= contrarian_min_p
        and row["l1_delta"] >= l1_delta_th_high
    )
    secos = len(picks) - len(duplo_indices)
    summary = summarize_card(p_acertos)
    pulverization_metrics = compute_pulverization_metrics(
        probabilities,
        duplo_indices,
        p_acertos=p_acertos,
        seco_pmax_threshold=seco_fraco_th,
        seco_top2_threshold=seco_top2_th,
        market_probabilities=market_probabilities,
        duplo_value_alpha=duplo_value_alpha,
    )
    if pulverization_metrics["secos_fracos_top2"] > 0:
        logger.warning(
            "secos_fracos_top2=%s -> modo agressivo penalizou +%s; considere mover um desses jogos para duplo",
            pulverization_metrics["secos_fracos_top2"],
            pulverization_metrics["secos_fracos_top2"],
        )
        log_event(
            "warning_secos_fracos_top2",
            stage="predict",
            level="WARNING",
            secos_fracos_top2=pulverization_metrics["secos_fracos_top2"],
            action="considere mover um desses jogos para duplo",
        )
    sensitivity_summary = {
        str(shift): summarize_sensitivity(p_acertos, downshift=shift) for shift in SENSITIVITY_SHIFTS
    }
    log_card_summary(
        summary,
        secos,
        len(duplo_indices),
        pulverization_metrics,
        sensitivity_summary,
        list(SENSITIVITY_SHIFTS),
    )
    log_event(
        "final_summary",
        stage="predict",
        summary=summary,
        pulverization_metrics=pulverization_metrics,
        secos=secos,
        duplos=len(duplo_indices),
    )
    logger.info(
        "count_source_mercado=%s count_source_mercado_min=%s count_source_flip_contrarian=%s",
        source_counts["mercado"],
        source_counts["mercado_min"],
        source_counts["flip_contrarian"],
    )
    if secos_fracos_detalhes:
        logger.debug("secos_fracos_detalhes=[(jogo,p_acerto,pmax,margin,entropy,l1_delta,source)]:")
        for detalhe in secos_fracos_detalhes:
            logger.debug(
                "seco_fraco jogo=%s p_acerto=%.4f pmax=%.4f margin=%.4f entropy=%.4f l1_delta=%.4f source=%s",
                detalhe[0],
                detalhe[1],
                detalhe[2],
                detalhe[3],
                detalhe[4],
                detalhe[5],
                detalhe[6],
            )

    table_rows = []
    for idx, row in enumerate(rows):
        table_rows.append(
            {
                "Jogo": row["Jogo"],
                "Mandante": row["Mandante"],
                "Visitante": row["Visitante"],
                "probs": probabilities[idx],
                "is_duplo": idx in duplo_indices,
                "pick_source": pick_sources[idx],
            }
        )
    log_card_table(table_rows, picks, p_acertos, per_game_scores)

    if selection_info["best_pair"] is not None:
        best_summary = selection_info["best_summary"]
        runner_up_summary = selection_info["runner_up_summary"]
        best_pair = tuple(index + 1 for index in selection_info["best_pair"])
        runner_up_pair = (
            tuple(index + 1 for index in selection_info["runner_up_pair"])
            if selection_info["runner_up_pair"] is not None
            else None
        )
        logger.info(
            "best_pair=%s best_pair_jogos=%s best_P13_exact=%.6f best_P13_plus=%.6f best_P14=%.6f best_E=%.3f best_Var=%.3f",
            selection_info["best_pair"],
            best_pair,
            best_summary["p13_exact_est"],
            best_summary["p13_plus_est"],
            best_summary["p14_est"],
            best_summary["expected"],
            best_summary["variance"],
        )
        if runner_up_summary is not None:
            logger.info(
                "runner_up_pair=%s runner_up_pair_jogos=%s runner_up_P13_plus=%.6f delta_P13_plus=%.6f",
                selection_info["runner_up_pair"],
                runner_up_pair,
                runner_up_summary["p13_plus_est"],
                selection_info["delta_p13_plus"],
            )
            if selection_info["delta_p13_plus"] is not None and abs(selection_info["delta_p13_plus"]) <= 1e-6:
                logger.warning(
                    "best_pair_tie delta_P13_plus=%.6f -> empate tecnico: usar critério secundario",
                    selection_info["delta_p13_plus"],
                )
                log_event(
                    "warning_best_pair_tie",
                    stage="predict",
                    level="WARNING",
                    delta_p13_plus=selection_info["delta_p13_plus"],
                    action="empate tecnico: usar critério secundario",
                    tie_breaker_used=selection_info.get("tie_breaker_used"),
                )
        if selection_info["top_candidates"]:
            best_penalty = selection_info["top_candidates"][0]
            logger.info(
                "objetivo_otimizacao=%.6f penalty_total=%.6f penalty_fragilidade=%.6f penalty_secos_fracos=%.6f penalty_placebo=%.6f",
                selection_info["best_objective"],
                best_penalty["penalty_total"],
                best_penalty["penalty_fragilidade"],
                best_penalty["penalty_secos_fracos"],
                best_penalty["penalty_placebo"],
            )
            logger.info(
                "penalty_market_follow=%.6f contrarian_bonus=%.6f",
                best_penalty["penalty_market_follow"],
                best_penalty["contrarian_bonus"],
            )
            logger.info(
                "objective_formula=objective_base - lambda*penalty_total term_p13=%.6f term_p14=%.6f term_penalty=%.6f",
                best_penalty["summary"]["p13_plus_est"],
                best_penalty["summary"]["p14_est"],
                best_penalty["penalty_total"],
            )
            log_event(
                "objective_decomposition",
                stage="predict",
                objective_base=best_penalty["objective_base"],
                lambda_penalty=lambda_penalty,
                penalty_total=best_penalty["penalty_total"],
                term_p13=best_penalty["summary"]["p13_plus_est"],
                term_p14=best_penalty["summary"]["p14_est"],
                term_penalty=best_penalty["penalty_total"],
            )
        else:
            logger.info("objetivo_otimizacao=%.6f", selection_info["best_objective"])

    logger.info(
        "modo_otimizacao=%s penalty_budget=%s min_p13_plus=%s",
        optimization_mode,
        penalty_budget,
        min_p13_plus,
    )
    logger.info(
        "Duplos escolhidos (objetivo=%s): %s (jogos=%s)",
        objective,
        duplo_indices,
        [idx + 1 for idx in duplo_indices],
    )
    if selection_info["top_candidates"]:
        logger.info("Top-%s pares de duplos:", len(selection_info["top_candidates"]))
        for rank, candidate in enumerate(selection_info["top_candidates"], start=1):
            summary = candidate["summary"]
            pair_jogos = tuple(index + 1 for index in candidate["pair"])
            secos_fracos_top2_jogos = tuple(index + 1 for index in candidate["seco_fraco_top2_games"])
            logger.info(
                "rank=%s pair=%s pair_jogos=%s P13_exact=%.6f P13_plus=%.6f P14=%.6f penalty=%.6f frag=%.6f secos=%.6f placebo=%.6f objective=%.6f secos_fracos_top2_restantes=%s",
                rank,
                candidate["pair"],
                pair_jogos,
                summary["p13_exact_est"],
                summary["p13_plus_est"],
                summary["p14_est"],
                candidate["penalty_total"],
                candidate["penalty_fragilidade"],
                candidate["penalty_secos_fracos"],
                candidate["penalty_placebo"],
                candidate["objective"],
                secos_fracos_top2_jogos,
            )
            log_event(
                "duplo_candidate_ranked",
                stage="predict",
                level="DEBUG",
                rank=rank,
                pair=candidate["pair"],
                games=pair_jogos,
                P13_exact=summary["p13_exact_est"],
                P13_plus=summary["p13_plus_est"],
                P14=summary["p14_est"],
                penalty_total=candidate["penalty_total"],
                frag=candidate["penalty_fragilidade"],
                secos=candidate["penalty_secos_fracos"],
                placebo=candidate["penalty_placebo"],
                objective=candidate["objective"],
                secos_fracos_top2_restantes=secos_fracos_top2_jogos,
            )
    if selection_info["pareto_frontier"]:
        logger.info("Pareto frontier (P13_plus vs penalty):")
        for candidate in selection_info["pareto_frontier"]:
            summary = candidate["summary"]
            pair_jogos = tuple(index + 1 for index in candidate["pair"])
            secos_fracos_top2_jogos = tuple(index + 1 for index in candidate["seco_fraco_top2_games"])
            logger.info(
                "pareto pair=%s pair_jogos=%s P13_plus=%.6f penalty=%.6f frag=%.6f secos=%.6f placebo=%.6f secos_fracos_top2_restantes=%s",
                candidate["pair"],
                pair_jogos,
                summary["p13_plus_est"],
                candidate["penalty_total"],
                candidate["penalty_fragilidade"],
                candidate["penalty_secos_fracos"],
                candidate["penalty_placebo"],
                secos_fracos_top2_jogos,
            )
            log_event(
                "pareto_point",
                stage="predict",
                level="DEBUG",
                pair=candidate["pair"],
                games=pair_jogos,
                P13_plus=summary["p13_plus_est"],
                penalty_total=candidate["penalty_total"],
                frag=candidate["penalty_fragilidade"],
                secos=candidate["penalty_secos_fracos"],
                placebo=candidate["penalty_placebo"],
                secos_fracos_top2_restantes=secos_fracos_top2_jogos,
            )
    return picks, {
        "summary": summary,
        "pulverization_metrics": pulverization_metrics,
        "duplo_indices": duplo_indices,
        "best_pair": selection_info["best_pair"],
        "decision_table": decision_table,
        "warnings": warnings,
        "contrarian_count": contrarian_count,
        "top_candidates": selection_info["top_candidates"],
        "pareto_frontier": selection_info["pareto_frontier"],
        "tie_breaker_used": selection_info.get("tie_breaker_used"),
    }


def save_card(rows: list[dict], picks: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["Jogo", "Mandante", "Visitante", "Palpite"])
        for idx, pick in enumerate(picks):
            jogo = rows[idx]["Jogo"]
            mandante = rows[idx]["Mandante"]
            visitante = rows[idx]["Visitante"]
            writer.writerow([jogo, mandante, visitante, pick])


def save_alerts(run_dir: Path, decision_rows: list[dict]) -> None:
    l1_high_games = [row["game"] for row in decision_rows if row["l1_delta_high"]]
    secos_fracos_top2_games = [row["game"] for row in decision_rows if row["seco_fraco_top2"]]
    coin_toss_games = [row["game"] for row in decision_rows if row["coin_toss"]]
    candidatos_duplo = sorted(set(l1_high_games + secos_fracos_top2_games + coin_toss_games))
    lines = [
        "# Alerts",
        "",
        "## L1 delta alto",
        ", ".join(str(game) for game in l1_high_games) if l1_high_games else "nenhum",
        "",
        "## Secos fracos top2",
        ", ".join(str(game) for game in secos_fracos_top2_games) if secos_fracos_top2_games else "nenhum",
        "",
        "## Coin toss",
        ", ".join(str(game) for game in coin_toss_games) if coin_toss_games else "nenhum",
        "",
        "## Candidatos a virar duplo",
        ", ".join(str(game) for game in candidatos_duplo) if candidatos_duplo else "nenhum",
        "",
    ]
    (run_dir / "alerts.md").write_text("\n".join(lines), encoding="utf-8")


def compute_file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def save_run_manifest(payload: dict) -> None:
    save_json("manifest.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerador de cartao Loteca.")
    parser.add_argument("--objective", choices=["p13", "p13plus", "mix"], default=OBJECTIVE_MODE)
    parser.add_argument("--alpha", type=float, default=OBJECTIVE_ALPHA)
    parser.add_argument("--lambda", dest="lambda_penalty", type=float, default=PULVERIZATION_LAMBDA)
    parser.add_argument(
        "--mode",
        choices=["agressivo", "robusto"],
        default=OPTIMIZATION_MODE,
        help="Define o modo de otimizacao (agressivo=mais P13+, robusto=menos fragilidade).",
    )
    parser.add_argument(
        "--penalty-budget",
        type=float,
        default=None,
        help="Orcamento maximo de penalidade (override do modo agressivo).",
    )
    parser.add_argument(
        "--min-p13-plus",
        type=float,
        default=None,
        help="P13+ minimo aceitavel (override do modo robusto).",
    )
    parser.add_argument("--seco-fraco-th", type=float, default=SECO_FRACO_TH)
    parser.add_argument("--seco-top2-th", type=float, default=SECO_FRACO_TOP2_TH)
    parser.add_argument("--l1-th", type=float, default=L1_DELTA_TH)
    parser.add_argument("--l1-th-high", type=float, default=L1_DELTA_TH_HIGH)
    parser.add_argument("--l1-th-low", type=float, default=L1_DELTA_TH_LOW)
    parser.add_argument("--seco-flip-margin-th", type=float, default=SECO_FLIP_MARGIN_TH)
    parser.add_argument("--seco-flip-gap-min", type=float, default=SECO_FLIP_GAP_MIN)
    parser.add_argument("--topk-pairs", type=int, default=DUPLO_TOPK_PAIRS)
    return parser.parse_args()


def predict() -> None:
    setup_logging()
    args = parse_args()
    model = load_model(MODEL_PATH)
    rows = load_next_contest(NEXT_CONTEST_PATH)
    features, market_probs = build_features(rows)
    probabilities = predict_probabilities(model, features)
    biased_probabilities = apply_antipulverization_bias(
        probabilities,
        market_probs,
        ANTI_PULVERIZATION_BIAS,
    )
    penalty_budget = args.penalty_budget
    min_p13_plus = args.min_p13_plus
    if args.mode == "agressivo" and penalty_budget is None:
        penalty_budget = PULVERIZATION_PENALTY_BUDGET_AGRESSIVE
    if args.mode == "robusto" and min_p13_plus is None:
        min_p13_plus = ROBUST_MIN_P13_PLUS
    if args.mode == "robusto" and penalty_budget is None:
        penalty_budget = PULVERIZATION_PENALTY_BUDGET_ROBUSTO
    picks, summary_payload = build_card(
        rows,
        biased_probabilities,
        market_probs,
        objective=args.objective,
        objective_alpha=args.alpha,
        lambda_penalty=args.lambda_penalty,
        penalty_weights=PULVERIZATION_PENALTY_WEIGHTS,
        seco_fraco_th=args.seco_fraco_th,
        seco_top2_th=args.seco_top2_th,
        l1_delta_th=args.l1_th,
        l1_delta_th_high=args.l1_th_high,
        l1_delta_th_low=args.l1_th_low,
        seco_flip_margin_th=args.seco_flip_margin_th,
        seco_flip_gap_min=args.seco_flip_gap_min,
        top_k_pairs=args.topk_pairs,
        optimization_mode=args.mode,
        penalty_budget=penalty_budget,
        min_p13_plus=min_p13_plus,
        risk_weights=RISK_WEIGHTS,
        duplo_value_alpha=DUPLO_VALUE_ALPHA,
        market_penalty_weight=PULVERIZATION_MARKET_PENALTY_WEIGHT,
        contrarian_bonus_weight=PULVERIZATION_CONTRARIAN_BONUS_WEIGHT,
        risk_top_k_games=DUPLO_RISK_TOPK_GAMES,
        market_favorite_mid_low=MARKET_FAVORITE_MID_LOW,
        market_favorite_mid_high=MARKET_FAVORITE_MID_HIGH,
        market_favorite_high_th=MARKET_FAVORITE_HIGH_TH,
        market_favorite_flip_th=MARKET_FAVORITE_FLIP_TH,
        contrarian_min_p=CONTRARIAN_MIN_P,
        entropy_coin_toss_th=ENTROPY_COIN_TOSS_TH,
        margin_coin_toss_th=MARGIN_COIN_TOSS_TH,
        seco_min_acerto_th=SECO_MIN_ACERTO_TH,
    )
    run_dir = get_context().run_dir
    card_path = OUTPUT_CARD_PATH
    save_card(rows, picks, card_path)
    logger.info("Cartao salvo em %s", card_path)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": get_context().run_id,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "seed": SEED,
        "objective": args.objective,
        "objective_alpha": args.alpha,
        "lambda_penalty": args.lambda_penalty,
        "optimization_mode": args.mode,
        "penalty_budget": penalty_budget,
        "min_p13_plus": min_p13_plus,
        "seco_fraco_th": args.seco_fraco_th,
        "seco_top2_th": args.seco_top2_th,
        "l1_delta_th": args.l1_th,
        "l1_delta_th_high": args.l1_th_high,
        "l1_delta_th_low": args.l1_th_low,
        "seco_flip_margin_th": args.seco_flip_margin_th,
        "seco_flip_gap_min": args.seco_flip_gap_min,
        "topk_pairs": args.topk_pairs,
        "dataset_hash": compute_file_hash(NEXT_CONTEST_PATH),
        "model_hash": compute_file_hash(MODEL_PATH),
        "duplos": summary_payload["duplo_indices"],
        "best_pair": summary_payload["best_pair"],
        "summary": summary_payload["summary"],
        "pulverization_metrics": summary_payload["pulverization_metrics"],
        "anti_pulverization_bias": ANTI_PULVERIZATION_BIAS,
        "config": {
            "duplo_count": DUPLO_COUNT,
            "objective_alpha": args.alpha,
            "lambda_penalty": args.lambda_penalty,
            "seco_fraco_th": args.seco_fraco_th,
            "seco_top2_th": args.seco_top2_th,
            "l1_delta_th": args.l1_th,
            "l1_delta_th_high": args.l1_th_high,
            "l1_delta_th_low": args.l1_th_low,
            "seco_flip_margin_th": args.seco_flip_margin_th,
            "seco_flip_gap_min": args.seco_flip_gap_min,
            "seco_min_acerto_th": SECO_MIN_ACERTO_TH,
            "risk_weights": RISK_WEIGHTS,
            "risk_top_k_games": DUPLO_RISK_TOPK_GAMES,
            "duplo_value_alpha": DUPLO_VALUE_ALPHA,
            "market_penalty_weight": PULVERIZATION_MARKET_PENALTY_WEIGHT,
            "contrarian_bonus_weight": PULVERIZATION_CONTRARIAN_BONUS_WEIGHT,
            "market_favorite_mid_low": MARKET_FAVORITE_MID_LOW,
            "market_favorite_mid_high": MARKET_FAVORITE_MID_HIGH,
            "market_favorite_high_th": MARKET_FAVORITE_HIGH_TH,
            "market_favorite_flip_th": MARKET_FAVORITE_FLIP_TH,
            "contrarian_min_p": CONTRARIAN_MIN_P,
            "entropy_coin_toss_th": ENTROPY_COIN_TOSS_TH,
            "margin_coin_toss_th": MARGIN_COIN_TOSS_TH,
            "anti_pulverization_bias": ANTI_PULVERIZATION_BIAS,
        },
    }
    save_run_manifest(manifest)
    logger.info("Manifest salvo em %s", run_dir / "manifest.json")
    decision_rows = summary_payload["decision_table"]
    save_csv(
        "decision_table.csv",
        [
            "game",
            "home",
            "away",
            "p1",
            "pX",
            "p2",
            "market_p1",
            "market_pX",
            "market_p2",
            "pmax",
            "second",
            "third",
            "p_top2",
            "margin",
            "entropy",
            "l1_delta",
            "l1_delta_high",
            "risk",
            "p_acerto",
            "market_max",
            "market_pick",
            "source",
            "pick",
            "is_duplo",
            "seco_fraco",
            "seco_fraco_top2",
            "flip_risk",
            "coin_toss",
        ],
        [
            [
                row["game"],
                row["home"],
                row["away"],
                f"{row['p1']:.6f}",
                f"{row['pX']:.6f}",
                f"{row['p2']:.6f}",
                f"{row['market_p1']:.6f}",
                f"{row['market_pX']:.6f}",
                f"{row['market_p2']:.6f}",
                f"{row['pmax']:.6f}",
                f"{row['second']:.6f}",
                f"{row['third']:.6f}",
                f"{row['p_top2']:.6f}",
                f"{row['margin']:.6f}",
                f"{row['entropy']:.6f}",
                f"{row['l1_delta']:.6f}",
                row["l1_delta_high"],
                f"{row['risk']:.6f}",
                f"{row['p_acerto']:.6f}",
                f"{row['market_max']:.6f}",
                row["market_pick"],
                row["source"],
                row["pick"],
                row["is_duplo"],
                row["seco_fraco"],
                row["seco_fraco_top2"],
                row["flip_risk"],
                row["coin_toss"],
            ]
            for row in decision_rows
        ],
    )
    save_alerts(run_dir, decision_rows)
    top_candidates = summary_payload["top_candidates"]
    best_objective = top_candidates[0]["objective"] if top_candidates else None
    save_csv(
        "duplo_pairs.csv",
        [
            "pair",
            "games",
            "P13_plus",
            "P13_exact",
            "P12_plus",
            "P14",
            "penalty_total",
            "frag",
            "secos",
            "placebo",
            "penalty_market_follow",
            "contrarian_bonus",
            "objective",
            "delta_vs_best",
        ],
        [
            [
                candidate["pair"],
                tuple(index + 1 for index in candidate["pair"]),
                f"{candidate['summary']['p13_plus_est']:.6f}",
                f"{candidate['summary']['p13_exact_est']:.6f}",
                f"{candidate['summary']['p12_plus_est']:.6f}",
                f"{candidate['summary']['p14_est']:.6f}",
                f"{candidate['penalty_total']:.6f}",
                f"{candidate['penalty_fragilidade']:.6f}",
                f"{candidate['penalty_secos_fracos']:.6f}",
                f"{candidate['penalty_placebo']:.6f}",
                f"{candidate['penalty_market_follow']:.6f}",
                f"{candidate['contrarian_bonus']:.6f}",
                f"{candidate['objective']:.6f}",
                None
                if best_objective is None
                else f"{candidate['objective'] - best_objective:.6f}",
            ]
            for candidate in top_candidates
        ],
    )
    risks = sorted(
        decision_rows,
        key=lambda row: row["risk"],
        reverse=True,
    )[:3]
    final_summary = {
        "best_val_logloss": model.get("training_metrics", {}).get("best_val_logloss"),
        "ece_post": model.get("training_metrics", {}).get("ece_post"),
        "temperature": model.get("temperature"),
        "P13_plus": summary_payload["summary"]["p13_plus_est"],
        "P13_exact": summary_payload["summary"]["p13_exact_est"],
        "P12_plus": summary_payload["summary"]["p12_plus_est"],
        "P14": summary_payload["summary"]["p14_est"],
        "fragilidade_secos": summary_payload["pulverization_metrics"]["fragilidade_secos"],
        "pulv_secos_fracos": summary_payload["pulverization_metrics"]["secos_fracos"],
        "pulv_secos_fracos_top2": summary_payload["pulverization_metrics"]["secos_fracos_top2"],
        "placebo_duplos": summary_payload["pulverization_metrics"]["placebo_duplos"],
        "duplos_escolhidos": summary_payload["duplo_indices"],
        "tie_breaker_used": summary_payload["tie_breaker_used"],
        "contrarian_count": summary_payload["contrarian_count"],
        "top_riscos": [
            {
                "game": risk["game"],
                "risk": risk["risk"],
                "entropy": risk["entropy"],
                "l1_delta": risk["l1_delta"],
                "pmax": risk["pmax"],
            }
            for risk in risks
        ],
    }
    metrics_path = run_dir / "metrics.json"
    metrics_payload = {}
    if metrics_path.exists():
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_payload["predict"] = final_summary
    save_json("metrics.json", metrics_payload)
    logger.info("Resumo executivo:")
    for key, value in final_summary.items():
        logger.info("%s=%s", key, value)
    log_event(
        "final_summary",
        stage="predict",
        best_val_logloss=final_summary["best_val_logloss"],
        ece_post=final_summary["ece_post"],
        temperature=final_summary["temperature"],
        P13_plus=final_summary["P13_plus"],
        P13_exact=final_summary["P13_exact"],
        P12_plus=final_summary["P12_plus"],
        P14=final_summary["P14"],
        fragilidade_secos=final_summary["fragilidade_secos"],
        pulv_secos_fracos=final_summary["pulv_secos_fracos"],
        pulv_secos_fracos_top2=final_summary["pulv_secos_fracos_top2"],
        placebo_duplos=final_summary["placebo_duplos"],
        duplos_escolhidos=final_summary["duplos_escolhidos"],
        tie_breaker_used=final_summary["tie_breaker_used"],
        contrarian_count=final_summary["contrarian_count"],
        top_riscos=final_summary["top_riscos"],
    )


if __name__ == "__main__":
    predict()
