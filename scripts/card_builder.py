import logging
import math
from typing import Iterable, List


logger = logging.getLogger("loteca_antipulverizacao")


def entropy(probabilities: Iterable[float]) -> float:
    total = 0.0
    for prob in probabilities:
        if prob > 0:
            total -= prob * math.log(prob)
    return total


def compute_risk(
    p_acerto: float,
    margin: float,
    entropy_value: float,
    weights: tuple[float, float, float],
) -> float:
    weight_p, weight_margin, weight_entropy = weights
    return weight_p * (1.0 - p_acerto) + weight_margin * (1.0 - margin) + weight_entropy * entropy_value


def score_game(probabilities: list[float]) -> float:
    sorted_probs = sorted(probabilities, reverse=True)
    pmax = sorted_probs[0]
    second = sorted_probs[1]
    third = sorted_probs[2]
    p_top2 = pmax + second
    return (1.0 - pmax) * second * p_top2 * (1.0 - third)


def select_duplos_heuristic(probabilities: list[list[float]], duplo_count: int) -> List[int]:
    scores = [score_game(prob) for prob in probabilities]
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return ranked[:duplo_count]


def top_two(probabilities: list[float]) -> tuple[float, float]:
    sorted_probs = sorted(probabilities, reverse=True)
    return sorted_probs[0], sorted_probs[1]


def choose_duplo_pick(
    probabilities: list[float],
    market_probabilities: list[float] | None,
    value_alpha: float = 0.0,
) -> dict:
    outcomes = ["1", "X", "2"]
    pairs = [(0, 1), (1, 2), (0, 2)]
    best = None
    for pair in pairs:
        p_cover = probabilities[pair[0]] + probabilities[pair[1]]
        popularity = p_cover
        if market_probabilities is not None:
            popularity = market_probabilities[pair[0]] + market_probabilities[pair[1]]
        value = p_cover - value_alpha * popularity
        candidate = {
            "pair": pair,
            "pick": f"{outcomes[pair[0]]}/{outcomes[pair[1]]}",
            "p_cover": p_cover,
            "popularity": popularity,
            "value": value,
        }
        if best is None:
            best = candidate
        else:
            if value > best["value"]:
                best = candidate
            elif abs(value - best["value"]) <= 1e-9:
                if p_cover > best["p_cover"]:
                    best = candidate
                elif abs(p_cover - best["p_cover"]) <= 1e-9 and pair < best["pair"]:
                    best = candidate
    return best


def compute_p_acertos(
    probabilities: list[list[float]],
    duplo_indices: Iterable[int],
    market_probabilities: list[list[float]] | None = None,
    duplo_value_alpha: float = 0.0,
) -> list[float]:
    duplo_set = set(duplo_indices)
    p_acertos = []
    for idx, probs in enumerate(probabilities):
        if idx in duplo_set:
            market_probs = market_probabilities[idx] if market_probabilities is not None else None
            duplo = choose_duplo_pick(probs, market_probs, value_alpha=duplo_value_alpha)
            p_acertos.append(duplo["p_cover"])
        else:
            pmax, _second = top_two(probs)
            p_acertos.append(pmax)
    return p_acertos


def compute_pulverization_metrics(
    probabilities: list[list[float]],
    duplo_indices: Iterable[int],
    p_acertos: list[float] | None = None,
    seco_pmax_threshold: float = 0.45,
    duplo_second_threshold: float = 0.20,
    seco_top2_threshold: float = 0.75,
    market_probabilities: list[list[float]] | None = None,
    duplo_value_alpha: float = 0.0,
) -> dict:
    duplo_set = set(duplo_indices)
    secos_fracos = 0
    secos_fracos_top2 = 0
    fragilidade_total = 0.0
    placebo_duplos = 0
    fragilidade_vals = p_acertos
    for idx, probs in enumerate(probabilities):
        if idx in duplo_set:
            market_probs = market_probabilities[idx] if market_probabilities is not None else None
            duplo = choose_duplo_pick(probs, market_probs, value_alpha=duplo_value_alpha)
            covered_probs = [probs[duplo["pair"][0]], probs[duplo["pair"][1]]]
            if min(covered_probs) < duplo_second_threshold:
                placebo_duplos += 1
        else:
            pmax, second = top_two(probs)
            p_top2 = pmax + second
            if pmax < seco_pmax_threshold:
                secos_fracos += 1
                if p_top2 >= seco_top2_threshold:
                    secos_fracos_top2 += 1
            fragilidade_base = pmax
            if fragilidade_vals is not None:
                fragilidade_base = fragilidade_vals[idx]
            fragilidade_total += 1.0 - fragilidade_base
    return {
        "secos_fracos": secos_fracos,
        "secos_fracos_top2": secos_fracos_top2,
        "placebo_duplos": placebo_duplos,
        "fragilidade_secos": fragilidade_total,
    }


def select_best_duplos(
    probabilities: list[list[float]],
    market_probabilities: list[list[float]] | None = None,
    must_include_games: Iterable[int] | None = None,
    duplo_count: int = 2,
    lambda_penalty: float = 0.0,
    penalty_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    market_penalty_weight: float = 0.0,
    contrarian_bonus_weight: float = 0.0,
    objective: str = "p13plus",
    objective_alpha: float = 0.4,
    seco_pmax_threshold: float = 0.45,
    seco_top2_threshold: float = 0.75,
    top_k_pairs: int = 10,
    risk_weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
    risk_top_k_games: int = 6,
    duplo_value_alpha: float = 0.0,
    market_favorite_mid_low: float = 0.45,
    market_favorite_mid_high: float = 0.60,
    market_favorite_high_th: float = 0.70,
    contrarian_l1_th: float = 0.12,
    contrarian_count_l1_th: float | None = None,
    contrarian_min_p: float = 0.35,
    entropy_coin_toss_th: float = 1.08,
    margin_coin_toss_th: float = 0.05,
    penalty_budget: float | None = None,
    mode: str = "agressivo",
    min_p13_plus: float | None = None,
    eps: float = 1e-6,
) -> dict:
    if duplo_count != 2:
        return {
            "duplo_indices": select_duplos_heuristic(probabilities, duplo_count),
            "best_pair": None,
            "runner_up_pair": None,
            "best_summary": None,
            "runner_up_summary": None,
            "best_metrics": None,
            "runner_up_metrics": None,
            "best_objective": None,
            "runner_up_objective": None,
            "delta_p13_plus": None,
            "top_candidates": [],
        }

    best = None
    runner_up = None
    top_candidates: list[dict] = []
    n_games = len(probabilities)

    def tie_breaker_key(candidate: dict) -> tuple[float, float, float, float, float]:
        metrics = candidate["metrics"]
        return (
            metrics["placebo_duplos"],
            metrics["secos_fracos_top2"],
            metrics["fragilidade_secos"],
            -candidate["contrarian_bonus"],
            -candidate["summary"]["p12_plus_est"],
        )

    def compare_aggressive(candidate: dict, current: dict | None) -> bool:
        if current is None:
            return True
        summary = candidate["summary"]
        current_summary = current["summary"]
        if abs(summary["p13_plus_est"] - current_summary["p13_plus_est"]) > eps:
            return summary["p13_plus_est"] > current_summary["p13_plus_est"]
        if abs(summary["p13_exact_est"] - current_summary["p13_exact_est"]) > eps:
            return summary["p13_exact_est"] > current_summary["p13_exact_est"]
        if abs(summary["p12_plus_est"] - current_summary["p12_plus_est"]) > eps:
            return summary["p12_plus_est"] > current_summary["p12_plus_est"]
        if abs(candidate["penalty_fragilidade"] - current["penalty_fragilidade"]) > eps:
            return candidate["penalty_fragilidade"] < current["penalty_fragilidade"]
        if abs(candidate["penalty_secos_fracos"] - current["penalty_secos_fracos"]) > eps:
            return candidate["penalty_secos_fracos"] < current["penalty_secos_fracos"]
        if abs(candidate["penalty_placebo"] - current["penalty_placebo"]) > eps:
            return candidate["penalty_placebo"] < current["penalty_placebo"]
        if abs(candidate["penalty_market_follow"] - current["penalty_market_follow"]) > eps:
            return candidate["penalty_market_follow"] < current["penalty_market_follow"]
        if abs(candidate["contrarian_bonus"] - current["contrarian_bonus"]) > eps:
            return candidate["contrarian_bonus"] > current["contrarian_bonus"]
        if tie_breaker_key(candidate) != tie_breaker_key(current):
            return tie_breaker_key(candidate) < tie_breaker_key(current)
        return candidate["pair"] < current["pair"]

    def compare_robust(candidate: dict, current: dict | None) -> bool:
        if current is None:
            return True
        if abs(candidate["penalty_fragilidade"] - current["penalty_fragilidade"]) > eps:
            return candidate["penalty_fragilidade"] < current["penalty_fragilidade"]
        if abs(candidate["penalty_total"] - current["penalty_total"]) > eps:
            return candidate["penalty_total"] < current["penalty_total"]
        return compare_aggressive(candidate, current)

    base_risks = []
    seco_fraco_top2_games = []
    coin_toss_games = []
    l1_high_games = list(must_include_games) if must_include_games is not None else []
    for idx, probs in enumerate(probabilities):
        pmax, second = top_two(probs)
        margin = pmax - second
        p_top2 = pmax + second
        risk = compute_risk(pmax, margin, entropy(probs), risk_weights)
        base_risks.append((risk, idx))
        if pmax < seco_pmax_threshold and p_top2 >= seco_top2_threshold:
            seco_fraco_top2_games.append(idx)
        if entropy(probs) > entropy_coin_toss_th and margin < margin_coin_toss_th:
            coin_toss_games.append(idx)

    logger.info(
        "select_best_duplos grupos: seco_fraco_top2_games=%s coin_toss_games=%s l1_high_games=%s",
        seco_fraco_top2_games,
        coin_toss_games,
        l1_high_games,
    )

    base_risks.sort(reverse=True)
    candidate_games = [idx for _risk, idx in base_risks[: max(2, risk_top_k_games)]]
    candidate_games = list(
        dict.fromkeys(candidate_games + seco_fraco_top2_games + coin_toss_games + l1_high_games)
    )
    if len(candidate_games) < 2:
        candidate_games = list(range(n_games))

    strict_seco_fraco = mode == "agressivo" and len(seco_fraco_top2_games) >= 2
    must_cover_groups = []
    if not strict_seco_fraco:
        if seco_fraco_top2_games:
            must_cover_groups.append(set(seco_fraco_top2_games))
        if coin_toss_games:
            must_cover_groups.append(set(coin_toss_games))
    if l1_high_games:
        must_cover_groups.append(set(l1_high_games))

    logger.info(
        "select_best_duplos must_cover_groups=%s candidate_games(len=%s)=%s",
        must_cover_groups,
        len(candidate_games),
        candidate_games,
    )

    contrarian_count_l1_th = contrarian_l1_th if contrarian_count_l1_th is None else contrarian_count_l1_th
    for idx_i, i in enumerate(candidate_games[:-1]):
        for j in candidate_games[idx_i + 1 :]:
            duplo_indices = [i, j]
            if strict_seco_fraco and not (i in seco_fraco_top2_games or j in seco_fraco_top2_games):
                continue
            if must_cover_groups:
                pair_set = set(duplo_indices)
                if any(not (pair_set & group) for group in must_cover_groups):
                    continue
            p_acertos = compute_p_acertos(
                probabilities,
                duplo_indices,
                market_probabilities=market_probabilities,
                duplo_value_alpha=duplo_value_alpha,
            )
            summary = summarize_card(p_acertos)
            metrics = compute_pulverization_metrics(
                probabilities,
                duplo_indices,
                p_acertos=p_acertos,
                seco_pmax_threshold=seco_pmax_threshold,
                seco_top2_threshold=seco_top2_threshold,
                market_probabilities=market_probabilities,
                duplo_value_alpha=duplo_value_alpha,
            )
            frag_weight, secos_weight, placebo_weight = penalty_weights
            penalty = (
                frag_weight * metrics["fragilidade_secos"]
                + secos_weight * (metrics["secos_fracos"] + metrics["secos_fracos_top2"])
                + placebo_weight * metrics["placebo_duplos"]
            )
            objective_base = summary["p13_plus_est"]
            if objective == "p13":
                objective_base = summary["p13_exact_est"]
            elif objective == "mix":
                objective_base = summary["p13_exact_est"] + objective_alpha * summary["p14_est"]
            market_follow_penalty = 0.0
            contrarian_bonus = 0.0
            market_follow_count = 0
            contrarian_count = 0
            if market_probabilities is not None:
                for idx, probs in enumerate(probabilities):
                    if idx in duplo_indices:
                        continue
                    market_probs = market_probabilities[idx]
                    pick_index = max(range(len(probs)), key=probs.__getitem__)
                    market_index = max(range(len(market_probs)), key=market_probs.__getitem__)
                    market_max = market_probs[market_index]
                    p_acerto = probs[pick_index]
                    l1_delta = sum(abs(model - market) for model, market in zip(probs, market_probs))
                    if pick_index == market_index:
                        market_follow_count += 1
                        if market_max > market_favorite_high_th:
                            market_follow_penalty += 0.25
                        elif market_favorite_mid_low <= market_max <= market_favorite_mid_high:
                            market_follow_penalty += 1.0
                        elif market_max >= market_favorite_mid_high:
                            market_follow_penalty += 0.6
                    elif l1_delta >= contrarian_l1_th and p_acerto >= contrarian_min_p:
                        contrarian_bonus += 1.0
                    if l1_delta >= contrarian_count_l1_th and p_acerto >= contrarian_min_p:
                        contrarian_count += 1
            total_penalty = (
                penalty
                + market_penalty_weight * market_follow_penalty
                - contrarian_bonus_weight * contrarian_bonus
            )
            objective_value = objective_base - (lambda_penalty * total_penalty)
            max_l1_delta_secos = 0.0
            if market_probabilities is not None:
                for idx, probs in enumerate(probabilities):
                    if idx in duplo_indices:
                        continue
                    market_probs = market_probabilities[idx]
                    l1_delta = sum(abs(model - market) for model, market in zip(probs, market_probs))
                    max_l1_delta_secos = max(max_l1_delta_secos, l1_delta)
            seco_fraco_top2_remaining = []
            for idx, probs in enumerate(probabilities):
                if idx in duplo_indices:
                    continue
                pmax, second = top_two(probs)
                p_top2 = pmax + second
                if pmax < seco_pmax_threshold and p_top2 >= seco_top2_threshold:
                    seco_fraco_top2_remaining.append(idx)
            candidate = {
                "pair": (i, j),
                "summary": summary,
                "metrics": metrics,
                "objective_base": objective_base,
                "objective": objective_value,
                "penalty_total": total_penalty,
                "penalty_fragilidade": frag_weight * metrics["fragilidade_secos"],
                "penalty_secos_fracos": secos_weight * (metrics["secos_fracos"] + metrics["secos_fracos_top2"]),
                "penalty_placebo": placebo_weight * metrics["placebo_duplos"],
                "penalty_market_follow": market_penalty_weight * market_follow_penalty,
                "contrarian_bonus": contrarian_bonus,
                "market_follow_count": market_follow_count,
                "contrarian_count": contrarian_count,
                "max_l1_delta_secos": max_l1_delta_secos,
                "seco_fraco_top2_games": seco_fraco_top2_remaining,
                "tie_breaker": {
                    "secos_fracos_top2": metrics["secos_fracos_top2"],
                    "fragilidade_secos": metrics["fragilidade_secos"],
                    "max_l1_delta_secos": max_l1_delta_secos,
                    "p14_est": summary["p14_est"],
                },
            }
            top_candidates.append(candidate)

    delta_p13_plus = None
    feasible = top_candidates
    if mode == "robusto":
        if min_p13_plus is not None:
            feasible = [cand for cand in top_candidates if cand["summary"]["p13_plus_est"] >= min_p13_plus]
        if not feasible:
            feasible = top_candidates
        best = None
        for candidate in feasible:
            if compare_robust(candidate, best):
                best = candidate
        runner_up = None
        for candidate in feasible:
            if candidate is best:
                continue
            if compare_robust(candidate, runner_up):
                runner_up = candidate
    else:
        if penalty_budget is not None:
            feasible = [cand for cand in top_candidates if cand["penalty_total"] <= penalty_budget]
        if not feasible:
            feasible = top_candidates
        best = None
        for candidate in feasible:
            if compare_aggressive(candidate, best):
                best = candidate
        runner_up = None
        for candidate in feasible:
            if candidate is best:
                continue
            if compare_aggressive(candidate, runner_up):
                runner_up = candidate

    if best and runner_up:
        delta_p13_plus = best["summary"]["p13_plus_est"] - runner_up["summary"]["p13_plus_est"]
    tie_breaker_used = None
    if best and runner_up:
        if abs(delta_p13_plus) <= eps:
            primary_tie = (
                abs(best["summary"]["p13_exact_est"] - runner_up["summary"]["p13_exact_est"]) <= eps
                and abs(best["penalty_fragilidade"] - runner_up["penalty_fragilidade"]) <= eps
                and abs(best["penalty_secos_fracos"] - runner_up["penalty_secos_fracos"]) <= eps
                and abs(best["penalty_placebo"] - runner_up["penalty_placebo"]) <= eps
            )
            if primary_tie and tie_breaker_key(best) != tie_breaker_key(runner_up):
                tie_breaker_used = "fragilidade"

    def pareto_frontier(candidates: list[dict]) -> list[dict]:
        frontier = []
        for candidate in candidates:
            dominated = False
            for other in candidates:
                if other is candidate:
                    continue
                if (
                    other["summary"]["p13_plus_est"] >= candidate["summary"]["p13_plus_est"] - eps
                    and other["penalty_total"] <= candidate["penalty_total"] + eps
                    and (
                        other["summary"]["p13_plus_est"] > candidate["summary"]["p13_plus_est"] + eps
                        or other["penalty_total"] < candidate["penalty_total"] - eps
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)
        frontier.sort(
            key=lambda item: (item["penalty_total"], -item["summary"]["p13_plus_est"], item["pair"])
        )
        return frontier

    all_candidates = list(top_candidates)
    sorted_candidates = sorted(
        all_candidates,
        key=lambda item: (item["summary"]["p13_plus_est"], -item["summary"]["p12_plus_est"], -item["penalty_total"]),
        reverse=True,
    )
    top_candidates = sorted_candidates[: max(1, top_k_pairs)]
    frontier = pareto_frontier(all_candidates)
    return {
        "duplo_indices": list(best["pair"]) if best else [],
        "best_pair": best["pair"] if best else None,
        "runner_up_pair": runner_up["pair"] if runner_up else None,
        "best_summary": best["summary"] if best else None,
        "runner_up_summary": runner_up["summary"] if runner_up else None,
        "best_metrics": best["metrics"] if best else None,
        "runner_up_metrics": runner_up["metrics"] if runner_up else None,
        "best_objective": best["objective"] if best else None,
        "runner_up_objective": runner_up["objective"] if runner_up else None,
        "delta_p13_plus": delta_p13_plus,
        "top_candidates": top_candidates,
        "pareto_frontier": frontier,
        "tie_breaker_used": tie_breaker_used,
    }


def build_error_distribution(probabilities: list[float]) -> list[float]:
    dp = [1.0] + [0.0] * len(probabilities)
    for prob in probabilities:
        q = 1.0 - prob
        next_dp = [0.0] * len(dp)
        for errors in range(len(dp)):
            if dp[errors] == 0:
                continue
            next_dp[errors] += dp[errors] * prob
            if errors + 1 < len(dp):
                next_dp[errors + 1] += dp[errors] * q
        dp = next_dp
    return dp


def summarize_card(p_acertos: list[float]) -> dict:
    p14_est = math.prod(p_acertos)
    error_distribution = build_error_distribution(p_acertos)
    p13_exact_est = error_distribution[1] if len(error_distribution) > 1 else 0.0
    p13_plus_est = sum(error_distribution[:2])
    p12_plus_est = sum(error_distribution[:3])
    expected = sum(p_acertos)
    variance = sum(prob * (1.0 - prob) for prob in p_acertos)
    return {
        "p14_est": p14_est,
        "p13_exact_est": p13_exact_est,
        "p13_plus_est": p13_plus_est,
        "p12_plus_est": p12_plus_est,
        "expected": expected,
        "variance": variance,
    }


def summarize_sensitivity(p_acertos: list[float], downshift: float = 0.02) -> dict:
    adjusted = [
        max(0.0, min(1.0, (1.0 - downshift) * prob + downshift / 3.0)) for prob in p_acertos
    ]
    summary_down = summarize_card(adjusted)
    return {
        "p13_exact_est_down": summary_down["p13_exact_est"],
        "p13_plus_est_down": summary_down["p13_plus_est"],
        "p12_plus_est_down": summary_down["p12_plus_est"],
    }


def log_card_summary(
    summary: dict,
    secos: int,
    duplos: int,
    pulverization_metrics: dict,
    sensitivity_summary: dict,
    sensitivity_shifts: list[float],
) -> None:
    logger.info("Resumo antipulverizacao: %s secos, %s duplos", secos, duplos)
    logger.info("P14_est=%.6f", summary["p14_est"])
    logger.info("P13_exact_est=%.6f", summary["p13_exact_est"])
    logger.info("P13_plus_est=%.6f", summary["p13_plus_est"])
    logger.info("P12_plus_est=%.6f", summary["p12_plus_est"])
    logger.info("E_acertos=%.3f", summary["expected"])
    logger.info("Var_acertos=%.3f", summary["variance"])
    logger.info("pulv_secos_fracos=%s", pulverization_metrics["secos_fracos"])
    logger.info("pulv_secos_fracos_top2=%s", pulverization_metrics["secos_fracos_top2"])
    logger.info("pulv_placebo_duplos=%s", pulverization_metrics["placebo_duplos"])
    logger.info("fragilidade_secos=%.3f", pulverization_metrics["fragilidade_secos"])
    for shift in sensitivity_shifts:
        summary_down = sensitivity_summary[str(shift)]
        logger.info(
            "P13_exact_est_down%.0fpct=%.6f P13_plus_est_down%.0fpct=%.6f P12_plus_est_down%.0fpct=%.6f",
            shift * 100,
            summary_down["p13_exact_est_down"],
            shift * 100,
            summary_down["p13_plus_est_down"],
            shift * 100,
            summary_down["p12_plus_est_down"],
        )
        logger.info(
            "delta_sensibilidade_%.0fpct=%.6f",
            shift * 100,
            summary["p13_plus_est"] - summary_down["p13_plus_est_down"],
        )


def log_card_table(
    rows: list[dict],
    picks: list[str],
    p_acertos: list[float],
    scores: list[float],
) -> None:
    header = "Jogo | Mandante x Visitante | p1 | pX | p2 | pick | p_acerto | duplo? | source | score"
    separator = "-" * len(header)
    logger.debug("%s", header)
    logger.debug("%s", separator)
    for idx, row in enumerate(rows):
        probs = row["probs"]
        logger.debug(
            "%s | %s x %s | %.4f | %.4f | %.4f | %s | %.4f | %s | %s | %.6f",
            row["Jogo"],
            row["Mandante"],
            row["Visitante"],
            probs[0],
            probs[1],
            probs[2],
            picks[idx],
            p_acertos[idx],
            "sim" if row["is_duplo"] else "nao",
            row.get("pick_source", "modelo"),
            scores[idx],
        )
