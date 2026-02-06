import json
import logging
import math
import random
from pathlib import Path

from config import (
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    L2,
    L2_PLATEAU,
    LEARNING_RATE,
    LOG_EVERY,
    EARLY_STOPPING_MIN_DELTA,
    LR_DECAY_EVERY,
    LR_DECAY_GAMMA,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    SEED,
    TRAIN_RATIO,
    CLASS_WEIGHT_MODE,
    VALIDATION_STRATEGY,
)
from scripts.telemetry import log_event, save_json, setup_logging


logger = logging.getLogger("loteca_train")
logger.setLevel(logging.INFO)


def load_training_data(path: Path) -> tuple[list[list[float]], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = [[float(value) for value in row] for row in payload["features"]]
    targets = [int(value) for value in payload["targets"]]
    return features, targets


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_scores = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


def log_loss(probabilities: list[list[float]], targets: list[int]) -> float:
    if not targets:
        return float("nan")
    epsilon = 1e-15
    total = 0.0
    for probs, target in zip(probabilities, targets):
        prob = max(min(probs[target], 1 - epsilon), epsilon)
        total += -math.log(prob)
    return total / len(targets)


def accuracy(probabilities: list[list[float]], targets: list[int]) -> float:
    if not targets:
        return float("nan")
    correct = 0
    for probs, target in zip(probabilities, targets):
        if probs.index(max(probs)) == target:
            correct += 1
    return correct / len(targets)


def weight_norm(weights: list[list[float]]) -> float:
    return math.sqrt(sum(value * value for row in weights for value in row))


def build_logits(weights: list[list[float]], features: list[list[float]]) -> list[list[float]]:
    num_features = len(features[0])
    num_classes = len(weights[0])
    logits = []
    for row in features:
        row_with_bias = row + [1.0]
        logits.append(
            [
                sum(row_with_bias[i] * weights[i][cls] for i in range(num_features + 1))
                for cls in range(num_classes)
            ]
        )
    return logits


def predict_probabilities(weights: list[list[float]], features: list[list[float]], temperature: float = 1.0) -> list[list[float]]:
    logits = build_logits(weights, features)
    scaled = [[value / temperature for value in row] for row in logits]
    return [softmax(row) for row in scaled]


def split_train_validation(
    features: list[list[float]],
    targets: list[int],
    seed: int,
    train_ratio: float,
    strategy: str = "random",
) -> tuple[list[list[float]], list[int], list[list[float]], list[int]]:
    indices = list(range(len(features)))
    if strategy == "random":
        random.Random(seed).shuffle(indices)
    if len(indices) < 2:
        return features, targets, [], []
    split = int(len(indices) * train_ratio)
    if split <= 0:
        split = max(1, len(indices) - 1)
    if split >= len(indices):
        split = len(indices) - 1
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_features = [features[i] for i in train_idx]
    train_targets = [targets[i] for i in train_idx]
    val_features = [features[i] for i in val_idx]
    val_targets = [targets[i] for i in val_idx]
    return train_features, train_targets, val_features, val_targets


def temperature_log_loss(logits: list[list[float]], targets: list[int], temperature: float) -> float:
    probabilities = [softmax([value / temperature for value in row]) for row in logits]
    return log_loss(probabilities, targets)


def find_best_temperature(logits: list[list[float]], targets: list[int]) -> float:
    if not logits or not targets:
        return 1.0

    best_temp = 1.0
    best_loss = float("inf")
    low, high = 0.5, 5.0

    for _ in range(3):
        step = (high - low) / 30
        candidates = [low + step * i for i in range(31)]
        for temp in candidates:
            loss = temperature_log_loss(logits, targets, temp)
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        low = max(0.05, best_temp - step)
        high = best_temp + step

    return best_temp


def compute_class_weights(targets: list[int], mode: str) -> list[float]:
    if not targets:
        return [1.0, 1.0, 1.0]
    counts = [targets.count(idx) for idx in range(max(targets) + 1)]
    total = sum(counts)
    weights = []
    for count in counts:
        if count == 0:
            weights.append(0.0)
            continue
        freq = count / total
        if mode == "inv":
            weight = 1.0 / freq
        elif mode == "sqrt_inv":
            weight = math.sqrt(1.0 / freq)
        else:
            weight = 1.0
        weights.append(weight)
    return weights


def expected_calibration_error(
    probabilities: list[list[float]],
    targets: list[int],
    bins: int = 10,
) -> float:
    if not targets:
        return float("nan")
    bin_counts = [0 for _ in range(bins)]
    bin_confidences = [0.0 for _ in range(bins)]
    bin_correct = [0.0 for _ in range(bins)]
    for probs, target in zip(probabilities, targets):
        confidence = max(probs)
        prediction = probs.index(confidence)
        bin_idx = min(bins - 1, int(confidence * bins))
        bin_counts[bin_idx] += 1
        bin_confidences[bin_idx] += confidence
        bin_correct[bin_idx] += 1.0 if prediction == target else 0.0
    total = sum(bin_counts)
    if total == 0:
        return float("nan")
    ece = 0.0
    for count, conf_sum, correct_sum in zip(bin_counts, bin_confidences, bin_correct):
        if count == 0:
            continue
        avg_conf = conf_sum / count
        avg_acc = correct_sum / count
        ece += abs(avg_acc - avg_conf) * (count / total)
    return ece


def train_softmax_regression(
    features: list[list[float]],
    targets: list[int],
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    l2: float = L2,
    log_every: int = LOG_EVERY,
) -> dict:
    train_features, train_targets, val_features, val_targets = split_train_validation(
        features,
        targets,
        seed=SEED,
        train_ratio=TRAIN_RATIO,
        strategy=VALIDATION_STRATEGY,
    )
    logger.info(
        "validation_split strategy=%s train_size=%s val_size=%s",
        VALIDATION_STRATEGY,
        len(train_features),
        len(val_features),
    )
    log_event(
        "split_info",
        stage="train",
        strategy=VALIDATION_STRATEGY,
        train_size=len(train_features),
        val_size=len(val_features),
    )
    num_samples = len(train_features)
    num_features = len(train_features[0])
    num_classes = max(targets) + 1
    weights = [[0.0 for _ in range(num_classes)] for _ in range(num_features + 1)]
    class_weights = compute_class_weights(train_targets, CLASS_WEIGHT_MODE)
    logger.info("class_weights=%s mode=%s", class_weights, CLASS_WEIGHT_MODE)
    log_event(
        "class_weights",
        stage="train",
        class_weights=class_weights,
        mode=CLASS_WEIGHT_MODE,
    )

    best_weights = None
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    current_lr = learning_rate

    for epoch in range(1, epochs + 1):
        if epoch > 1 and LR_DECAY_EVERY > 0 and epoch % LR_DECAY_EVERY == 0:
            current_lr *= LR_DECAY_GAMMA

        gradients = [[0.0 for _ in range(num_classes)] for _ in range(num_features + 1)]
        for row, target in zip(train_features, train_targets):
            row_with_bias = row + [1.0]
            logits = [
                sum(row_with_bias[i] * weights[i][cls] for i in range(num_features + 1))
                for cls in range(num_classes)
            ]
            probs = softmax(logits)
            sample_weight = class_weights[target] if target < len(class_weights) else 1.0
            for cls in range(num_classes):
                error = (probs[cls] - (1.0 if cls == target else 0.0)) * sample_weight
                for i in range(num_features + 1):
                    gradients[i][cls] += row_with_bias[i] * error

        for i in range(num_features + 1):
            for cls in range(num_classes):
                gradient = gradients[i][cls] / num_samples + l2 * weights[i][cls]
                weights[i][cls] -= current_lr * gradient

        train_probs = predict_probabilities(weights, train_features)
        val_probs = predict_probabilities(weights, val_features)
        train_loss = log_loss(train_probs, train_targets)
        val_loss = log_loss(val_probs, val_targets)

        if val_targets and val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = [row[:] for row in weights]
            bad_epochs = 0
        elif val_targets:
            bad_epochs += 1

        if val_targets and bad_epochs >= LR_DECAY_EVERY and l2 < L2_PLATEAU:
            l2 = L2_PLATEAU
            logger.info("l2_plateau=%.4f epoch=%s", l2, epoch)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            logger.info(
                "epoch=%s lr=%.6f train_logloss=%.5f train_accuracy=%.4f val_logloss=%.5f val_accuracy=%.4f weight_norm=%.4f",
                epoch,
                current_lr,
                train_loss,
                accuracy(train_probs, train_targets),
                val_loss,
                accuracy(val_probs, val_targets),
                weight_norm(weights),
            )
            log_event(
                "epoch_metrics",
                stage="train",
                level="DEBUG",
                epoch=epoch,
                lr=current_lr,
                train_logloss=train_loss,
                train_accuracy=accuracy(train_probs, train_targets),
                val_logloss=val_loss,
                val_accuracy=accuracy(val_probs, val_targets),
                weight_norm=weight_norm(weights),
            )

        if val_targets and bad_epochs >= EARLY_STOPPING_PATIENCE:
            logger.info(
                "early_stopping epoch=%s best_epoch=%s best_val_logloss=%.5f",
                epoch,
                best_epoch,
                best_val_loss,
            )
            log_event(
                "early_stopping",
                stage="train",
                epoch=epoch,
                best_epoch=best_epoch,
                best_val_logloss=best_val_loss,
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA,
            )
            break

    if best_weights is not None:
        weights = best_weights
        logger.info("best_epoch_selected=%s best_val_logloss=%.5f", best_epoch, best_val_loss)
        log_event(
            "best_epoch_selected",
            stage="train",
            best_epoch=best_epoch,
            best_val_logloss=best_val_loss,
        )

    val_logits = build_logits(weights, val_features)
    val_probs_pre = [softmax(row) for row in val_logits]
    val_logloss_pre = log_loss(val_probs_pre, val_targets)
    val_ece_pre = expected_calibration_error(val_probs_pre, val_targets)
    temperature = find_best_temperature(val_logits, val_targets)
    val_probs_post = [softmax([value / temperature for value in row]) for row in val_logits]
    val_logloss_post = log_loss(val_probs_post, val_targets)
    val_ece_post = expected_calibration_error(val_probs_post, val_targets)
    logger.info(
        "temperature_scaling=%.4f val_logloss_pre=%.5f val_logloss_post=%.5f ece_pre=%.5f ece_post=%.5f",
        temperature,
        val_logloss_pre,
        val_logloss_post,
        val_ece_pre,
        val_ece_post,
    )
    log_event(
        "calibration",
        stage="train",
        temperature=temperature,
        val_logloss_pre=val_logloss_pre,
        val_logloss_post=val_logloss_post,
        ece_pre=val_ece_pre,
        ece_post=val_ece_post,
    )

    return {
        "weights": weights,
        "classes": [0, 1, 2],
        "features": [
            "log_odds_1",
            "log_odds_x",
            "log_odds_2",
            "overround",
            "market_entropy",
            "favorite_gap",
        ],
        "temperature": temperature,
        "training_metrics": {
            "best_epoch": best_epoch,
            "best_val_logloss": best_val_loss,
            "val_logloss_pre": val_logloss_pre,
            "val_logloss_post": val_logloss_post,
            "ece_pre": val_ece_pre,
            "ece_post": val_ece_post,
        },
    }


def save_model(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")


def train() -> None:
    setup_logging()
    features, targets = load_training_data(PROCESSED_DATA_PATH)
    model = train_softmax_regression(features, targets)
    save_model(model, MODEL_PATH)
    save_json("model.json", model)
    if "training_metrics" in model:
        save_json("metrics.json", {"train": model["training_metrics"]})


if __name__ == "__main__":
    train()
