import asyncio

from typing import List

from fastapi import HTTPException, status

from .schemas import FarmRequest


MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "risk-scoring"

MODEL_READY = True

INFERENCE_TIMEOUT_SECONDS = 2.0

MODEL_FIXED_OVERHEAD_SECONDS = 0.020
MODEL_PER_ITEM_SECONDS = 0.003

ALLOWED_REGIONS = {
    "Krasnodar",
    "Rostov",
    "Stavropol",
}


def calculate_risk(
    data: FarmRequest,
) -> float:
    """
    Имитация ML-модели.
    """

    score = 0.1

    if data.payment_delay_days > 30:
        score += 0.3

    if data.previous_defaults > 0:
        score += 0.3

    if data.debt > 5_000_000:
        score += 0.2

    if data.precipitation_mm < 100:
        score += 0.1

    return round(
        min(score, 1.0),
        2,
    )


def get_risk_level(
    score: float,
) -> str:
    if score < 0.3:
        return "low"

    if score < 0.7:
        return "medium"

    return "high"


def get_recommendation(
    level: str,
) -> str:
    if level == "low":
        return "Стандартное рассмотрение"

    if level == "medium":
        return "Требуется дополнительная проверка"

    return "Высокий риск. Требуется ручное рассмотрение"


def validate_business_rules(
    data: FarmRequest,
) -> None:
    """
    Pydantic проверяет типы, обязательность и диапазоны.
    Здесь проверяются правила предметной области.
    """

    if data.region not in ALLOWED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown region: {data.region}. "
                f"Allowed regions: {sorted(ALLOWED_REGIONS)}"
            ),
        )


def make_prediction_result(
    data: FarmRequest,
    score: float,
    request_id: str,
) -> dict:
    level = get_risk_level(score)

    return {
        "request_id": request_id,
        "farm_id": data.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": get_recommendation(level),
        "model_version": MODEL_VERSION,
    }


model_lock = asyncio.Lock()


async def model_predict_single(
    data: FarmRequest,
) -> float:
    """Одиночный вызов модели."""

    async with model_lock:
        await asyncio.sleep(MODEL_FIXED_OVERHEAD_SECONDS + MODEL_PER_ITEM_SECONDS)

        return calculate_risk(data)


async def model_predict_batch(
    items: List[FarmRequest],
) -> List[float]:
    """Один пакетный вызов модели."""

    async with model_lock:
        await asyncio.sleep(
            MODEL_FIXED_OVERHEAD_SECONDS + MODEL_PER_ITEM_SECONDS * len(items)
        )

        return [calculate_risk(item) for item in items]
