import asyncio
import time

from dataclasses import dataclass

import httpx

from fastapi import HTTPException, status

from .logging_config import logger
from .model import (
    INFERENCE_TIMEOUT_SECONDS,
    model_predict_batch,
)
from .schemas import (
    FarmRequest,
    MAX_BATCH_SIZE,
    LLM_MODEL,
    LLM_DEFAULT_MAX_TOKENS,
    LLM_DEFAULT_TEMPERATURE,
)


LLM_BASE_URL = "https://vireonix.ai/v1"
LLM_TIMEOUT_SECONDS = 60.0

MAX_BATCH_WAIT_MS = 50
QUEUE_MAX_SIZE = 100


async def call_vireonix_llm(
    client: httpx.AsyncClient,
    prompt: str,
    request_id: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_DEFAULT_TEMPERATURE,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Асинхронный запрос к Vireonix.

    FastAPI endpoints в этом проекте асинхронные, поэтому
    используется httpx.AsyncClient + await, а не requests.post().

    Vireonix использует OpenAI-совместимый формат:
        POST /v1/chat/completions
    """

    url = f"{LLM_BASE_URL}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты помощник сервиса агроскоринга. "
                    "Отвечай кратко и понятно. "
                    "Не изменяй переданный risk_score и risk_level. "
                    "Не придумывай причины, которых нет во входных данных."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    started = time.perf_counter()

    try:
        response = await client.post(
            url,
            headers={
                "Content-Type": "application/json",
            },
            json=payload,
        )

    except httpx.TimeoutException as exc:
        logger.exception(
            "LLM timeout | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM request timeout",
        ) from exc

    except httpx.RequestError as exc:
        logger.exception(
            "LLM connection error | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is unavailable",
        ) from exc

    latency_ms = (time.perf_counter() - started) * 1000

    if response.status_code != status.HTTP_200_OK:
        logger.error(
            "LLM upstream error | request_id=%s | status=%s | body=%s",
            request_id,
            response.status_code,
            response.text[:500],
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(f"LLM provider returned an error (HTTP {response.status_code})"),
        )

    try:
        data = response.json()

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider returned invalid JSON",
        ) from exc

    choices = data.get(
        "choices",
        [],
    )

    if not choices:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM response does not contain choices",
        )

    choice = choices[0]

    message = choice.get(
        "message",
        {},
    )

    content = message.get("content")

    if not content:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM response does not contain message content",
        )

    return {
        "request_id": request_id,
        "requested_model": model,
        "returned_model": data.get("model"),
        "answer": content,
        "reasoning": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "latency_ms": round(
            latency_ms,
            2,
        ),
    }


def build_scoring_explanation_prompt(
    prediction: dict,
) -> str:
    """
    Формирует промпт только из результата
    скорингового алгоритма.

    Не отправляем внешний farm_id и исходные
    финансовые признаки в публичный LLM gateway.
    """

    return (
        "Сформируй краткое объяснение для оператора банка "
        "по результату алгоритмического агроскоринга.\n\n"
        f"risk_score: {prediction['risk_score']}\n"
        f"risk_level: {prediction['risk_level']}\n"
        f"базовая рекомендация: {prediction['recommendation']}\n"
        f"версия скоринговой модели: {prediction['model_version']}\n\n"
        "Требования к ответу:\n"
        "1. 2-4 предложения.\n"
        "2. Не меняй категорию риска.\n"
        "3. Не придумывай дополнительные факторы риска.\n"
        "4. Укажи, что решение требует применения правил банка "
        "и при необходимости проверки оператором."
    )


inference_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)


@dataclass
class InferenceTask:
    data: FarmRequest
    request_id: str
    future: asyncio.Future


async def batch_worker() -> None:
    """
    Фоновый worker:

    1. Берет первый запрос.
    2. Копит дополнительные запросы до MAX_BATCH_SIZE
       или до истечения MAX_BATCH_WAIT_MS.
    3. Выполняет один batch inference.
    4. Возвращает каждому HTTP-запросу его score.
    """

    logger.info("Batch worker started")

    while True:
        first_task = await inference_queue.get()

        if first_task.future.cancelled():
            inference_queue.task_done()
            continue

        batch = [first_task]

        deadline = time.perf_counter() + MAX_BATCH_WAIT_MS / 1000

        while len(batch) < MAX_BATCH_SIZE:
            remaining_time = deadline - time.perf_counter()

            if remaining_time <= 0:
                break

            try:
                task = await asyncio.wait_for(
                    inference_queue.get(),
                    timeout=remaining_time,
                )

                if task.future.cancelled():
                    inference_queue.task_done()
                    continue

                batch.append(task)

            except asyncio.TimeoutError:
                break

        logger.info(
            "Dynamic batch formed | size=%s | queue_size=%s",
            len(batch),
            inference_queue.qsize(),
        )

        try:
            scores = await asyncio.wait_for(
                model_predict_batch([task.data for task in batch]),
                timeout=INFERENCE_TIMEOUT_SECONDS,
            )

            for task, score in zip(
                batch,
                scores,
            ):
                if not task.future.done():
                    task.future.set_result(score)

        except Exception as exc:
            logger.exception("Batch inference failed")

            for task in batch:
                if not task.future.done():
                    task.future.set_exception(exc)

        finally:
            for _ in batch:
                inference_queue.task_done()
