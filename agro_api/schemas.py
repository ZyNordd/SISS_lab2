from typing import List, Optional

from pydantic import BaseModel, Field


LLM_MODEL = "auto"
LLM_DEFAULT_MAX_TOKENS = 300
LLM_DEFAULT_TEMPERATURE = 0.3
MAX_BATCH_SIZE = 8


class FarmRequest(BaseModel):
    """Входные данные для оценки риска одного хозяйства."""

    farm_id: str = Field(
        ...,
        min_length=1,
        description="Идентификатор хозяйства",
    )

    region: str = Field(
        ...,
        min_length=1,
        description="Регион",
    )

    crop_type: str = Field(
        ...,
        min_length=1,
        description="Основная сельскохозяйственная культура",
    )

    area_ha: float = Field(
        ...,
        gt=0,
        description="Площадь посевов, га",
    )

    temperature_avg: float = Field(
        ...,
        ge=-60,
        le=60,
        description="Средняя температура, °C",
    )

    precipitation_mm: float = Field(
        ...,
        ge=0,
        description="Количество осадков, мм",
    )

    payment_delay_days: int = Field(
        ...,
        ge=0,
        description="Количество дней просрочки платежа",
    )

    previous_defaults: int = Field(
        ...,
        ge=0,
        description="Количество предыдущих дефолтов",
    )

    debt: float = Field(
        ...,
        ge=0,
        description="Текущая задолженность",
    )


class PredictionResponse(BaseModel):
    """Ответ для одного прогноза."""

    request_id: str
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: str


class BatchRequest(BaseModel):
    """Явный batch, который клиент отправляет одним HTTP-запросом."""

    items: List[FarmRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="Список объектов для пакетного инференса",
    )


class BatchItemResponse(BaseModel):
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    request_id: str
    batch_size: int
    processing_time_ms: float
    items: List[BatchItemResponse]


class PaginationMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: Optional[str]
    prev_page: Optional[str]


class PaginatedPredictionsResponse(BaseModel):
    data: List[PredictionResponse]
    meta: PaginationMeta


class LLMChatRequest(BaseModel):
    """
    Запрос к внешней LLM.

    Не передавайте сюда реальные
    конфиденциальные банковские или персональные данные.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Текст запроса к LLM",
    )

    model: str = Field(
        default=LLM_MODEL,
        min_length=1,
        description="Модель Vireonix. Для демо используется auto",
    )

    temperature: float = Field(
        default=LLM_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
    )

    max_tokens: int = Field(
        default=LLM_DEFAULT_MAX_TOKENS,
        ge=1,
        le=2000,
    )


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: Optional[str]
    answer: str
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    latency_ms: float


class PredictWithLLMResponse(BaseModel):
    """
    Комбинированный ответ:
    1. детерминированный результат агроскоринга;
    2. текстовое объяснение, сформированное LLM.
    """

    prediction: PredictionResponse
    llm: LLMChatResponse
