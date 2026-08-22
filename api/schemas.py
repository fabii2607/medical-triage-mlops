from pydantic import BaseModel, field_validator


class PredictionRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("text must not be empty or contain only whitespace")

        return value


class PredictionResponse(BaseModel):
    triage_level: str
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool