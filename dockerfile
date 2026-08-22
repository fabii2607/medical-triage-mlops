FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala o uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Instala somente as dependências necessárias para a API
RUN uv pip install \
    --system \
    "fastapi>=0.141.1" \
    "joblib>=1.5.3" \
    "numpy>=2.4.6" \
    "pydantic>=2.13.4" \
    "scikit-learn>=1.9.0" \
    "uvicorn[standard]>=0.52.0"

# Copia a API e o modelo
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]