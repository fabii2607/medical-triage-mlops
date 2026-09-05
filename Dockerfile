FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Instala uma versão fixa do uv
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /uvx /bin/

# Instala somente o núcleo compartilhado e o extra da API.
COPY pyproject.toml uv.lock ./
RUN uv sync \
    --locked \
    --no-default-groups \
    --extra api \
    --no-install-project \
    --no-cache

# Cria um usuário sem privilégios para executar a API.
RUN groupadd --gid 10001 app \
    && useradd \
    --uid 10001 \
    --gid app \
    --home-dir /app \
    --no-create-home \
    --shell /usr/sbin/nologin \
    app

# Copia a API e o modelo
COPY api/ ./api/
COPY models/ ./models/

USER app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
