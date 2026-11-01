# The chat API (api/main.py). The image holds the code and the runtime dependencies only:
# the feature table, models and Zillow index are mounted at run time, read-only:
#
#   docker build -t let-me-have-a-house .
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/data:/app/data:ro" -v "$PWD/output:/app/output:ro" \
#     -e OLLAMA_URL=http://host.docker.internal:11434 let-me-have-a-house
#
# Base images are pinned by digest; Dependabot (.github/dependabot.yml) proposes updates.

FROM python:3.14-slim-trixie@sha256:cd89c30eda26ee911a4611b18680086e0ea42e6f5f8cbe653d07fa7958c9cec8 AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies only, without the dev and notebooks groups; their own layer, so a code
# change doesn't reinstall them
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups

FROM python:3.14-slim-trixie@sha256:cd89c30eda26ee911a4611b18680086e0ea42e6f5f8cbe653d07fa7958c9cec8
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY config ./config
COPY src ./src
COPY api ./api
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
