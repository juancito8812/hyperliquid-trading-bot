# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY bots ./bots

RUN useradd --create-home --uid 1000 bot && chown -R bot:bot /app
USER bot

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

HEALTHCHECK --interval=60s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import glob,sys; sys.exit(0 if any('run_bot.py' in open(p+'/cmdline').read() for p in glob.glob('/proc/[0-9]*')) else 1)"

ENTRYPOINT ["python", "src/run_bot.py"]
CMD ["bots/sol_ema_adx.yaml"]
