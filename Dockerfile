FROM ubuntu:jammy AS builder

RUN apt update -y && apt install patchelf build-essential --no-install-recommends -y

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_NO_DEV=1

COPY .python-version .
COPY pyproject.toml .
COPY uv.lock .

RUN uv venv
RUN uv sync --no-dev

COPY src src
COPY main.py .

RUN uv run nuitka main.py \
    --standalone \
    --onefile \
    --output-dir=dist \
    --lto=yes \
    --nofollow-import-to=pytest \
    --python-flag=nosite,-O \
    --plugin-enable=anti-bloat,implicit-imports,data-files,pylint-warnings

FROM ubuntu:jammy

WORKDIR /run

COPY --from=builder /app/dist/main.bin .
COPY LICENSE .
RUN chmod +x /run/main.bin

EXPOSE 80

ENTRYPOINT ["/run/main.bin"]
