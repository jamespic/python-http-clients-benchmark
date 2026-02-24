FROM ghcr.io/astral-sh/uv:python3.14-trixie
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen
COPY . .
EXPOSE 8000 8443
CMD ["uv", "run", "--no-dev", "--frozen", "invoke", "serve"]