# Use build argument to switch between CPU and GPU base images
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE} AS base

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# Install uv CLI for environment and dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency definitions (without project code)
COPY pyproject.toml uv.lock .python-version ./

# Install Python runtime and dependencies via uv (no project code yet)
RUN uv sync --locked --no-install-project --no-dev

# Copy application code
COPY ./src ./src
COPY ./models ./models
COPY .env ./

# Finalize sync to install project code
RUN uv sync --locked --no-dev

# Expose the port and set runtime command
EXPOSE 49494
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "49494"]
