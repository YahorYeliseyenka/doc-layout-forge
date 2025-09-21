IMAGE_NAME ?= doc-layout-api
CPU_BASE   ?= python:3.13-slim
GPU_BASE   ?= nvidia/cuda:13.0.1-runtime-ubuntu24.04
CONTAINER_PORT ?= 49494
PORT ?= $(CONTAINER_PORT)

ifneq (,$(wildcard .env))
include .env
export
endif

# ---- Autodetection by MODEL_DEVICE ----
# cpu                -> MODE=gpu off
# cuda:<digit(s)>    -> MODE=gpu on, device=<N>
# else               -> MODE=cpu
MODEL_DEVICE_SAFE := $(shell printf "%s" "$(MODEL_DEVICE)" | tr -d '\n')
GPU_INDEX := $(shell echo "$(MODEL_DEVICE_SAFE)" | sed -n 's/^cuda:\([0-9][0-9]*\)$$/\1/p')
ifeq ($(MODEL_DEVICE_SAFE),cpu)
  MODE := cpu
else ifneq ($(GPU_INDEX),)
  MODE := gpu
else
  MODE := cpu
endif

ifeq ($(MODE),gpu)
  BASE := $(GPU_BASE)
  ifneq ($(GPU_INDEX),)
    GPUFLAG := --gpus "device=$(GPU_INDEX)"
  else
    GPUFLAG := --gpus all
  endif
else
  BASE := $(CPU_BASE)
  GPUFLAG :=
endif

NAME := $(IMAGE_NAME)-$(MODE)
SHELL := /bin/bash

.PHONY: help init build rebuild run stop rm logs

help: ## Show this help (dynamic) and current defaults
	@echo "Usage: make <target> [PORT=<host_port>]"
	@echo "Detected from .env: MODEL_DEVICE='$(MODEL_DEVICE_SAFE)' -> MODE=$(MODE) $(if $(GPU_INDEX),GPU_INDEX=$(GPU_INDEX),)"
	@echo "IMAGE_NAME=$(IMAGE_NAME)  BASE=$(BASE)  PORT=$(PORT)  CONTAINER_PORT=$(CONTAINER_PORT)"
	@echo
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z0-9_.-]+:.*##/{printf "  %-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

init: ## Quick check: Docker present, optional nvidia-smi
	@docker --version
	@if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi | head -n 3; else echo "nvidia-smi not found (GPU optional)"; fi

build: ## Build image (BASE depends on MODEL_DEVICE)
	@echo ">> Building $(IMAGE_NAME):$(MODE) with BASE=$(BASE)"
	docker build --build-arg BASE_IMAGE=$(BASE) -t $(IMAGE_NAME):$(MODE) .

rebuild: ## Rebuild without cache
	@echo ">> Rebuilding (no cache) $(IMAGE_NAME):$(MODE) with BASE=$(BASE)"
	docker build --no-cache --build-arg BASE_IMAGE=$(BASE) -t $(IMAGE_NAME):$(MODE) .

run: ## Run container (PORT -> CONTAINER_PORT); passes MODEL_DEVICE inside
	@echo ">> Starting $(NAME): host $(PORT) -> container $(CONTAINER_PORT)  (MODE=$(MODE))"
	docker run -d $(GPUFLAG) --name $(NAME) \
		-e MODEL_DEVICE="$(MODEL_DEVICE_SAFE)" \
		-p $(PORT):$(CONTAINER_PORT) $(IMAGE_NAME):$(MODE)

stop: ## Stop container
	- docker stop $(NAME)

rm: ## Remove container (force)
	- docker rm -f $(NAME)

logs: ## Tail logs
	docker logs -f $(NAME)
