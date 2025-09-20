.PHONY: help
help: ## Display this help message with available commands
	@printf "Usage: make [target]\n\n"
	@awk 'BEGIN { FS = "##" }                      \
	     /^[^[:space:]#]+:.*##/ {                  \
	       target = $$1;                           \
	       sub(/:$$/, "", target);                 \
	       printf "  %-15s %s\n", target, $$2;     \
	     }' $(MAKEFILE_LIST)

.PHONY: init
init: ## Init service
	@echo "Init service"
