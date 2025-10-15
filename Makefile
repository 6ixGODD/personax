# ============================================================================
# PersonaX Makefile
# ============================================================================

# Variables
POETRY_CMD = poetry

# Scripts
FORMAT_SCRIPT         = bin/format.sh
LINT_SCRIPT           = bin/lint.sh
CHECK_POETRY_SCRIPT   = bin/tools/check-poetry.sh
INSTALL_POETRY_SCRIPT = bin/tools/install-poetry.sh

SHELL_CMD = bash

# ============================================================================
# Environment Verification
# ============================================================================

.PHONY: verify-env
verify-env:
	@echo "========================================"
	@echo "Environment Verification"
	@echo "========================================"
	@echo ""
	@echo "Poetry version:"
	@$(POETRY_CMD) --version 2>/dev/null || echo [E] Poetry not found - run 'make install-poetry'
	@echo ""
	@echo "Python version:"
	@$(POETRY_CMD) run python --version 2>/dev/null || echo [E] Python environment not found - run 'make setup-env'
	@echo ""
	@echo "========================================"

.PHONY: install-poetry
install-poetry: verify-env
	@echo "Installing Poetry..."
	@$(SHELL_CMD) $(INSTALL_POETRY_SCRIPT)

.PHONY: setup-env
setup-env: verify-env check-poetry
	@echo "Setting up environment..."
	@$(POETRY_CMD) lock --no-update
	@$(POETRY_CMD) install
	@echo "Environment setup complete."

.PHONY: setup-dev
setup-dev: verify-env check-poetry
	@echo "Setting up development environment..."
	@$(POETRY_CMD) install --extras dev --extras test
	@echo "Development environment setup complete."

# ============================================================================
# Development
# ============================================================================

.PHONY: format
format: verify-env check-poetry
	@echo "Formatting code..."
	@$(SHELL_CMD) $(FORMAT_SCRIPT)

.PHONY: lint
lint: verify-env check-poetry
	@echo "Linting code..."
	@$(SHELL_CMD) $(LINT_SCRIPT)

.PHONY: check
check: verify-env format lint
	@echo "Code check complete."

.PHONY: test
test: verify-env check-poetry
	@echo "Running tests..."
	@$(POETRY_CMD) run pytest -v
	@echo "Tests complete."

.PHONY: test-coverage
test-coverage: verify-env check-poetry
	@echo "Running tests with coverage..."
	@$(POETRY_CMD) run pytest --cov=prototypex --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

# ============================================================================
# Cleanup
# ============================================================================

.PHONY: clean
clean: verify-env clean-proto
	@echo "Cleaning build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .mypy_cache .coverage htmlcov/ dist/ build/ 2>/dev/null || true
	@echo "Clean complete."

.PHONY: clean-all
clean-all: verify-env clean
	@echo "Removing virtual environment..."
	@$(POETRY_CMD) env remove --all 2>/dev/null || true
	@echo "Deep clean complete."

# ============================================================================
# Utilities
# ============================================================================

.PHONY: help
help:
	@echo "========================================"
	@echo "PersonaX Makefile Commands"
	@echo "========================================"
	@echo ""
	@echo "Environment Setup:"
	@echo "  make verify-env          - Verify environment setup"
	@echo "  make check-poetry        - Check if Poetry is installed"
	@echo "  make install-poetry      - Install Poetry"
	@echo "  make setup-env           - Setup poetry environment"
	@echo "  make setup-dev           - Setup development environment"
	@echo ""
	@echo "Development:"
	@echo "  make format              - Format code with isort and yapf"
	@echo "  make lint                - Lint code with mypy and pylint"
	@echo "  make check               - Run format and lint"
	@echo "  make test                - Run tests with pytest"
	@echo "  make test-coverage       - Run tests with coverage report"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean               - Clean build artifacts and cache"
	@echo "  make clean-all           - Deep clean including virtual environment"
	@echo ""
	@echo "Utilities:"
	@echo "  make help                - Show this help message"
	@echo "========================================"

.DEFAULT_GOAL := help
