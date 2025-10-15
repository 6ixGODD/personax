#!/bin/sh
# ============================================================================
# Code Linting Script
# ============================================================================
# Description: Lint Python code using mypy and pylint
# Usage: lint-code.sh [directory]
# Default directory: personax/
# ============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source common utilities
# shellcheck source=bin/common.sh
. "$SCRIPT_DIR/common.sh"

# Variables
TARGET_DIR="${1:-personax}"
POETRY_CMD="poetry"

# Check if test dependencies are installed
check_dependencies() {
    log_info "Checking linting dependencies..."

    if ! $POETRY_CMD run python -c "import mypy" 2>/dev/null; then
        log_warn "mypy not found, installing test dependencies..."
        $POETRY_CMD install --extras test
    fi

    if ! $POETRY_CMD run python -c "import pylint" 2>/dev/null; then
        log_warn "pylint not found, installing test dependencies..."
        $POETRY_CMD install --extras test
    fi

    log_success "Dependencies check complete"
}

# Run mypy
run_mypy() {
    log_step "Running mypy (type checking)..."

    if ! dir_exists "$TARGET_DIR"; then
        log_warn "Directory not found: $TARGET_DIR"
        return 1
    fi

    if $POETRY_CMD run mypy "$TARGET_DIR" --ignore-missing-imports --no-error-summary 2>/dev/null; then
        log_success "✓ mypy: No type errors found"
        return 0
    else
        log_warn "⚠ mypy: Type errors found (see above)"
        return 1
    fi
}

# Run pylint
run_pylint() {
    log_step "Running pylint (code quality)..."

    if ! dir_exists "$TARGET_DIR"; then
        log_warn "Directory not found: $TARGET_DIR"
        return 1
    fi

    # Run pylint and capture exit code
    if $POETRY_CMD run pylint "$TARGET_DIR" --exit-zero --score=yes 2>/dev/null; then
        log_success "✓ pylint: Check complete"
        return 0
    else
        log_warn "⚠ pylint: Issues found (see above)"
        return 1
    fi
}

# Main
main() {
    print_header "Code Linting"

    log_info "Target directory: $TARGET_DIR"
    echo ""

    check_dependencies
    echo ""

    mypy_result=0
    pylint_result=0

    run_mypy || mypy_result=$?
    echo ""

    run_pylint || pylint_result=$?
    echo ""

    print_separator
    if [ $mypy_result -eq 0 ] && [ $pylint_result -eq 0 ]; then
        log_success "✓ All checks passed!"
    else
        log_warn "⚠ Some checks failed"
        exit 1
    fi
    print_separator
}

main