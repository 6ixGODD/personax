#!/bin/sh
# ============================================================================
# Code Formatting Script
# ============================================================================
# Description: Format Python code using isort and yapf
# Usage: format-code.sh [directory]
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

# Check if dev dependencies are installed
check_dependencies() {
    log_info "Checking formatting dependencies..."

    if ! $POETRY_CMD run python -c "import isort" 2>/dev/null; then
        log_warn "isort not found, installing dev dependencies..."
        $POETRY_CMD install --extras dev
    fi

    if ! $POETRY_CMD run python -c "import yapf" 2>/dev/null; then
        log_warn "yapf not found, installing dev dependencies..."
        $POETRY_CMD install --extras dev
    fi

    log_success "Dependencies check complete"
}

# Run isort
run_isort() {
    log_step "Running isort (import sorting)..."

    if ! dir_exists "$TARGET_DIR"; then
        log_warn "Directory not found: $TARGET_DIR"
        return 1
    fi

    if $POETRY_CMD run isort "$TARGET_DIR"; then
        log_success "isort complete"
        return 0
    else
        log_error "isort failed"
        return 1
    fi
}

# Run yapf
run_yapf() {
    log_step "Running yapf (code formatting)..."

    if ! dir_exists "$TARGET_DIR"; then
        log_warn "Directory not found: $TARGET_DIR"
        return 1
    fi

    if $POETRY_CMD run yapf -i -r "$TARGET_DIR"; then
        log_success "yapf complete"
        return 0
    else
        log_error "yapf failed"
        return 1
    fi
}

# Main
main() {
    print_header "Code Formatting"

    log_info "Target directory: $TARGET_DIR"
    echo ""

    check_dependencies
    echo ""

    isort_result=0
    yapf_result=0

    run_isort || isort_result=$?
    echo ""

    run_yapf || yapf_result=$?
    echo ""

    print_separator
    if [ $isort_result -eq 0 ] && [ $yapf_result -eq 0 ]; then
        log_success "All formatting complete!"
    else
        log_error "Some formatting failed"
        exit 1
    fi
    print_separator
}

main