#!/usr/bin/env bash
# Shared helpers for shell-driven tests.

RAN=0
PASS=0
FAIL=0
SKIP=0

assert_passes() {
    if (( $# == 0 )); then
        printf 'assert_passes: expected a command\n' >&2
        return 2
    fi

    RAN=$((RAN + 1))
    if "$@"; then
        PASS=$((PASS + 1))
        printf 'PASS: %s\n' "$*"
        return 0
    else
        local status=$?
        FAIL=$((FAIL + 1))
        printf 'FAIL: %s (exit %d)\n' "$*" "$status" >&2
        return 1
    fi
}

assert_fails() {
    if (( $# < 2 )); then
        printf 'assert_fails: expected a label and command\n' >&2
        return 2
    fi

    local label=$1
    shift
    RAN=$((RAN + 1))
    if "$@"; then
        FAIL=$((FAIL + 1))
        printf 'FAIL: %s (expected failure)\n' "$label" >&2
        return 1
    else
        PASS=$((PASS + 1))
        printf 'PASS: %s (failed as expected)\n' "$label"
        return 0
    fi
}

assert_skip() {
    SKIP=$((SKIP + 1))
    printf 'SKIP: %s\n' "$*"
}

print_result() {
    printf '%s\n' '=========================' 'Result:'
    printf '[TOTAL_TESTS]: %d\n' "$((RAN + SKIP))"
    printf '[TESTS_RAN]: %d\n' "$RAN"
    printf '[TESTS_PASSED]: %d\n' "$PASS"
    printf '[TESTS_FAILED]: %d\n' "$FAIL"
    printf '[TESTS_SKIPPED]: %d\n' "$SKIP"
    printf '%s\n' '========================='

    if (( RAN == 0 )); then
        printf 'No tests ran.\n' >&2
        return 1
    fi
    (( FAIL == 0 ))
}
