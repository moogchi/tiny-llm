#!/usr/bin/env bash
# Shared helpers for E2E tests

RAN=0
PASS=0
FAIL=0
SKIP=0

assert_passes(){
    RAN=$((RAN+1))
    if "$@"; then
        PASS=$((PASS+1))
        echo "PASS: $*"
    else
        FAIL=$((FAIL+1))
        echo "FAIL: $* (expected success)"
    fi
}

assert_fails(){
    local MSG=$1
    shift
    RAN=$((RAN+1))
    if ! "$@"; then
        PASS=$((PASS+1))
        echo "PASS: Test passed $MSG"
    else
        FAIL=$((FAIL+1))
        echo "FAIL: Test failed $MSG (expected failure)"
    fi
}

assert_skip(){
    SKIP=$((SKIP+1))
    echo "SKIP: $*"
}


print_result(){
    echo "========================="
    echo "Result:"
    echo "[TOTAL_TESTS]: $((RAN+SKIP))"
    echo "[TESTS_RAN]: $RAN"
    echo "[TESTS_PASSED]: $PASS"
    echo "[TESTS_FAILED]: $FAIL"
    echo "[TESTS_SKIPPED]: $SKIP"
    echo "========================="
    [ "$FAIL" -eq 0 ]
}
