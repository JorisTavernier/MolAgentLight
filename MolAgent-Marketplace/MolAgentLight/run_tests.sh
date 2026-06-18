#!/bin/bash
# Run the MolAgentLight test suite (MCP server + skills).
#
# Scope: mcp/tests/ only. This intentionally EXCLUDES the bundled AutoMol
# package tests (AutoMol/automol/tests) — those belong to the upstream library,
# not this plugin. The loose mcp/test_*.py files are manual live-server scripts,
# not pytest suites, and are excluded by targeting the tests/ directory.
#
# Usage:
#   ./run_tests.sh              # run everything
#   ./run_tests.sh -k auth      # forward args to pytest (filter, -x, etc.)
set -euo pipefail

cd "$(dirname "$0")"

exec uv run --active --no-sync pytest mcp/tests/ -v "$@"
