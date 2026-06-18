#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["fastmcp", "httpx"]
# ///
"""
End-to-end test script for the AutoMol MCP server (remote mode with auth).

Tests all tools and parameter options against a running server. Designed for
manual verification — prints pass/fail with details for each test.

Usage:
  uv run mcp/test_mcp_server.py \
    --url http://127.0.0.1:8001/mcp \
    --csv /path/to/data.csv \
    --admin-token molagent_adm_... \
    --user-token molagent_usr_...

Or load tokens from a JSON file:
  uv run mcp/test_mcp_server.py \
    --url http://127.0.0.1:8001/mcp \
    --csv /path/to/data.csv \
    --token-file /path/to/auth_tokens.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


# ── Helpers ──────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error: str | None = None
        self.result: dict | None = None
        self.duration: float = 0.0


results: list[TestResult] = []


def parse_tool_result(raw) -> dict:
    """Extract JSON dict from MCP tool result (handles various fastmcp response shapes)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list):
        for item in raw:
            if hasattr(item, "text"):
                return json.loads(item.text)
    if hasattr(raw, "content"):
        for item in raw.content:
            if hasattr(item, "text"):
                return json.loads(item.text)
    raise ValueError(f"Cannot parse tool result: {type(raw)}")


async def call(client: Client, tool: str, args: dict) -> dict:
    """Call a tool and parse the result."""
    raw = await client.call_tool(tool, args)
    return parse_tool_result(raw)


async def call_task(client: Client, tool: str, args: dict, timeout: float = 600) -> dict:
    """Call a long-running task tool and poll until completion."""
    raw = await client.call_tool(tool, args)
    result = parse_tool_result(raw)

    # If it's a task, we get a task_id and need to poll
    if "task_id" in result:
        task_id = result["task_id"]
        start = time.time()
        while time.time() - start < timeout:
            await asyncio.sleep(3)
            status_raw = await client.call_tool("tasks/get", {"task_id": task_id})
            status = parse_tool_result(status_raw)
            state = status.get("status") or status.get("state", "")
            if state in ("completed", "success", "done"):
                return status.get("result", status)
            if state in ("failed", "error"):
                raise RuntimeError(f"Task failed: {status.get('error', status)}")
        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

    return result


def run_test(name: str):
    """Decorator-like context for test functions."""
    r = TestResult(name)
    results.append(r)
    return r


# ── Test Functions ───────────────────────────────────────────────────────────

async def test_list_options(client: Client):
    """Test list_options with all categories."""
    categories = [
        ("feature_generators", None),
        ("base_estimators", "Regression"),
        ("base_estimators", "Classification"),
        ("blender_estimators", "Regression"),
        ("dim_reduction", None),
        ("model_configs", None),
        ("search_types", None),
        ("scorers", "Regression"),
        ("scorers", "Classification"),
    ]
    for category, task in categories:
        t = run_test(f"list_options({category}, task={task})")
        start = time.time()
        try:
            args = {"category": category}
            if task:
                args["task"] = task
            result = await call(client, "list_options", args)
            t.result = result
            # Should return a non-empty dict
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            assert len(result) > 0, "Empty result"
            t.passed = True
        except Exception as e:
            t.error = str(e)
        t.duration = time.time() - start


async def test_start_training_session(client: Client, csv_file: str) -> str | None:
    """Test start_training_session and return session_id."""
    t = run_test("start_training_session")
    start = time.time()
    try:
        result = await call(client, "start_training_session", {"csv_file": csv_file})
        t.result = result
        assert "session_id" in result, "No session_id in response"
        assert "detected" in result, "No detected config"
        assert "options" in result, "No options"
        assert "question" in result, "No question"
        detected = result["detected"]
        assert detected["smiles_column"], "No SMILES column detected"
        assert len(detected["properties"]) > 0, "No properties detected"
        # Check new fields
        assert "categorical" in detected, "Missing categorical in detected"
        assert "nb_classes" in detected, "Missing nb_classes in detected"
        t.passed = True
        t.duration = time.time() - start
        return result["session_id"]
    except Exception as e:
        t.error = str(e)
        t.duration = time.time() - start
        return None


async def test_answer_training_question_overrides(client: Client, csv_file: str):
    """Test answer_training_question with various parameter overrides."""
    # Start a fresh session for override tests
    result = await call(client, "start_training_session", {"csv_file": csv_file})
    sid = result["session_id"]

    # Test basic overrides
    override_tests = [
        ("override: task", {"task": "Classification"}),
        ("override: computational_load", {"computational_load": "moderate"}),
        ("override: feature_keys", {"feature_keys": ["Bottleneck", "rdkit"]}),
        ("override: use_log10", {"use_log10": True}),
        ("override: use_logit", {"use_logit": True}),
        ("override: split_strategy", {"split_strategy": "stratified"}),
        ("override: scorer", {"scorer": "balanced_accuracy"}),
        # Classification params
        ("override: categorical", {"categorical": True}),
        ("override: nb_classes", {"nb_classes": [3]}),
        # Refit control
        ("override: refit=False", {"refit": False}),
        ("override: include_test_in_refit=False", {"include_test_in_refit": False}),
        # Sample weights
        ("override: use_sample_weight", {"use_sample_weight": True}),
        ("override: sample_weight_selection", {"sample_weight_selection": "<1"}),
        ("override: sample_weight_multiplier", {"sample_weight_multiplier": 10.0}),
        # Model config
        ("override: model_config", {"model_config": "inner_methods"}),
        ("override: search_type", {"search_type": "randomized"}),
        ("override: randomized_iterations", {"randomized_iterations": 50}),
    ]

    for name, overrides in override_tests:
        t = run_test(f"answer_training_question: {name}")
        start = time.time()
        try:
            args = {"session_id": sid, **overrides}
            result = await call(client, "answer_training_question", args)
            t.result = result
            assert result.get("validation_error") is False, f"Validation error: {result.get('question')}"
            t.passed = True
        except Exception as e:
            t.error = str(e)
        t.duration = time.time() - start


async def test_answer_training_question_validation(client: Client, csv_file: str):
    """Test that invalid overrides produce validation errors."""
    result = await call(client, "start_training_session", {"csv_file": csv_file})
    sid = result["session_id"]

    validation_tests = [
        ("invalid: smiles_column", {"smiles_column": "nonexistent_column"}),
        ("invalid: properties", {"properties": ["fake_col"]}),
        ("invalid: feature_keys", {"feature_keys": ["totally_invalid_feature"]}),
    ]

    for name, overrides in validation_tests:
        t = run_test(f"answer_training_question: {name}")
        start = time.time()
        try:
            args = {"session_id": sid, **overrides}
            result = await call(client, "answer_training_question", args)
            t.result = result
            assert result.get("validation_error") is True, "Expected validation error"
            t.passed = True
        except Exception as e:
            t.error = str(e)
        t.duration = time.time() - start


async def test_answer_confirm(client: Client, csv_file: str) -> dict | None:
    """Test confirming a session returns a valid TrainingConfig."""
    t = run_test("answer_training_question: confirm")
    start = time.time()
    try:
        result = await call(client, "start_training_session", {"csv_file": csv_file})
        sid = result["session_id"]
        # Set minimal valid config and confirm
        await call(client, "answer_training_question", {
            "session_id": sid,
            "properties": ["prop1"],
            "computational_load": "free",
        })
        result = await call(client, "answer_training_question", {
            "session_id": sid,
            "confirm": True,
        })
        t.result = result
        assert result.get("validation_error") is False, f"Validation error on confirm: {result.get('question')}"
        assert result.get("config") is not None, "No config returned on confirm"
        config = result["config"]
        assert config["csv_file"], "No csv_file in config"
        assert config["properties"] == ["prop1"], f"Properties mismatch: {config['properties']}"
        t.passed = True
        t.duration = time.time() - start
        return config
    except Exception as e:
        t.error = str(e)
        t.duration = time.time() - start
        return None


async def test_train_and_visualize(client: Client, csv_file: str) -> dict | None:
    """Test full pipeline (free load for speed)."""
    t = run_test("train_and_visualize (free, single property)")
    start = time.time()
    try:
        sess = await call(client, "start_training_session", {"csv_file": csv_file})
        sid = sess["session_id"]
        confirmed = await call(client, "answer_training_question", {
            "session_id": sid,
            "properties": ["prop1"],
            "task": "Regression",
            "computational_load": "free",
            "refit": False,
            "confirm": True,
        })
        config = confirmed["config"]
        assert config is not None, "No config returned on confirm"

        result = await call_task(client, "train_and_visualize", {"config": config}, timeout=600)
        t.result = {k: v for k, v in result.items() if k != "dashboard_html"}
        assert result.get("metrics"), "No metrics returned"
        assert result.get("model_id"), "No model_id returned"
        t.passed = True
        t.duration = time.time() - start
        return result
    except Exception as e:
        t.error = f"{e}\n{traceback.format_exc()}"
        t.duration = time.time() - start
        return None


async def test_train_classification(client: Client, csv_file: str) -> dict | None:
    """Test classification training with class params."""
    t = run_test("train_and_visualize (classification, prop5)")
    start = time.time()
    try:
        sess = await call(client, "start_training_session", {"csv_file": csv_file})
        sid = sess["session_id"]
        confirmed = await call(client, "answer_training_question", {
            "session_id": sid,
            "properties": ["prop5"],
            "task": "Classification",
            "categorical": True,
            "computational_load": "free",
            "refit": False,
            "confirm": True,
        })
        config = confirmed["config"]
        result = await call_task(client, "train_and_visualize", {"config": config}, timeout=600)
        t.result = {k: v for k, v in result.items() if k != "dashboard_html"}
        assert result.get("metrics"), "No metrics returned"
        t.passed = True
        t.duration = time.time() - start
        return result
    except Exception as e:
        t.error = f"{e}\n{traceback.format_exc()}"
        t.duration = time.time() - start
        return None


async def test_list_models(client: Client):
    """Test list_models."""
    t = run_test("list_models")
    start = time.time()
    try:
        result = await call(client, "list_models", {})
        t.result = result
        assert "models" in result, "No models key"
        assert isinstance(result["models"], list), "models is not a list"
        t.passed = True
    except Exception as e:
        t.error = str(e)
    t.duration = time.time() - start
    return result.get("models", []) if t.passed else []


async def test_predict(client: Client, models: list[dict]):
    """Test predict with a model from the registry."""
    if not models:
        t = run_test("predict (skipped: no models)")
        t.error = "No models available to test prediction"
        return

    model = models[0]
    model_id = model["id"]

    t = run_test(f"predict (model={model_id})")
    start = time.time()
    try:
        smiles_list = [
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # ibuprofen
            "c1ccccc1",  # benzene
        ]
        result = await call_task(client, "predict", {
            "model_id": model_id,
            "smiles_list": smiles_list,
            "compute_sd": True,
        }, timeout=120)
        t.result = {k: v for k, v in result.items() if k != "csv_content"}
        assert result.get("status") == "completed" or result.get("n_predictions"), "Prediction did not complete"
        t.passed = True
    except Exception as e:
        t.error = f"{e}\n{traceback.format_exc()}"
    t.duration = time.time() - start


async def test_merge_models(client: Client, models: list[dict]):
    """Test merge_models with compatible models from registry."""
    # Find 2 models with different properties and same task type
    regression_models = [m for m in models if m.get("task_type") == "regression"]
    if len(regression_models) < 2:
        t = run_test("merge_models (skipped: need 2+ regression models)")
        t.error = "Need at least 2 regression models with different properties to test merge"
        return

    # Find two with non-overlapping properties
    pair = None
    for i, m1 in enumerate(regression_models):
        for m2 in regression_models[i + 1:]:
            props1 = set(m1.get("target_properties", []))
            props2 = set(m2.get("target_properties", []))
            if not props1 & props2:
                pair = (m1, m2)
                break
        if pair:
            break

    if not pair:
        t = run_test("merge_models (skipped: no non-overlapping property pair)")
        t.error = "All available models have overlapping properties"
        return

    m1, m2 = pair
    t = run_test(f"merge_models ({m1['id']} + {m2['id']})")
    start = time.time()
    try:
        result = await call_task(client, "merge_models", {
            "model_ids": [m1["id"], m2["id"]],
            "verify_encoder": True,
        }, timeout=120)
        t.result = result
        assert result.get("status") == "merged" or result.get("model_id"), "Merge did not complete"
        t.passed = True
    except Exception as e:
        t.error = f"{e}\n{traceback.format_exc()}"
    t.duration = time.time() - start


async def test_delete_model(client: Client, models: list[dict]):
    """Test delete_model on the last model (if any merged model exists)."""
    merged = [m for m in models if m.get("model_format") == "merged"]
    if not merged:
        t = run_test("delete_model (skipped: no merged model to safely delete)")
        t.error = "No merged model available to delete safely"
        return

    target = merged[-1]
    t = run_test(f"delete_model ({target['id']})")
    start = time.time()
    try:
        result = await call(client, "delete_model", {"model_id": target["id"]})
        t.result = result
        assert result.get("status") == "deleted", f"Unexpected status: {result.get('status')}"
        t.passed = True
    except Exception as e:
        t.error = str(e)
    t.duration = time.time() - start


async def test_admin_manage(admin_client: Client):
    """Test admin_manage operations."""
    # list_users
    t = run_test("admin_manage: list_users")
    start = time.time()
    try:
        result = await call(admin_client, "admin_manage", {"action": "list_users"})
        t.result = result
        assert "users" in result or isinstance(result, dict), "Unexpected list_users response"
        t.passed = True
    except Exception as e:
        t.error = str(e)
    t.duration = time.time() - start

    # create_token
    t = run_test("admin_manage: create_token")
    start = time.time()
    created_token = None
    try:
        result = await call(admin_client, "admin_manage", {
            "action": "create_token",
            "user_id": "test_integration_user",
        })
        t.result = result
        assert "token" in result or "user_token" in result, f"No token in response: {result}"
        created_token = result.get("token") or result.get("user_token")
        t.passed = True
    except Exception as e:
        t.error = str(e)
    t.duration = time.time() - start

    # revoke_user (by owner_id — plaintext tokens are unrecoverable from the store)
    if created_token:
        t = run_test("admin_manage: revoke_user")
        start = time.time()
        try:
            listed = await call(admin_client, "admin_manage", {"action": "list_users"})
            owner_id = next(
                (u["owner_id"] for u in listed.get("users", [])
                 if u.get("user_id") == "test_integration_user" and not u.get("revoked")),
                None,
            )
            assert owner_id, f"Could not find owner_id for created user in: {listed}"
            result = await call(admin_client, "admin_manage", {
                "action": "revoke_user",
                "owner_id": owner_id,
            })
            t.result = result
            assert result.get("status") == "revoked" or "revoked" in str(result).lower(), \
                f"Unexpected revoke response: {result}"
            t.passed = True
        except Exception as e:
            t.error = str(e)
        t.duration = time.time() - start


async def test_auth_denied(url: str):
    """Test that requests without valid auth are rejected."""
    t = run_test("auth: rejected without token")
    start = time.time()
    try:
        transport = StreamableHttpTransport(url, headers={})
        async with Client(transport) as client:
            try:
                await call(client, "list_models", {})
                t.error = "Expected auth error but call succeeded"
            except Exception as e:
                if "auth" in str(e).lower() or "401" in str(e) or "403" in str(e) or "token" in str(e).lower():
                    t.passed = True
                else:
                    t.error = f"Got error but not auth-related: {e}"
    except Exception as e:
        # Connection-level auth rejection is also valid
        if "auth" in str(e).lower() or "401" in str(e) or "403" in str(e):
            t.passed = True
        else:
            t.error = f"Unexpected error: {e}"
    t.duration = time.time() - start


async def test_sample_weight_training(client: Client, csv_file: str):
    """Test training with sample weights configured."""
    t = run_test("train_and_visualize (with sample weights)")
    start = time.time()
    try:
        sess = await call(client, "start_training_session", {"csv_file": csv_file})
        sid = sess["session_id"]
        confirmed = await call(client, "answer_training_question", {
            "session_id": sid,
            "properties": ["prop1"],
            "task": "Regression",
            "computational_load": "free",
            "use_sample_weight": True,
            "sample_weight_selection": "<3",
            "sample_weight_multiplier": 5.0,
            "refit": False,
            "confirm": True,
        })
        config = confirmed["config"]
        assert config.get("use_sample_weight") is True, "sample weight not set"
        assert config.get("sample_weight_selection") == "<3", "selection not set"
        assert config.get("sample_weight_multiplier") == 5.0, "multiplier not set"

        result = await call_task(client, "train_and_visualize", {"config": config}, timeout=600)
        t.result = {k: v for k, v in result.items() if k != "dashboard_html"}
        assert result.get("metrics"), "No metrics returned"
        t.passed = True
    except Exception as e:
        t.error = f"{e}\n{traceback.format_exc()}"
    t.duration = time.time() - start


# ── Main ─────────────────────────────────────────────────────────────────────

async def run_all_tests(url: str, csv_file: str, admin_token: str, user_token: str):
    """Run all tests in sequence."""
    print(f"\n{'=' * 70}")
    print(f"  AutoMol MCP Server Integration Tests")
    print(f"  URL:   {url}")
    print(f"  CSV:   {csv_file}")
    print(f"{'=' * 70}\n")

    # Create clients
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    user_headers = {"Authorization": f"Bearer {user_token}"}
    admin_transport = StreamableHttpTransport(url, headers=admin_headers)
    user_transport = StreamableHttpTransport(url, headers=user_headers)

    # ── Auth tests ───────────────────────────────────────────────────────────
    print("[1/9] Testing authentication...")
    await test_auth_denied(url)

    # ── list_options (admin) ────────────────────────────────────────────────
    print("[2/9] Testing list_options...")
    async with Client(admin_transport) as admin_client:
        await test_list_options(admin_client)

    # ── Session flow (user) ─────────────────────────────────────────────────
    print("[3/9] Testing start_training_session...")
    async with Client(user_transport) as user_client:
        await test_start_training_session(user_client, csv_file)

    # ── Override tests (user) ───────────────────────────────────────────────
    print("[4/9] Testing answer_training_question overrides...")
    async with Client(user_transport) as user_client:
        await test_answer_training_question_overrides(user_client, csv_file)

    # ── Validation tests (user) ─────────────────────────────────────────────
    print("[5/9] Testing answer_training_question validation...")
    async with Client(user_transport) as user_client:
        await test_answer_training_question_validation(user_client, csv_file)

    # ── Confirm test (user) ─────────────────────────────────────────────────
    print("[6/9] Testing session confirm...")
    async with Client(user_transport) as user_client:
        await test_answer_confirm(user_client, csv_file)

    # ── Training (user) ────────────────────────────────────────────────────
    print("[7/9] Testing train_and_visualize (regression, free)...")
    async with Client(user_transport) as user_client:
        await test_train_and_visualize(user_client, csv_file)

    print("       Testing train_and_visualize (classification)...")
    async with Client(user_transport) as user_client:
        await test_train_classification(user_client, csv_file)

    print("       Testing train_and_visualize (sample weights)...")
    async with Client(user_transport) as user_client:
        await test_sample_weight_training(user_client, csv_file)

    # ── list_models + predict ──────────────────────────────────────────────
    print("[8/9] Testing list_models + predict + merge...")
    async with Client(user_transport) as user_client:
        models = await test_list_models(user_client)
        await test_predict(user_client, models)
        await test_merge_models(user_client, models)

    # ── Admin operations ───────────────────────────────────────────────────
    print("[9/9] Testing admin_manage...")
    async with Client(admin_transport) as admin_client:
        await test_admin_manage(admin_client)

    # Optionally test delete on merged models
    async with Client(user_transport) as user_client:
        fresh_models = await call(client, "list_models", {}) if False else {"models": []}
        # Skip delete test to avoid destroying real data
        # await test_delete_model(user_client, fresh_models.get("models", []))

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}\n")

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        icon = "+" if r.passed else "x"
        line = f"  [{icon}] {status} {r.name} ({r.duration:.1f}s)"
        print(line)
        if r.error:
            for err_line in r.error.split("\n")[:3]:
                print(f"        {err_line}")

    print(f"\n  Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print(f"{'=' * 70}\n")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="AutoMol MCP server integration tests")
    parser.add_argument("--url", required=True, help="MCP server URL (e.g. http://127.0.0.1:8001/mcp)")
    parser.add_argument("--csv", required=True, help="Path to a CSV file with SMILES and properties")
    parser.add_argument("--admin-token", help="Admin token (or use --token-file)")
    parser.add_argument("--user-token", help="User token (or use --token-file)")
    parser.add_argument("--token-file", help="Path to auth_tokens.json (reads admin token from the admin_token.txt sidecar; pass --user-token separately)")

    args = parser.parse_args()

    # Resolve tokens
    admin_token = args.admin_token
    user_token = args.user_token

    if args.token_file and not admin_token:
        # Tokens are stored hashed; the admin plaintext is written once to the
        # admin_token.txt sidecar next to the store. User plaintext tokens are
        # not recoverable from the store — pass --user-token explicitly.
        sidecar = Path(args.token_file).with_name("admin_token.txt")
        if sidecar.exists():
            admin_token = sidecar.read_text().strip()

    if not admin_token:
        print(
            "ERROR: --admin-token required (or --token-file pointing at a store "
            "whose admin_token.txt sidecar still exists)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not user_token:
        print(
            "ERROR: --user-token required (user tokens are hashed at rest and "
            "cannot be recovered from the store)",
            file=sys.stderr,
        )
        sys.exit(1)

    csv_file = str(Path(args.csv).resolve())
    if not Path(csv_file).exists():
        print(f"ERROR: CSV file not found: {csv_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Admin token: {admin_token[:20]}...")
    print(f"User token:  {user_token[:20]}...")

    success = asyncio.run(run_all_tests(args.url, csv_file, admin_token, user_token))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
