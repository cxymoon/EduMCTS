"""Offline checks for the public CLI, portable paths, and generated SFT data."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import llm_client
import main
import mcts
from config import ModelConfig


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Network access is forbidden in offline tests")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    yield
    assert not attempts, "A test attempted network access"


@pytest.fixture
def isolated_cli(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for name in ("main.py", "config.py", "llm_client.py", "mcts.py", "harvester.py"):
        shutil.copyfile(Path(main.__file__).parent / name, project / name)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    return project, unrelated


def run_offline_cli(project, unrelated, *args):
    # Execute the actual entry point in a fresh interpreter; reject socket use
    # even if a future change accidentally makes a request while parsing args.
    runner = """
import pathlib
import runpy
import sys

def forbid_socket(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname"}:
        raise AssertionError("OFFLINE_NETWORK_ATTEMPT")

sys.addaudithook(forbid_socket)
sys.argv = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    env = os.environ.copy()
    env.update({
        "EDUMCTS_BASE_URL": "<YOUR_API_BASE_URL>",
        "EDUMCTS_API_KEY": "<YOUR_API_KEY>",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    result = subprocess.run(
        [sys.executable, "-B", "-c", runner, str(project / "main.py"), *args],
        cwd=unrelated,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert "OFFLINE_NETWORK_ATTEMPT" not in result.stdout + result.stderr
    return result


def test_help_from_another_directory_has_no_side_effects(isolated_cli):
    project, unrelated = isolated_cli
    before = set(project.rglob("*"))

    result = run_offline_cli(project, unrelated, "--help")

    assert result.returncode == 0, result.stderr
    assert "--input" in result.stdout and "--output-dir" in result.stdout
    assert set(project.rglob("*")) == before
    assert list(unrelated.iterdir()) == []


def test_placeholder_config_fails_before_creating_outputs(isolated_cli):
    project, unrelated = isolated_cli
    before = set(project.rglob("*"))

    result = run_offline_cli(project, unrelated, "--mode", "demo")

    assert result.returncode == 2
    assert "EDUMCTS_API_KEY" in result.stderr
    assert "EDUMCTS_BASE_URL" in result.stderr
    assert "before running synthesis" in result.stderr
    assert "Traceback" not in result.stderr
    assert set(project.rglob("*")) == before
    assert list(unrelated.iterdir()) == []


def test_placeholder_config_fails_before_constructing_api_client(monkeypatch):
    constructor = Mock(side_effect=AssertionError("API client must not be created"))
    monkeypatch.setattr(llm_client, "OpenAI", constructor)
    cfg = ModelConfig(base_url="<YOUR_API_BASE_URL>", api_key="<YOUR_API_KEY>")

    with pytest.raises(ValueError, match="EDUMCTS_API_KEY.*EDUMCTS_BASE_URL"):
        llm_client.LLMClient(cfg)

    constructor.assert_not_called()


def test_relative_data_paths_resolve_against_project_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    source = project / "data" / "seed.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps([{"id": "example", "problem": "What is 6 times 7?", "answer": "42"}]),
        encoding="utf-8",
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.setattr(main, "PROJECT_ROOT", project)
    monkeypatch.chdir(unrelated)

    problems = main.load_problems("data/seed.json")
    main.save_outputs([], "output/smoke")

    assert problems == [{"id": "example", "problem": "What is 6 times 7?", "answer": "42"}]
    for name in (
        "raw_search_results.json", "training_data_sharegpt.json",
        "training_data_alpaca.json", "training_data_cot.json",
    ):
        assert json.loads((project / "output" / "smoke" / name).read_text()) == []
    assert list(unrelated.iterdir()) == []


def test_mocked_demo_exports_three_formats_with_reflections(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.setattr(main, "PROJECT_ROOT", project)
    monkeypatch.chdir(unrelated)
    cfg = ModelConfig(base_url="<YOUR_API_BASE_URL>", api_key="dummy-test-key")
    # Only the SDK boundary is mocked. The real client, search, and harvester
    # still execute; endpoint validation is covered separately above.
    monkeypatch.setattr(cfg, "validate", lambda: None)
    monkeypatch.setattr(main, "MODEL_CFG", cfg)
    monkeypatch.setattr(llm_client, "_OPENAI_AVAILABLE", True)
    responses = {
        llm_client.STUDENT_STEP_SYSTEM_PROMPT: "The first draw is red with probability 3/8.",
        llm_client.TEACHER_SYSTEM_PROMPT: "Account for the changed composition after the first draw.",
        llm_client.ROLLOUT_SYSTEM_PROMPT: "The second red draw has probability 2/7, so the product is 3/28.\nFINAL ANSWER: 3/28",
        llm_client.PROGRESS_EVALUATOR_SYSTEM_PROMPT: '{"progress": 1, "reason": "Useful next step"}',
        llm_client.SOLUTION_VERIFIER_SYSTEM_PROMPT: '{"correct": true, "reason": "Correct product"}',
    }

    def complete(**kwargs):
        text = responses[kwargs["messages"][0]["content"]]
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason="stop"
        )])

    create = Mock(side_effect=complete)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    constructor = Mock(return_value=client)
    monkeypatch.setattr(llm_client, "OpenAI", constructor)
    monkeypatch.setattr(mcts.random, "choice", lambda actions: "hint")
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--mode", "demo", "--problems", "1", "--rollouts", "1",
        "--depth", "2", "--workers", "1", "--output-dir", "output/demo",
    ])

    main.main()

    output = project / "output" / "demo"
    formats = {
        name: json.loads((output / f"training_data_{name}.json").read_text(encoding="utf-8"))
        for name in ("sharegpt", "alpaca", "cot")
    }
    assert all(len(rows) == 1 for rows in formats.values())
    sharegpt, alpaca, cot = (formats[name][0] for name in ("sharegpt", "alpaca", "cot"))
    assert [turn["from"] for turn in sharegpt["conversations"]] == ["human", "gpt"]
    assert alpaca["input"] == cot["problem"] == main.demo_problems()[0]["problem"]
    assert cot["answer"] == "3/28"
    text = sharegpt["conversations"][1]["value"]
    assert text == alpaca["output"] == cot["cot"]
    assert '<reflection type="hint">' in text and "</reflection>" in text
    assert text.endswith("FINAL ANSWER: 3/28")
    assert cot["_meta"]["has_teacher_turns"] is True
    assert cot["_meta"]["teacher_turns"] == 1
    constructor.assert_called_once()
    assert create.call_count > 0
    assert list(unrelated.iterdir()) == []
