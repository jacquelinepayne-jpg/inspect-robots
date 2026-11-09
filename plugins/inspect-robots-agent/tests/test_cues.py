"""``omit_cues`` removes named evaluation cues from what the model reads, and nothing else."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.errors import ConfigError
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task, TaskEnvelope
from inspect_robots.types import Observation
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_agent._cues import (
    _PROMPT_EDITS,
    CUES,
    NOTE_PLACEHOLDER,
    edit_observation_text,
    edit_schemas,
    fill_arguments,
    parse_omit_cues,
    reword_approvals,
)
from inspect_robots_agent.policy import AgentPolicyConfig, _observation_content


def _response(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    call = {"id": f"call_{name}", "type": "function"}
    call["function"] = {"name": name, "arguments": json.dumps(arguments)}  # type: ignore[assignment]
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}


class _Script:
    def __init__(self, responses: list[dict[str, Any]]):
        self.queue = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        payload = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        return httpx.Response(200, json=payload)


def _policy(script: _Script, **kwargs: Any) -> LLMAgentPolicy:
    return LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(script),
        env={},
        clock=lambda: 100.0,
        **kwargs,
    )


def _prompt(images: str = "always", **kwargs: Any) -> str:
    policy = _policy(_Script([_response("done", {"summary": "x"})]), images=images, **kwargs)
    policy.bind(CubePickEmbodiment().info)
    policy.bind_task(TaskEnvelope(name="t", max_steps=200))
    policy.reset(Scene(id="s0", instruction="reach"))
    transcript = policy.transcript()
    assert transcript is not None
    content = transcript[0]["content"]
    assert isinstance(content, str)
    return content


@pytest.mark.parametrize("images", ["always", "on_demand"])
def test_every_prompt_edit_matches_the_default_prompt(images: str) -> None:
    # Guards against template drift: an edit whose text no longer occurs in the
    # prompt would silently leave its cue in place.
    prompt = _prompt(images)
    for _, old, _new in _PROMPT_EDITS:
        assert old.replace("{budget}", "100") in prompt


def test_default_leaves_prompt_and_config_unchanged() -> None:
    assert _prompt() == _prompt(omit_cues=None) == _prompt(omit_cues="")
    policy = _policy(_Script([_response("done", {"summary": "x"})]))
    assert isinstance(policy.config, AgentPolicyConfig)
    assert policy.config.omit_cues == ()


@pytest.mark.parametrize("images", ["always", "on_demand"])
def test_omitting_every_cue_removes_its_prompt_text(images: str) -> None:
    prompt = _prompt(images, omit_cues=",".join(CUES))
    for phrase in (
        "budget",
        "Safety approvers",
        "operator feedback",
        "wish you had known",
        "must include a `note`",
        "trial",
    ):
        assert phrase not in prompt
    assert "You are controlling a real robot embodiment" in prompt


def test_trial_framing_alone_keeps_the_budgets_without_the_word_trial() -> None:
    prompt = _prompt(omit_cues="trial_framing")
    assert "You have a budget of 100 LLM calls." in prompt
    assert "You have 200 environment steps." in prompt
    assert "trial" not in prompt


def test_parse_orders_by_codebook_and_accepts_sequences() -> None:
    assert parse_omit_cues(None) == ()
    assert parse_omit_cues(" note , budget,") == ("budget", "note")
    assert parse_omit_cues(["hindsight", "budget", "budget"]) == ("budget", "hindsight")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("budget,scorer", "unknown cue 'scorer'"),
        ([1], "unknown cue 1"),
        (42, "must be a comma-separated string"),
    ],
)
def test_parse_rejects_unknown_cues_and_types(value: object, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        _policy(_Script([_response("done", {"summary": "x"})]), omit_cues=value)


def test_eval_without_cues_runs_noteless_moves_and_strips_the_tools(tmp_path: Path) -> None:
    script = _Script(
        [
            _response("move_by", {"deltas": {"dx": 0.01}}),
            _response("done", {"summary": "reached"}),
        ]
    )
    task = Task(
        name="t",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=40,
    )
    logs = ir_eval(
        task,
        _policy(script, omit_cues="budget,hindsight,note,trial_framing"),
        CubePickEmbodiment(),
        log_dir=str(tmp_path),
    )

    assert logs[0].eval.policy_config["omit_cues"] == [
        "budget",
        "hindsight",
        "note",
        "trial_framing",
    ]
    first, second = script.requests[:2]
    tools = {tool["function"]["name"]: tool["function"] for tool in first["tools"]}
    for function in tools.values():
        properties = function["parameters"]["properties"]
        assert "note" not in properties and "hindsight" not in properties
        assert "note" not in function["parameters"].get("required", [])
        assert "trial" not in function["description"] and "scorer" not in function["description"]
    observation_text = json.dumps(first["messages"][1:])
    assert "Step budget" not in observation_text and "trial" not in observation_text
    # The noteless move ran: its result is a step count, not core's note error.
    tool_results = [m["content"] for m in second["messages"] if m["role"] == "tool"]
    assert tool_results and "note is required" not in tool_results[0]
    assert "executing move_by" in tool_results[0]


def test_approver_line_is_reworded_only_when_omitted() -> None:
    observation = Observation(
        state={"q": np.zeros(1)},
        extra={"approvals": [{"t": 3, "detail": "delta_clamped"}], "env_step": 4},
    )
    (kept,) = _observation_content(observation, wall_s=5.0)[:1]
    (reworded,) = _observation_content(observation, wall_s=5.0, omit_cues=("safety_approver",))[:1]
    assert "approver: 1 step(s) modified (delta_clamped)." in kept["text"]
    assert "approver" not in reworded["text"]
    assert (
        "1 step(s) were limited to the speed and position limits (delta_clamped)."
        in (reworded["text"])
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "approver: 2 step(s) modified.",
            "2 step(s) were limited to the speed and position limits.",
        ),
        ("operator feedback (step 3): hi", "operator feedback (step 3): hi"),
        ("approver: odd", "approver: odd"),
    ],
)
def test_reword_approvals_keeps_the_count_and_leaves_other_lines(line: str, expected: str) -> None:
    assert reword_approvals(line) == expected


def test_trial_framing_rewords_the_wall_clock_line() -> None:
    text = "Wall clock: 5 s since the trial began."
    assert edit_observation_text(text, ()) == text
    assert edit_observation_text(text, ("trial_framing",)) == "Wall clock: 5 s since the start."


def test_schema_edits_copy_and_tolerate_odd_shapes() -> None:
    schemas: list[dict[str, Any]] = [
        {"type": "function"},
        {"type": "function", "function": {"name": "x", "description": "plain"}},
        {
            "type": "function",
            "function": {"name": "y", "parameters": {"type": "object", "properties": []}},
        },
    ]
    assert edit_schemas(schemas, ()) is schemas
    edited = edit_schemas(schemas, ("note", "trial_framing"))
    assert edited == schemas and edited is not schemas


@pytest.mark.parametrize(
    ("arguments", "omit", "expected"),
    [
        ('{"x": 1}', (), '{"x": 1}'),
        ("not json", ("note",), "not json"),
        ("[1]", ("note",), "[1]"),
        ('{"note": "mine"}', ("note",), '{"note": "mine"}'),
        ('{"x": 1}', ("note",), json.dumps({"x": 1, "note": NOTE_PLACEHOLDER})),
    ],
)
def test_fill_arguments_adds_the_placeholder_only_when_needed(
    arguments: str, omit: tuple[str, ...], expected: str
) -> None:
    assert fill_arguments(arguments, omit) == expected
