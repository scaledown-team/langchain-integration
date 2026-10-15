"""Classification output must respect the configured label sets."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware import AgentState
from langchain_core.messages import HumanMessage

from langchain_scaledown import ScaledownClient, ScaledownExtractionMiddleware


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("choice", ["unexpected", [], {}, 42])
async def test_rejects_unconfigured_choice(choice: Any, use_async: bool) -> None:
    client = ScaledownClient(api_key="test")
    middleware = ScaledownExtractionMiddleware(
        {"sentiment": {"labels": ["neutral", {"name": "happy", "rubric": "Happy"}]}},
        client=client,
    )
    with patch.object(
        client, "classify", return_value={"answers": {"sentiment": {"choice": choice}}}
    ):
        state: AgentState[Any] = {"messages": [HumanMessage("Hello")]}
        if use_async:
            update = await middleware.abefore_agent(state, MagicMock())
        else:
            update = middleware.before_agent(state, MagicMock())
    assert update is None


@pytest.mark.parametrize("answers", [[{"choice": "happy"}], "bad"])
def test_malformed_answers_fail_open(answers: Any) -> None:
    client = ScaledownClient(api_key="test")
    middleware = ScaledownExtractionMiddleware(
        {"sentiment": {"labels": ["happy"]}}, client=client
    )
    with patch.object(client, "classify", return_value={"answers": answers}):
        assert (
            middleware.before_agent({"messages": [HumanMessage("Hello")]}, MagicMock())
            is None
        )


def test_unrequested_answers_do_not_override_extractive_fields() -> None:
    client = ScaledownClient(api_key="test")
    middleware = ScaledownExtractionMiddleware(
        {"order": "Order number", "sentiment": {"labels": [{"name": "happy"}]}},
        client=client,
    )
    with (
        patch.object(client, "extract", return_value={"entities": []}),
        patch.object(
            client,
            "classify",
            return_value={
                "answers": {
                    "order": {"choice": "fabricated"},
                    "sentiment": {"choice": "happy", "confidence": 0.9},
                }
            },
        ),
    ):
        update = middleware.before_agent(
            {"messages": [HumanMessage("Hello")]}, MagicMock()
        )
    assert update is not None
    record = update["scaledown_extraction"]
    assert record["fields"] == {"order": None, "sentiment": "happy"}
    assert [entity["type"] for entity in record["entities"]] == ["sentiment"]
