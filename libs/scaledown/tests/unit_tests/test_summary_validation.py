"""Invalid API summaries must never replace conversation history."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_scaledown import (
    ScaledownAPIError,
    ScaledownClient,
    ScaledownSummarizationMiddleware,
)
from tests.unit_tests.test_middleware import agent_model, conversation


@pytest.mark.parametrize("summary", [None, 123, False, [], {}, "", " \n\t"])
def test_client_rejects_invalid_summary(summary: Any) -> None:
    client = ScaledownClient(api_key="test")
    with patch.object(client, "_chat_task", return_value={"summary": summary}):
        with pytest.raises(ScaledownAPIError, match="summary"):
            client.summarize("history")


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("summary", [None, "", " \n"])
async def test_invalid_summary_keeps_history(summary: Any, use_async: bool) -> None:
    client = ScaledownClient(api_key="test")
    middleware = ScaledownSummarizationMiddleware(
        agent_model(), trigger=("messages", 4), keep=("messages", 1), client=client
    )
    messages = conversation(4)
    with patch.object(client, "_chat_task", return_value={"summary": summary}):
        if use_async:
            update = await middleware.abefore_model({"messages": messages}, MagicMock())
        else:
            update = middleware.before_model({"messages": messages}, MagicMock())
    assert update is None
    assert [m.content for m in messages] == [m.content for m in conversation(4)]


def test_valid_summary_is_not_stripped() -> None:
    client = ScaledownClient(api_key="test")
    with patch.object(client, "_chat_task", return_value={"summary": " summary\n"}):
        assert client.summarize("history") == " summary\n"
