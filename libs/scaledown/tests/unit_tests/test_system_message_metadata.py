"""Content rewrites should retain the existing system message's metadata."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_scaledown import (
    ScaledownClient,
    ScaledownCompressionMiddleware,
    ScaledownExtractionMiddleware,
)
from tests.unit_tests.test_middleware import make_request


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("mode", ["compress", "extract"])
@pytest.mark.parametrize(
    "content", ["original context", [{"type": "text", "text": "context"}]]
)
async def test_rewrite_preserves_system_metadata(
    use_async: bool, mode: str, content: Any
) -> None:
    client = ScaledownClient(api_key="test")
    system = SystemMessage(
        content=content,
        id="system-id",
        name="instructions",
        additional_kwargs={"custom": "value"},
        response_metadata={"source": "retrieval"},
    )
    original = system.model_dump()
    request = make_request(
        [HumanMessage("question")],
        system,
        state={"scaledown_extraction": {"fields": {"order": "A-1"}}},
    )
    middleware = (
        ScaledownCompressionMiddleware(min_context_chars=0, client=client)
        if mode == "compress"
        else ScaledownExtractionMiddleware(
            {"order": "Order number"}, inject_into_prompt=True, client=client
        )
    )
    handler = AsyncMock() if use_async else MagicMock()
    with patch.object(
        client,
        "compress",
        return_value={"successful": True, "compressed_prompt": "short"},
    ):
        if use_async:
            await middleware.awrap_model_call(request, handler)
        else:
            middleware.wrap_model_call(request, handler)
    rewritten = handler.call_args.args[0].system_message
    assert rewritten is not system
    assert rewritten.content != system.content
    assert rewritten.model_dump(exclude={"content"}) == system.model_dump(
        exclude={"content"}
    )
    assert system.model_dump() == original
