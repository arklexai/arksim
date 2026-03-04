# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from ..core.agent import Agent  # noqa: TID252

logger = logging.getLogger(__name__)

# =============================================================================
# USING YOUR OWN AGENT ENDPOINT
# =============================================================================
#
# To connect
# your own agent running at an external endpoint instead:
#
# 1. Remove the `from ..core.agent import Agent` import above
# 2. Add `import httpx` to the imports
# 3. Replace the code between --- AGENT CHAT LOGIC STARTS HERE --- and
#    --- AGENT CHAT LOGIC ENDS HERE --- with:
#
#    AGENT_ENDPOINT = "<YOUR_AGENT_ENDPOINT>"  # e.g., "http://localhost:8888/chat"
#
#    payload = {
#        "question": last_user_msg.content
#    }
#
#    async with httpx.AsyncClient(timeout=30.0) as client:
#        agent_response = await client.post(
#            AGENT_ENDPOINT,
#            json=payload,
#            headers={"Content-Type": "application/json"}
#        )
#
#    if agent_response.status_code >= 400:
#        raise HTTPException(agent_response.status_code, f"Agent error: {agent_response.text}")
#
#    response = agent_response.json()
#    answer_text = response["answer"]
#
# Note: Adjust the payload structure and response parsing to match your agent's API.
# =============================================================================

AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")
app = FastAPI(title="Chat Completion Wrapper")


class ChatCompletionRequestMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatCompletionRequestMessage]


class ChatCompletionResponse(BaseModel):
    choices: list[dict]


@app.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, authorization: str | None = Header(None)
) -> ChatCompletionResponse:
    # Authorization Check
    if not authorization:
        logger.warning("Request missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.split(" ")
    if len(token) != 2:
        logger.warning("Invalid Authorization header format")
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    if token[1] != AGENT_API_KEY:
        logger.warning("Invalid API token")
        raise HTTPException(status_code=403, detail="Invalid API token")

    # Find the last user message from the messages which contains the simulator query
    try:
        last_user_msg = next(m for m in reversed(request.messages) if m.role == "user")
    except StopIteration:
        logger.warning("No user message found in request")
        raise HTTPException(
            status_code=400, detail="No user message found in request."
        ) from None

    # --- AGENT CHAT LOGIC STARTS HERE ---
    chat_history = [
        {"role": m.role, "content": m.content}
        for m in request.messages
        if m is not last_user_msg and m.role != "system"
    ]
    agent = Agent(history=chat_history)
    answer_text = await agent.invoke(last_user_msg.content)
    # --- AGENT CHAT LOGIC ENDS HERE ---

    # Format response to chat completion format
    return ChatCompletionResponse(
        choices=[
            {
                "message": {"role": "assistant", "content": answer_text},
            }
        ],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
