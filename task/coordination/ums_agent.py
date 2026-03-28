import json
from typing import Optional

import httpx
from aidial_sdk.chat_completion import Role, Request, Message, Stage, Choice
from pydantic import StrictStr


_UMS_CONVERSATION_ID = "ums_conversation_id"


class UMSAgentGateway:

    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        ums_conversation_id = self.__get_ums_conversation_id(request)
        if not ums_conversation_id:
            ums_conversation_id = await self.__create_ums_conversation()
            choice.state = {_UMS_CONVERSATION_ID: ums_conversation_id}

        last_message = request.messages[-1]
        user_message = last_message.content

        if additional_instructions:
            user_message = f"{user_message}\n\n{additional_instructions}"

        content = await self.__call_ums_agent(
            conversation_id = ums_conversation_id,
            user_message = user_message,
            stage = stage
        )

        return Message(role = Role.ASSISTANT, content = StrictStr(content))


    def __get_ums_conversation_id(self, request: Request) -> Optional[str]:
        """Extract UMS conversation ID from previous messages if it exists"""
        for message in request.messages:
            if message.role == Role.ASSISTANT:
                if (
                    hasattr(message, 'custom_content')
                    and message.custom_content is not None
                    and hasattr(message.custom_content, 'state')
                    and message.custom_content.state is not None
                ):
                    state = message.custom_content.state
                    if isinstance(state, dict) and _UMS_CONVERSATION_ID in state:
                        return state[_UMS_CONVERSATION_ID]
        # Nothing found — this is a brand new conversation
        return None

    async def __create_ums_conversation(self) -> str:
        """Create a new conversation on UMS agent side"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ums_agent_endpoint}/conversations",
                json = {}
            )
            data = response.json()
            print(f"[UMS create conversation response] {data}")
            return data["id"]

    async def __call_ums_agent(self, conversation_id, user_message, stage):
        accumulated_content = ""

        async with httpx.AsyncClient(timeout = 120.0, http2 = False) as client:
            async with client.stream(
                "POST",
                f"{self.ums_agent_endpoint}/conversations/{conversation_id}/chat",
                json={
                    "message": {
                        "role": "user",
                        "content": user_message
                    },
                    "stream": True
                }
            ) as response:
                try:
                    async for line in response.aiter_lines():
                        print(f"[UMS raw line] {repr(line)}")
                        if not line:
                            continue
                        if line.startswith("data: "):
                            raw = line[6:]
                        else:
                            raw = line
                        if raw.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content_chunk = delta.get("content")
                            if content_chunk:
                                accumulated_content += content_chunk
                                stage.append_content(content_chunk)
                except httpx.RemoteProtocolError:
                    pass

        return accumulated_content