import json
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage
from pydantic import StrictStr

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        api_key = request.headers.get("api-key", "dial_api_key")
        client = AsyncDial(
            api_key = api_key,
            base_url = self.endpoint,
            api_version = "2025-01-01-preview"
        )

        coordination_stage = StageProcessor.open_stage(choice, "🔀 Coordination")

        coordination_request = await self.__prepare_coordination_request(client, request)
        logger.info(f"Coordination decision: {coordination_request}")

        coordination_stage.append_content(
            f"Routing to: **{coordination_request.agent_name}**\n"
            f"Instructions: {coordination_request.additional_instructions or 'none'}"
        )
        StageProcessor.close_stage_safely(coordination_stage)

        agent_stage = StageProcessor.open_stage(choice, "🤖 Agent Response")
        agent_message = await self.__handle_coordination_request(
            coordination_request = coordination_request,
            choice = choice,
            stage = agent_stage,
            request = request,
            api_key = api_key
        ) 

        StageProcessor.close_stage_safely(agent_stage)

        final_message = await self.__final_response(
            client = client,
            choice = choice,
            request = request,
            agent_message = agent_message
        )

        return final_message

    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequest:
        messages = self.__prepare_messages(request, COORDINATION_REQUEST_SYSTEM_PROMPT)

        response = await client.chat.completions.create(
            deployment_name = self.deployment_name,
            messages = messages,
            stream = False,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": CoordinationRequest.model_json_schema()
                    }
                }
            }
        )

        content = response.choices[0].message.content
        logger.debug(f"Raw coordination response: {content}")

        result_dict = json.loads(content)

        return CoordinationRequest.model_validate(result_dict)

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

        for msg in request.messages:
            if msg.role == Role.USER:
                messages.append({
                    "role": "user",
                    "content": msg.content or ""
                })
            else:
                messages.append(msg.dict(exclude_none = True))
 
        return messages

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request,
            api_key: str
    ) -> Message:
        if coordination_request.agent_name == AgentName.UMS:
            gateway = UMSAgentGateway(ums_agent_endpoint = self.ums_agent_endpoint)
            return await gateway.response(
                choice = choice,
                stage = stage,
                request = request,
                additional_instructions = coordination_request.additional_instructions
            )
 
        elif coordination_request.agent_name == AgentName.GPA:
            gpa_endpoint = "http://localhost:8052"
            gateway = GPAGateway(endpoint = gpa_endpoint, api_key = api_key)
            return await gateway.response(
                choice = choice,
                stage = stage,
                request = request,
                additional_instructions = coordination_request.additional_instructions
            )
 
        else:
            logger.warning(f"Unknown agent: {coordination_request.agent_name}, defaulting to GPA")
            gpa_endpoint = "http://localhost:8052"
            gateway = GPAGateway(endpoint = gpa_endpoint, api_key = api_key)
            return await gateway.response(
                choice = choice,
                stage = stage,
                request = request,
                additional_instructions = coordination_request.additional_instructions
            )

    async def __final_response(
            self, client: AsyncDial,
            choice: Choice,
            request: Request,
            agent_message: Message
    ) -> Message:
        messages = self.__prepare_messages(request, FINAL_RESPONSE_SYSTEM_PROMPT)

        original_user_request = request.messages[-1].content or ""
        agent_response_text = agent_message.content or ""

        augmented_content = (
            f"[AGENT RESPONSE CONTEXT]\n"
            f"{agent_response_text}\n"
            f"[END AGENT RESPONSE CONTEXT]\n\n"
            f"[USER REQUEST]\n"
            f"{original_user_request}\n"
            f"[END USER REQUEST]"
        )

        messages[-1]["content"] = augmented_content

        stream = await client.chat.completions.create(
            deployment_name = self.deployment_name,
            messages = messages,
            stream = True
        )

        full_content = ""
 
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                full_content += delta.content
                choice.append_content(delta.content)
 
        # Return the complete assistant message
        return Message(role = Role.ASSISTANT, content = StrictStr(full_content))
