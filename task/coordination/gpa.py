from copy import deepcopy
from typing import Optional, Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, CustomContent, Stage, Attachment
from pydantic import StrictStr

from task.stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"


class GPAGateway:

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        """
        Call GPA with the prepared message history and stream its response back,
        propagating stages and attachments into our MAS Coordinator's choice.
        """
        client = AsyncDial(
            api_key = self.api_key,
            base_url = self.endpoint,
            api_version = "2025-01-01-preview"
        )
 
        messages = self.__prepare_gpa_messages(request, additional_instructions)
 
        stream = await client.chat.completions.create(
            deployment_name = "general-purpose-agent",
            messages = messages,
            stream = True,
            extra_headers = {
                "x-conversation-id": request.headers.get("x-conversation-id", "")
            }
        )
 
        content = ""

        result_custom_content = CustomContent(attachments = [])
 
        stages_map: dict[int, Stage] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
 
            delta = chunk.choices[0].delta
            print(f"[GPA delta] {delta}")
 
            if delta.content:
                content += delta.content
                stage.append_content(delta.content)
 
            if delta.custom_content:
                cc = delta.custom_content
 
                if cc.attachments:
                    for attachment in cc.attachments:
                        result_custom_content.attachments.append(attachment)
 
                if cc.state is not None:
                    result_custom_content.state = cc.state
 
                cc_dict = cc.model_dump(exclude_none=True)
 
                if "stages" in cc_dict:
                    for stg in cc_dict["stages"]:
                        index = stg["index"]
 
                        if index in stages_map:
                            existing_stage = stages_map[index]
 
                            if "content" in stg:
                                existing_stage.append_content(stg["content"])
 
                            if "attachments" in stg:
                                for att in stg["attachments"]:
                                    existing_stage.add_attachment(
                                        Attachment(**att) if isinstance(att, dict) else att
                                    )
 
                            if stg.get("status") == "completed":
                                StageProcessor.close_stage_safely(existing_stage)
                        else:
                            new_stage = StageProcessor.open_stage(
                                choice = choice,
                                name=stg.get("name", f"Step {index}")
                            )
                            stages_map[index] = new_stage
 
                            if "content" in stg:
                                new_stage.append_content(stg["content"])
 
        if result_custom_content.attachments:
            for attachment in result_custom_content.attachments:
                if hasattr(attachment, "dict"):
                    choice.add_attachment(**attachment.dict(exclude_none = True))
                else:
                    choice.add_attachment(**attachment)
 
        if result_custom_content.state is not None:
            choice.state = result_custom_content.state
 
        choice.state = {
            _IS_GPA: True,
            _GPA_MESSAGES: result_custom_content.state
        }
 
        return Message(role = Role.ASSISTANT, content = StrictStr(content))

    def __prepare_gpa_messages(self, request: Request, additional_instructions: Optional[str]) -> list[dict[str, Any]]:
        # This list will hold only the messages relevant to GPA.
        res_messages: list[dict[str, Any]] = []
 
        for i in range(len(request.messages)):
            msg = request.messages[i]
 
            if msg.role == Role.ASSISTANT:
                has_state = (
                    hasattr(msg, "custom_content")
                    and msg.custom_content is not None
                    and hasattr(msg.custom_content, "state")
                    and msg.custom_content.state is not None
                )
 
                if has_state:
                    state = msg.custom_content.state
                    if isinstance(state, dict) and state.get(_IS_GPA) is True:
                        user_msg = request.messages[i - 1]
                        res_messages.append(user_msg.dict(exclude_none=True))
                        copied_msg = deepcopy(msg)
                        if copied_msg.custom_content:
                            copied_msg.custom_content.state = state.get(_GPA_MESSAGES)
                        res_messages.append(copied_msg.dict(exclude_none=True))
 
        last_message = request.messages[-1]
        last_message_dict = last_message.dict(exclude_none=True)
        res_messages.append(last_message_dict)
 
        if additional_instructions:
            current_content = res_messages[-1].get("content", "")
            res_messages[-1]["content"] = f"{current_content}\n\n{additional_instructions}"
 
        return res_messages
