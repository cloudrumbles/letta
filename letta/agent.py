# python 3.12
# mypy: strict

import json
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from openai.types.beta.function_tool import FunctionTool as OpenAITool

from letta.constants import (
    CLI_WARNING_PREFIX,
    ERROR_MESSAGE_PREFIX,
    FIRST_MESSAGE_ATTEMPTS,
    FUNC_FAILED_HEARTBEAT_MESSAGE,
    LETTA_CORE_TOOL_MODULE_NAME,
    LETTA_MULTI_AGENT_TOOL_MODULE_NAME,
    LLM_MAX_TOKENS,
    REQ_HEARTBEAT_MESSAGE,
)
from letta.errors import ContextWindowExceededError
from letta.functions.ast_parsers import coerce_dict_args_by_annotations, get_function_annotations_from_source
from letta.functions.functions import get_function_from_module
from letta.helpers import ToolRulesSolver
from letta.interface import AgentInterface
from letta.llm_api.helpers import calculate_summarizer_cutoff, get_token_counts_for_messages, is_context_overflow_error
from letta.llm_api.llm_api_tools import create
from letta.local_llm.utils import num_tokens_from_functions, num_tokens_from_messages
from letta.log import get_logger
from letta.memory import summarize_messages
from letta.orm import User
from letta.orm.enums import ToolType
from letta.schemas.agent import AgentState, AgentStepResponse, UpdateAgent
from letta.schemas.block import BlockUpdate
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import MessageRole
from letta.schemas.memory import ContextWindowOverview, Memory
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_response import ChatCompletionResponse
from letta.schemas.openai.chat_completion_response import Message as ChatCompletionMessage
from letta.schemas.openai.chat_completion_response import UsageStatistics
from letta.schemas.tool import Tool
from letta.schemas.tool_rule import TerminalToolRule
from letta.schemas.usage import LettaUsageStatistics
from letta.services.agent_manager import AgentManager
from letta.services.block_manager import BlockManager
from letta.services.helpers.agent_manager_helper import check_supports_structured_output, compile_memory_metadata_block
from letta.services.job_manager import JobManager
from letta.services.message_manager import MessageManager
from letta.services.passage_manager import PassageManager
from letta.services.provider_manager import ProviderManager
from letta.services.step_manager import StepManager
from letta.services.tool_execution_sandbox import ToolExecutionSandbox
from letta.settings import summarizer_settings
from letta.streaming_interface import StreamingRefreshCLIInterface
from letta.system import get_heartbeat, get_token_limit_warning, package_function_response, package_summarize_message, package_user_message
from letta.utils import (
    count_tokens,
    get_friendly_error_msg,
    get_tool_call_id,
    get_utc_time,
    json_dumps,
    json_loads,
    parse_json,
    printd,
    validate_function_response,
)

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract class for all agents.
    Only one interface is required: step().
    """

    @abstractmethod
    def step(self, messages: Union[Message, List[Message]]) -> LettaUsageStatistics:
        """
        Top-level event message handler for the agent.
        Must be overridden by implementing classes.
        """
        raise NotImplementedError


class Agent(BaseAgent):
    """
    Main agent class. Handles conversation logic, memory management, and
    tool function invocation. Integrates with the AgentManager to track
    agent state.

    The code has been updated with some performance optimizations:
      1) caching in-context messages
      2) skipping deep copies where feasible
      3) avoiding repeated memory compile calls
      4) retaining the same high-level logic and method signatures
    """

    def __init__(
        self,
        interface: Optional[Union[AgentInterface, StreamingRefreshCLIInterface]],
        agent_state: AgentState,
        user: User,
        first_message_verify_mono: bool = True,
    ) -> None:

        self._in_context_cache: Optional[List[Message]] = None
        self._in_context_cache_dirty: bool = True

        # verify memory type
        msg = f"Memory object is not of type Memory: " f"{type(agent_state.memory)}"
        assert isinstance(agent_state.memory, Memory), msg

        self.agent_state: AgentState = agent_state
        self.user: User = user
        self.first_message_verify_mono: bool = first_message_verify_mono

        if self.agent_state.tool_rules:
            for rule in self.agent_state.tool_rules:
                if not isinstance(rule, TerminalToolRule):
                    warnings.warn("Tool rules only work reliably for the latest OpenAI " "models that support structured outputs.")
                    break

        self.tool_rules_solver = ToolRulesSolver(tool_rules=agent_state.tool_rules)

        self.tool_rules_solver: ToolRulesSolver = ToolRulesSolver(tool_rules=self.agent_state.tool_rules)
        self.model = self.agent_state.llm_config.model
        self.supports_structured_output: bool = check_supports_structured_output(
            model=self.model,
            tool_rules=self.agent_state.tool_rules,
        )

        self.block_manager: BlockManager = BlockManager()
        self.message_manager: MessageManager = MessageManager()
        self.passage_manager: PassageManager = PassageManager()
        self.provider_manager: ProviderManager = ProviderManager()
        self.agent_manager: AgentManager = AgentManager()
        self.job_manager: JobManager = JobManager()
        self.step_manager: StepManager = StepManager()

        self.agent_alerted_about_memory_pressure: bool = False
        self.last_function_response: Optional[str] = self.load_last_function_response()

        self.logger = get_logger(agent_state.id)
        self.interface: Optional[Union[AgentInterface, StreamingRefreshCLIInterface]] = interface

        # CHANGED: in-context messages cache for performance
        self._in_context_cache: Optional[List[Message]] = None
        self._in_context_cache_dirty: bool = True

    def _clear_in_context_cache(self) -> None:
        """Invalidate the cached in-context messages."""
        self._in_context_cache = None
        self._in_context_cache_dirty = True

    def _get_in_context_messages_cache(self) -> List[Message]:
        """
        Returns in-context messages from cache if clean; otherwise fetches
        from agent_manager and caches them.
        """
        if self._in_context_cache is not None and not self._in_context_cache_dirty:
            return self._in_context_cache
        messages = self.agent_manager.get_in_context_messages(
            agent_id=self.agent_state.id,
            actor=self.user,
        )
        self._in_context_cache = messages
        self._in_context_cache_dirty = False
        return messages

    def load_last_function_response(self) -> Optional[str]:
        """
        Load the last function response from the agent's in-context message
        history, using the new caching method.
        """
        in_context_messages = self._get_in_context_messages_cache()
        for i in range(len(in_context_messages) - 1, -1, -1):
            msg = in_context_messages[i]
            if msg.role == MessageRole.tool and msg.text:
                try:
                    response_json = json.loads(msg.text)
                    if response_json.get("message"):
                        return str(response_json["message"])
                except (json.JSONDecodeError, KeyError):
                    raise ValueError(f"Invalid JSON format in message: {msg.text}")
        return None

    def update_memory_if_changed(self, new_memory: Memory) -> bool:
        """
        Update internal memory if it has changed. Returns True if updated.
        """
        current_memory_compile = self.agent_state.memory.compile()
        new_memory_compile = new_memory.compile()

        if current_memory_compile == new_memory_compile:
            return False

        for label in self.agent_state.memory.list_block_labels():
            updated_value = new_memory.get_block(label).value
            old_value = self.agent_state.memory.get_block(label).value
            if updated_value != old_value:
                block_id = self.agent_state.memory.get_block(label).id
                self.block_manager.update_block(
                    block_id=block_id,
                    block_update=BlockUpdate(value=updated_value),
                    actor=self.user,
                )

        new_blocks = []
        for block in self.agent_state.memory.get_blocks():
            refreshed = self.block_manager.get_block_by_id(
                block.id,
                actor=self.user,
            )
            new_blocks.append(refreshed)

        self.agent_state.memory = Memory(blocks=new_blocks)
        self.agent_state = self.agent_manager.rebuild_system_prompt(
            agent_id=self.agent_state.id,
            actor=self.user,
        )
        return True

    def execute_tool_and_persist_state(
        self,
        function_name: str,
        function_args: Dict[str, Any],
        target_letta_tool: Tool,
    ) -> str:
        """
        Given a function name, arguments, and a target Tool definition, executes
        the function, handling any memory updates or sandboxing as required.

        Returns:
            str: The function's raw string response (packaged or truncated).
        """
        orig_memory_str = self.agent_state.memory.compile()
        function_response: str = ""

        try:
            # handle core tools
            if target_letta_tool.tool_type == ToolType.LETTA_CORE:
                callable_func = get_function_from_module(
                    LETTA_CORE_TOOL_MODULE_NAME,
                    function_name,
                )
                function_args["self"] = self
                f_res = callable_func(**function_args)
                function_response = str(f_res)

            elif target_letta_tool.tool_type == ToolType.LETTA_MULTI_AGENT_CORE:
                callable_func = get_function_from_module(
                    LETTA_MULTI_AGENT_TOOL_MODULE_NAME,
                    function_name,
                )
                function_args["self"] = self
                f_res = callable_func(**function_args)
                function_response = str(f_res)

            elif target_letta_tool.tool_type == ToolType.LETTA_MEMORY_CORE:
                callable_func = get_function_from_module(
                    LETTA_CORE_TOOL_MODULE_NAME,
                    function_name,
                )
                # CHANGED: skip deep copy, rely on direct usage
                # if your memory tools need full isolation,
                # you can revert to deep copy here.
                agent_state_copy = self.agent_state.__copy__()  # shallow copy
                function_args["agent_state"] = agent_state_copy
                f_res = callable_func(**function_args)
                function_response = str(f_res)
                self.update_memory_if_changed(agent_state_copy.memory)

            else:
                # external custom tools require a sandbox
                ann = get_function_annotations_from_source(
                    target_letta_tool.source_code,
                    function_name,
                )
                function_args = coerce_dict_args_by_annotations(
                    function_args,
                    ann,
                )
                # CHANGED: skipping deep copy for performance
                agent_state_copy = self.agent_state.__copy__()
                agent_state_copy.tools = []
                agent_state_copy.tool_rules = []

                sandbox_res = ToolExecutionSandbox(
                    function_name,
                    function_args,
                    self.user,
                ).run(agent_state=agent_state_copy)
                function_response, updated_state = (
                    sandbox_res.func_return,
                    sandbox_res.agent_state,
                )
                after_str = self.agent_state.memory.compile()
                msg = "Memory should not be modified in a sandbox tool; " "unexpected changes found."
                assert orig_memory_str == after_str, msg

                if updated_state is not None:
                    self.update_memory_if_changed(updated_state.memory)

        except Exception as e:
            function_response = get_friendly_error_msg(
                function_name=function_name,
                exception_name=type(e).__name__,
                exception_message=str(e),
            )

        return function_response

    def _get_ai_reply(
        self,
        message_sequence: List[Message],
        function_call: Optional[str] = None,
        first_message: bool = False,
        stream: bool = False,
        empty_response_retry_limit: int = 3,
        backoff_factor: float = 0.5,
        max_delay: float = 10.0,
        step_count: Optional[int] = None,
    ) -> ChatCompletionResponse:
        """
        Calls the LLM with possible function definitions. Adds some gating
        for which tools are allowed. This includes a retry loop to handle
        empty messages or length issues.
        """
        allowed_tool_names = self.tool_rules_solver.get_allowed_tool_names(last_function_response=self.last_function_response)
        agent_state_tool_jsons = [t.json_schema for t in self.agent_state.tools]
        if not allowed_tool_names:
            allowed_functions = agent_state_tool_jsons
        else:
            allowed_functions = [f for f in agent_state_tool_jsons if f["name"] in allowed_tool_names]

        force_tool_call = None
        # first step "init_tool_rules" possibility
        if (
            step_count is not None
            and step_count == 0
            and not self.supports_structured_output
            and len(self.tool_rules_solver.init_tool_rules) > 0
        ):
            force_tool_call = self.tool_rules_solver.init_tool_rules[0].tool_name
        elif step_count is not None and step_count > 0:
            if len(allowed_tool_names) == 1:
                force_tool_call = allowed_tool_names[0]

        for attempt in range(1, empty_response_retry_limit + 1):
            try:
                response = create(
                    llm_config=self.agent_state.llm_config,
                    messages=message_sequence,
                    user_id=self.agent_state.created_by_id,
                    functions=allowed_functions,
                    function_call=function_call,
                    first_message=first_message,
                    force_tool_call=force_tool_call,
                    stream=stream,
                    stream_interface=self.interface,
                )

                if len(response.choices) == 0 or response.choices[0] is None:
                    raise ValueError(f"API call returned empty message: {response}")

                finish_reason = response.choices[0].finish_reason
                if finish_reason not in ["stop", "function_call", "tool_calls"]:
                    if finish_reason == "length":
                        raise RuntimeError("Finish reason was length " "(max context length hit)")
                    raise ValueError(f"Unexpected finish reason: {finish_reason}")

                return response

            except ValueError as ve:
                if attempt >= empty_response_retry_limit:
                    warnings.warn(f"Retry limit reached. Final error: {ve}")
                    msg = "Retries exhausted; no valid response. " f"Final error: {ve}"
                    raise Exception(msg)
                delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                warnings.warn(f"Attempt {attempt} failed: {ve}. " f"Retrying in {delay} sec...")
                time.sleep(delay)

            except Exception as e:
                raise e

        raise Exception("All retries exhausted; no valid response received.")

    def _handle_ai_response(
        self,
        response_message: ChatCompletionMessage,
        override_tool_call_id: bool = False,
        response_message_id: Optional[str] = None,
    ) -> Tuple[List[Message], bool, bool]:
        """
        Interprets the LLM's ChatCompletionMessage. If it calls a function, we
        run the tool. Otherwise, we store normal text. Returns:
            new_messages, heartbeat_request, function_failed
        """
        messages: List[Message] = []
        function_name: Optional[str] = None

        if response_message_id is not None:
            assert response_message_id.startswith("message-")

        if response_message.function_call or (response_message.tool_calls is not None and len(response_message.tool_calls) > 0):
            if response_message.function_call:
                raise DeprecationWarning(response_message)

            if response_message.tool_calls is not None and len(response_message.tool_calls) > 1:
                self.logger.warning(">1 tool call not supported; using index=0 only\n" f"{response_message.tool_calls}")
                response_message.tool_calls = [response_message.tool_calls[0]]

            if override_tool_call_id or response_message.function_call:
                warnings.warn("Overriding tool call can break streaming consistency.")
                tool_call_id = get_tool_call_id()
                response_message.tool_calls[0].id = tool_call_id
            else:
                tool_call_id = response_message.tool_calls[0].id
                assert tool_call_id

            messages.append(
                Message.dict_to_message(
                    id=response_message_id,
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.created_by_id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )
            self.logger.info(f"Function call message: {messages[-1]}")

            nonnull_content = False
            if response_message.content:
                self.interface.internal_monologue(
                    response_message.content,
                    msg_obj=messages[-1],
                )
                nonnull_content = True

            tool_call = response_message.tool_calls[0].function
            function_name = tool_call.name
            self.logger.info(f"Request to call function {function_name} " f"with tool_call_id: {tool_call_id}")

            target_letta_tool = None
            for t in self.agent_state.tools:
                if t.name == function_name:
                    target_letta_tool = t
                    break

            if not target_letta_tool:
                error_msg = f"No function named {function_name}"
                func_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.created_by_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": func_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )
                self.interface.function_message(
                    f"Error: {error_msg}",
                    msg_obj=messages[-1],
                )
                return messages, False, True

            try:
                raw_function_args = tool_call.arguments
                function_args = parse_json(raw_function_args)
            except Exception:
                err = f"Error parsing JSON for function '{function_name}' " f"arguments: {tool_call.arguments}"
                func_response = package_function_response(False, err)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.created_by_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": func_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )
                self.interface.function_message(
                    f"Error: {err}",
                    msg_obj=messages[-1],
                )
                return messages, False, True

            if "inner_thoughts" in function_args:
                response_message.content = function_args.pop("inner_thoughts")

            # (Still parsing function args)
            # Handle requests for immediate heartbeat
            heartbeat_request = function_args.pop("request_heartbeat", None)

            # Edge case: heartbeat_request is returned as a stringified boolean, we will attempt to parse:
            if isinstance(heartbeat_request, str) and heartbeat_request.lower().strip() == "true":
                heartbeat_request = True

            if heartbeat_request is None:
                heartbeat_request = False

            if not isinstance(heartbeat_request, bool):
                self.logger.warning(
                    f"{CLI_WARNING_PREFIX}'request_heartbeat' arg parsed was not a bool or None, type={type(heartbeat_request)}, value={heartbeat_request}"
                )

            heartbeat_request = bool(function_args.pop("request_heartbeat", 0))

            self.interface.function_message(
                f"Running {function_name}({function_args})",
                msg_obj=messages[-1],
            )
            try:
                function_response = self.execute_tool_and_persist_state(
                    function_name=function_name,
                    function_args=function_args,
                    target_letta_tool=target_letta_tool,
                )
                if function_name in ["conversation_search", "conversation_search_date", "archival_memory_search"]:
                    truncate = False
                else:
                    truncate = True

                return_char_limit = target_letta_tool.return_char_limit
                validated_resp = validate_function_response(
                    function_response,
                    return_char_limit=return_char_limit,
                    truncate=truncate,
                )
                function_args.pop("self", None)
                func_package = package_function_response(True, validated_resp)
                function_failed = False

            except Exception as e:
                function_args.pop("self", None)
                err_msg = get_friendly_error_msg(
                    function_name=function_name,
                    exception_name=type(e).__name__,
                    exception_message=str(e),
                )
                function_response = package_function_response(False, err_msg)
                self.last_function_response = function_response
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.created_by_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )
                self.interface.function_message(
                    f"Ran {function_name}({function_args})",
                    msg_obj=messages[-1],
                )
                self.interface.function_message(
                    f"Error: {err_msg}",
                    msg_obj=messages[-1],
                )
                return messages, False, True

            if validated_resp.startswith(ERROR_MESSAGE_PREFIX):
                func_package = package_function_response(False, validated_resp)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.created_by_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": func_package,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )
                self.interface.function_message(
                    f"Ran {function_name}({function_args})",
                    msg_obj=messages[-1],
                )
                self.interface.function_message(
                    f"Error: {validated_resp}",
                    msg_obj=messages[-1],
                )
                return messages, False, True

            messages.append(
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.created_by_id,
                    model=self.model,
                    openai_message_dict={
                        "role": "tool",
                        "name": function_name,
                        "content": func_package,
                        "tool_call_id": tool_call_id,
                    },
                )
            )
            self.interface.function_message(
                f"Ran {function_name}({function_args})",
                msg_obj=messages[-1],
            )
            self.interface.function_message(
                f"Success: {validated_resp}",
                msg_obj=messages[-1],
            )
            self.last_function_response = func_package

        else:
            # normal text response
            messages.append(
                Message.dict_to_message(
                    id=response_message_id,
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.created_by_id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )
            self.interface.internal_monologue(
                response_message.content,
                msg_obj=messages[-1],
            )
            heartbeat_request = False
            function_failed = False

        # CHANGED: since we added messages, we must clear cache
        self._clear_in_context_cache()

        self.agent_state = self.agent_manager.rebuild_system_prompt(
            agent_id=self.agent_state.id,
            actor=self.user,
        )

        self.tool_rules_solver.update_tool_usage(function_name)
        if self.tool_rules_solver.has_children_tools(function_name):
            heartbeat_request = True
        elif self.tool_rules_solver.is_terminal_tool(function_name):
            heartbeat_request = False

        return messages, heartbeat_request, function_failed

    def step(
        self,
        messages: Union[Message, List[Message]],
        chaining: bool = True,
        max_chaining_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> LettaUsageStatistics:
        """
        Public step method. Takes user messages, calls inner_step, handles
        repeated tool usage if the assistant requests more calls (“heartbeat”).
        """
        next_input_message = messages if isinstance(messages, list) else [messages]
        counter = 0
        total_usage = UsageStatistics()
        step_count = 0

        while True:
            kwargs["first_message"] = False
            kwargs["step_count"] = step_count
            step_response = self.inner_step(next_input_message, **kwargs)

            heartbeat_req = step_response.heartbeat_request
            function_failed = step_response.function_failed
            token_warning = step_response.in_context_memory_warning
            usage = step_response.usage

            total_usage += usage
            step_count += 1
            counter += 1
            self.interface.step_complete()

            save_agent(self)

            if not chaining:
                self.logger.info("No chaining, stopping after one step.")
                break

            if max_chaining_steps is not None and counter > max_chaining_steps:
                self.logger.info(f"Hit max chaining steps, " f"stopping after {counter} steps.")
                break

            if token_warning and summarizer_settings.send_memory_warning_message:
                # memory usage too high, warn the agent
                assert self.agent_state.created_by_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.created_by_id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",
                        "content": get_token_limit_warning(),
                    },
                )
            elif function_failed:
                assert self.agent_state.created_by_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.created_by_id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",
                        "content": get_heartbeat(FUNC_FAILED_HEARTBEAT_MESSAGE),
                    },
                )
            elif heartbeat_req:
                assert self.agent_state.created_by_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.created_by_id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",
                        "content": get_heartbeat(REQ_HEARTBEAT_MESSAGE),
                    },
                )
            else:
                break

        return LettaUsageStatistics(**total_usage.model_dump(), step_count=step_count)

    def inner_step(
        self,
        messages: Union[Message, List[Message]],
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        stream: bool = False,
        step_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        summarize_attempt_count: int = 0,
    ) -> AgentStepResponse:
        """
        Internal method that sends conversation data to the LLM for exactly one
        ChatCompletion call. If there's a function call, we handle it here.
        """
        try:
            job_id = metadata.get("job_id") if metadata else None

            # 0. ensure memory is up to date (and compile only once)
            updated_memory = Memory(
                blocks=[self.block_manager.get_block_by_id(b.id, actor=self.user) for b in self.agent_state.memory.get_blocks()]
            )

            self.update_memory_if_changed(updated_memory)

            # 1. add user message(s)
            if isinstance(messages, Message):
                messages = [messages]
            if not all(isinstance(m, Message) for m in messages):
                raise ValueError("Expected Message or list[Message]. " f"Got: {type(messages)}")

            in_context = self._get_in_context_messages_cache()
            input_sequence = in_context + messages

            if len(input_sequence) > 1 and input_sequence[-1].role != "user":
                self.logger.warning(f"{CLI_WARNING_PREFIX}Attempting to run ChatCompletion " "without a final user message!")

            # 2. call LLM
            response = self._get_ai_reply(
                message_sequence=input_sequence,
                first_message=first_message,
                stream=stream,
                step_count=step_count,
            )
            response_msg = response.choices[0].message

            # Step 3: check if LLM wanted to call a function
            # (if yes) Step 4: call the function
            # (if yes) Step 5: send the info on the function call and function response to LLM
            response_message = response.choices[0].message

            response_message.model_copy()  # TODO why are we copying here?
            all_response_messages, heartbeat_request, function_failed = self._handle_ai_response(
                response_message,
                # TODO this is kind of hacky, find a better way to handle this
                # the only time we set up message creation ahead of time is when streaming is on
                response_message_id=response.id if stream else None,
            )

            # 6. combine user + new assistant messages
            all_new_messages = list(messages) + new_msgs if messages else new_msgs

            # memory usage checks
            current_total_tokens = response.usage.total_tokens
            active_mem_warning = False

            if self.agent_state.llm_config.context_window is None:
                self.logger.warning(f"{CLI_WARNING_PREFIX}context_window not set in config, " f"using default {LLM_MAX_TOKENS['DEFAULT']}")
                if self.model and self.model in LLM_MAX_TOKENS:
                    self.agent_state.llm_config.context_window = LLM_MAX_TOKENS[self.model]
                else:
                    self.agent_state.llm_config.context_window = LLM_MAX_TOKENS["DEFAULT"]

            threshold = summarizer_settings.memory_warning_threshold * int(self.agent_state.llm_config.context_window)
            if current_total_tokens > threshold:
                printd(f"{CLI_WARNING_PREFIX}last response total_tokens " f"({current_total_tokens}) > {threshold}")
                if not self.agent_alerted_about_memory_pressure:
                    active_mem_warning = True
                    self.agent_alerted_about_memory_pressure = True
            else:
                printd(f"last response total_tokens ({current_total_tokens}) < " f"{threshold}")

            step = self.step_manager.log_step(
                actor=self.user,
                provider_name=self.agent_state.llm_config.model_endpoint_type,
                model=self.agent_state.llm_config.model,
                context_window_limit=self.agent_state.llm_config.context_window,
                usage=response.usage,
                provider_id=(
                    self.provider_manager.get_anthropic_override_provider_id()
                    if self.agent_state.llm_config.model_endpoint_type == "anthropic"
                    else None
                ),
                job_id=job_id,
            )

            for m in all_new_messages:
                m.step_id = step.id

            # CHANGED: we just added new messages → clear cache
            self._clear_in_context_cache()

            self.agent_state = self.agent_manager.append_to_in_context_messages(
                all_new_messages,
                agent_id=self.agent_state.id,
                actor=self.user,
            )
            if job_id:
                for m in all_new_messages:
                    self.job_manager.add_message_to_job(
                        job_id=job_id,
                        message_id=m.id,
                        actor=self.user,
                    )

            return AgentStepResponse(
                messages=all_new_messages,
                heartbeat_request=heartbeat_req,
                function_failed=function_failed,
                in_context_memory_warning=active_mem_warning,
                usage=response.usage,
            )

        except Exception as e:
            logger.error(f"step() failed\nmessages = {messages}\nerror = {e}")

            if is_context_overflow_error(e):
                in_context = self._get_in_context_messages_cache()
                if summarize_attempt_count <= summarizer_settings.max_summarizer_retries:
                    logger.warning(
                        "Context window exceeded with limit "
                        f"{self.agent_state.llm_config.context_window}, "
                        "attempting summarization "
                        f"({summarize_attempt_count}/"
                        f"{summarizer_settings.max_summarizer_retries})"
                    )
                    self.summarize_messages_inplace()
                    return self.inner_step(
                        messages=messages,
                        first_message=first_message,
                        first_message_retry_limit=first_message_retry_limit,
                        skip_verify=skip_verify,
                        stream=stream,
                        metadata=metadata,
                        summarize_attempt_count=summarize_attempt_count + 1,
                        step_count=step_count,
                    )
                else:
                    msg = "Ran summarizer but messages are still overflowing " f"context after {summarize_attempt_count} attempts."
                    token_counts = get_token_counts_for_messages(in_context)
                    logger.error(msg)
                    logger.error(f"num_in_context_messages: " f"{len(self.agent_state.message_ids)}")
                    logger.error(f"token_counts: {token_counts}")
                    raise ContextWindowExceededError(
                        msg,
                        details={
                            "num_in_context_messages": len(self.agent_state.message_ids),
                            "in_context_messages_text": [m.text for m in in_context],
                            "token_counts": token_counts,
                        },
                    )
            else:
                logger.error(f"step() failed with unrecognized exception: '{str(e)}'")
                raise e

    def step_user_message(
        self,
        user_message_str: str,
        **kwargs: Any,
    ) -> AgentStepResponse:
        """
        Helper to create a user message from a simple string, with JSON
        metadata. Then calls inner_step() with that message.
        """
        if not user_message_str or not isinstance(user_message_str, str):
            raise ValueError("user_message_str must be a non-empty string, " f"got {type(user_message_str)}")

        user_message_json_str = package_user_message(user_message_str)
        user_message = validate_json(user_message_json_str)
        cleaned_text, name = strip_name_field_from_user_message(user_message)

        openai_msg = {
            "role": "user",
            "content": cleaned_text,
            "name": name,
        }

        if self.agent_state.created_by_id is None:
            raise ValueError("User ID is not set on agent_state.")

        msg_obj = Message.dict_to_message(
            agent_id=self.agent_state.id,
            user_id=self.agent_state.created_by_id,
            model=self.model,
            openai_message_dict=openai_msg,
        )
        return self.inner_step(messages=[msg_obj], **kwargs)

    def summarize_messages_inplace(self) -> None:
        """
        Summarizes older messages to reduce token usage. This modifies
        in-context messages by removing all but the system prompt, then
        prepending a summary user message.
        """
        in_context = self._get_in_context_messages_cache()
        in_context_openai = [m.to_openai_dict() for m in in_context]
        in_context_openai_no_system = in_context_openai[1:]
        token_counts = get_token_counts_for_messages(in_context)

        logger.info(f"System message token count={token_counts[0]}")
        logger.info(f"token_counts_no_system={token_counts[1:]}")

        if in_context_openai[0]["role"] != "system":
            msg = "in_context_messages[0].role should be system, found " f"{in_context_openai[0]['role']}"
            raise RuntimeError(msg)

        if len(in_context_openai_no_system) == 0:
            raise ContextWindowExceededError(
                "No messages to compress for summarization.",
                details={
                    "num_candidate_messages": len(in_context_openai_no_system),
                    "num_total_messages": len(in_context_openai),
                },
            )

        cutoff = calculate_summarizer_cutoff(
            in_context_messages=in_context,
            token_counts=token_counts,
            logger=logger,
        )
        messages_to_summarize = in_context[1:cutoff]
        logger.info(f"Summarizing {len(messages_to_summarize)} messages out of " f"{len(in_context)} total.")

        if self.agent_state.llm_config.context_window is None:
            logger.warning(f"{CLI_WARNING_PREFIX}context_window not set, using default " f"{LLM_MAX_TOKENS['DEFAULT']}")
            if self.model and self.model in LLM_MAX_TOKENS:
                self.agent_state.llm_config.context_window = LLM_MAX_TOKENS[self.model]
            else:
                self.agent_state.llm_config.context_window = LLM_MAX_TOKENS["DEFAULT"]

        summary_text = summarize_messages(
            agent_state=self.agent_state,
            message_sequence_to_summarize=messages_to_summarize,
        )
        logger.info(f"Got summary: {summary_text}")

        all_time_count = self.message_manager.size(
            agent_id=self.agent_state.id,
            actor=self.user,
        )
        remain_count = 1 + len(in_context) - cutoff  # system + leftover
        hidden_count = all_time_count - remain_count
        summary_count = len(messages_to_summarize)

        summary_message_str = package_summarize_message(
            summary_text,
            summary_count,
            hidden_count,
            all_time_count,
        )
        logger.info(f"Packaged summary message: {summary_message_str}")

        prior_len = len(in_context_openai)
        self.agent_state = self.agent_manager.trim_all_in_context_messages_except_system(
            agent_id=self.agent_state.id,
            actor=self.user,
        )

        # CHANGED: clear cache because we trimmed messages
        self._clear_in_context_cache()

        if self.agent_state.created_by_id is None:
            raise ValueError("User ID is not set on agent_state.")

        packed_summary = {
            "role": "user",
            "content": summary_message_str,
        }
        summary_msg_obj = Message.dict_to_message(
            agent_id=self.agent_state.id,
            user_id=self.agent_state.created_by_id,
            model=self.model,
            openai_message_dict=packed_summary,
        )
        self.agent_state = self.agent_manager.prepend_to_in_context_messages(
            messages=[summary_msg_obj],
            agent_id=self.agent_state.id,
            actor=self.user,
        )

        # clearing cache once more, just for consistency
        self._clear_in_context_cache()

        self.agent_alerted_about_memory_pressure = False

        curr_messages = self._get_in_context_messages_cache()

        logger.info(f"Summarizer shrunk messages from {prior_len} -> " f"{len(curr_messages)}")
        before_tokens = sum(token_counts)
        after_tokens = sum(get_token_counts_for_messages(curr_messages))
        logger.info(f"Total token count from {before_tokens} -> {after_tokens}")

    def add_function(self, function_name: str) -> str:
        raise NotImplementedError

    def remove_function(self, function_name: str) -> str:
        raise NotImplementedError

    def migrate_embedding(self, embedding_config: EmbeddingConfig) -> None:
        """
        Migrate the agent to a new embedding config (not implemented).
        """
        raise NotImplementedError

    def get_context_window(self) -> ContextWindowOverview:
        """
        Build a breakdown of the agent's current context window usage,
        including system prompt, memory, in-context messages, etc.
        """
        system_prompt = self.agent_state.system
        num_tokens_system = count_tokens(system_prompt)

        # compile memory once
        core_mem = self.agent_state.memory.compile()
        num_tokens_core_memory = count_tokens(core_mem)

        in_context = self._get_in_context_messages_cache()
        in_context_openai = [m.to_openai_dict() for m in in_context]

        summary_message: Optional[str] = None
        num_tokens_summary_memory = 0

        if (
            len(in_context) > 1
            and in_context[1].role == MessageRole.user
            and in_context[1].text
            and "The following is a summary of the previous " in in_context[1].text
        ):
            summary_message = in_context[1].text
            num_tokens_summary_memory = count_tokens(summary_message)
            if len(in_context_openai) > 2:
                num_tokens_msgs = num_tokens_from_messages(
                    in_context_openai[2:],
                    model=self.model,
                )
            else:
                num_tokens_msgs = 0
        else:
            if len(in_context_openai) > 1:
                num_tokens_msgs = num_tokens_from_messages(
                    in_context_openai[1:],
                    model=self.model,
                )
            else:
                num_tokens_msgs = 0

        agent_mgr_passage_size = self.agent_manager.passage_size(
            actor=self.user,
            agent_id=self.agent_state.id,
        )
        msg_mgr_size = self.message_manager.size(
            actor=self.user,
            agent_id=self.agent_state.id,
        )

        external_mem_summary = compile_memory_metadata_block(
            memory_edit_timestamp=get_utc_time(),
            previous_message_count=msg_mgr_size,
            archival_memory_size=agent_mgr_passage_size,
        )
        num_tokens_external_mem_summary = count_tokens(external_mem_summary)

        agent_state_tool_jsons = [t.json_schema for t in self.agent_state.tools]
        if agent_state_tool_jsons:
            available_functions_defs = [OpenAITool(type="function", function=f) for f in agent_state_tool_jsons]
            num_tokens_functions_defs = num_tokens_from_functions(
                functions=agent_state_tool_jsons,
                model=self.model,
            )
        else:
            available_functions_defs = []
            num_tokens_functions_defs = 0

        num_tokens_used_total = (
            num_tokens_system
            + num_tokens_functions_defs
            + num_tokens_core_memory
            + num_tokens_external_mem_summary
            + num_tokens_summary_memory
            + num_tokens_msgs
        )

        return ContextWindowOverview(
            num_messages=len(in_context),
            num_archival_memory=agent_mgr_passage_size,
            num_recall_memory=msg_mgr_size,
            num_tokens_external_memory_summary=num_tokens_external_mem_summary,
            external_memory_summary=external_mem_summary,
            context_window_size_max=self.agent_state.llm_config.context_window,
            context_window_size_current=num_tokens_used_total,
            num_tokens_system=num_tokens_system,
            system_prompt=system_prompt,
            num_tokens_core_memory=num_tokens_core_memory,
            core_memory=core_mem,
            num_tokens_summary_memory=num_tokens_summary_memory,
            summary_memory=summary_message,
            num_tokens_messages=num_tokens_msgs,
            messages=in_context,
            num_tokens_functions_definitions=num_tokens_functions_defs,
            functions_definitions=available_functions_defs,
        )

    def count_tokens(self) -> int:
        """
        Convenience function to count the total tokens in the agent's
        current context window.
        """
        return self.get_context_window().context_window_size_current


def save_agent(agent: Agent) -> None:
    """
    Save agent state to persistent storage. For simplicity,
    updates the agent's record in the agent manager.
    """
    agent_state = agent.agent_state
    if not isinstance(agent_state.memory, Memory):
        msg = f"Memory is not a Memory object: {type(agent_state.memory)}"
        raise TypeError(msg)

    agent_manager = AgentManager()
    update_agent = UpdateAgent(
        name=agent_state.name,
        tool_ids=[t.id for t in agent_state.tools],
        source_ids=[s.id for s in agent_state.sources],
        block_ids=[b.id for b in agent_state.memory.blocks],
        tags=agent_state.tags,
        system=agent_state.system,
        tool_rules=agent_state.tool_rules,
        llm_config=agent_state.llm_config,
        embedding_config=agent_state.embedding_config,
        message_ids=agent_state.message_ids,
        description=agent_state.description,
        metadata=agent_state.metadata,
    )
    agent_manager.update_agent(
        agent_id=agent_state.id,
        agent_update=update_agent,
        actor=agent.user,
    )


def strip_name_field_from_user_message(
    user_message_text: str,
) -> Tuple[str, Optional[str]]:
    """
    If 'name' exists in the JSON string, remove it and return cleaned
    text plus name value.
    """
    try:
        user_message_json = dict(json_loads(user_message_text))
        name = user_message_json.pop("name", None)
        clean_message = json_dumps(user_message_json)
        return clean_message, name
    except Exception as e:
        print(f"{CLI_WARNING_PREFIX}handling of 'name' field failed: {e}")
        raise e


def validate_json(user_message_text: str) -> str:
    """
    Make sure the string can be loaded as valid JSON.
    Return the canonical JSON serialization if successful.
    """
    try:
        user_message_json = dict(json_loads(user_message_text))
        return json_dumps(user_message_json)
    except Exception as e:
        print(f"{CLI_WARNING_PREFIX}couldn't parse user input: {e}")
        raise e
