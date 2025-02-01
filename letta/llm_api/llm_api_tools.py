import logging
import random
import time
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import requests

from letta.constants import CLI_WARNING_PREFIX
from letta.errors import LettaConfigurationError, RateLimitExceededError
from letta.llm_api.anthropic import (
    anthropic_bedrock_chat_completions_request,
    anthropic_chat_completions_process_stream,
    anthropropic_chat_completions_request,
)
from letta.llm_api.aws_bedrock import has_valid_aws_credentials
from letta.llm_api.azure_openai import azure_openai_chat_completions_request
from letta.llm_api.google_ai import (
    convert_tools_to_google_ai_format,
    google_ai_chat_completions_request,
)
from letta.llm_api.helpers import (
    add_inner_thoughts_to_functions,
    unpack_all_inner_thoughts_from_kwargs,
)
from letta.llm_api.openai import (
    build_openai_chat_completions_request,
    openai_chat_completions_process_stream,
    openai_chat_completions_request,
)
from letta.local_llm.chat_completion_proxy import get_chat_completion
from letta.local_llm.constants import INNER_THOUGHTS_KWARG, INNER_THOUGHTS_KWARG_DESCRIPTION
from letta.local_llm.utils import num_tokens_from_functions, num_tokens_from_messages
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_request import ChatCompletionRequest, Tool, cast_message_to_subtype
from letta.schemas.openai.chat_completion_response import ChatCompletionResponse
from letta.settings import ModelSettings
from letta.streaming_interface import AgentChunkStreamingInterface, AgentRefreshStreamingInterface

# constant for default max tokens
DEFAULT_MAX_TOKENS = 1024

# configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


#
# ─────────────────────────────────────────────────────────────
#   1) ENUM FOR PROVIDERS
# ─────────────────────────────────────────────────────────────
#
class LLMProvider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    GOOGLE_AI = "google_ai"
    COHERE = "cohere"
    LOCAL = "local"
    GROQ = "groq"
    TOGETHER = "together"
    BEDROCK = "bedrock"


#
# ─────────────────────────────────────────────────────────────
#   2) DATA CLASSES FOR REQUEST PARAMETERS
# ─────────────────────────────────────────────────────────────
#
@dataclass
class BuildRequestParams:
    """
    A container for all the data needed to build a request
    to any provider.
    """
    llm_config: LLMConfig
    messages: List[Message]
    user_id: Optional[str] = None
    functions: Optional[List[Dict[str, Any]]] = None
    functions_python: Optional[Dict[str, Any]] = None
    function_call: Optional[str] = None
    first_message: bool = False
    force_tool_call: Optional[str] = None
    use_tool_naming: bool = True
    max_tokens: Optional[int] = None


@dataclass
class InvokeRequestParams:
    """
    A container for all data needed to actually invoke
    the request (including streaming flags, model settings, etc).
    """
    llm_config: LLMConfig
    model_settings: ModelSettings
    stream: bool
    stream_interface: Optional[
        Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]
    ]


#
# ─────────────────────────────────────────────────────────────
#   3) HANDLER PROTOCOL (for mypy compliance)
# ─────────────────────────────────────────────────────────────
#
class ProviderHandler(Protocol):
    def supports_streaming(self) -> bool:
        """Return True if this provider supports streaming."""
        ...

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        """
        Return environment keys required for this provider
        (e.g. 'openai_api_key' or 'azure_api_key').
        """
        ...

    def build_request(self, params: BuildRequestParams) -> Any:
        """Build and return the request object for the provider."""
        ...

    def invoke_request(
        self, request_obj: Any, invoke_params: InvokeRequestParams
    ) -> ChatCompletionResponse:
        """Invoke the LLM API call using the request object and params."""
        ...

    def postprocess(
        self, response: ChatCompletionResponse, llm_config: LLMConfig
    ) -> ChatCompletionResponse:
        """Postprocess the response (e.g., unpack inner thoughts)."""
        ...


#
# ─────────────────────────────────────────────────────────────
#   4) UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
#
def retry_with_exponential_backoff(
    func: Callable[..., Any],
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    max_retries: int = 20,
    error_codes: Tuple[int, ...] = (429,),
) -> Callable[..., Any]:
    """
    Decorator to retry a function call with exponential backoff on
    specific HTTP error codes.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        delay = initial_delay
        num_retries = 0
        while True:
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                raise KeyboardInterrupt("User intentionally stopped thread. Stopping...")
            except requests.exceptions.HTTPError as http_err:
                resp = getattr(http_err, "response", None)
                if not resp:
                    raise
                if resp.status_code in error_codes:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise RateLimitExceededError(
                            "Maximum retries exceeded", max_retries=max_retries
                        )
                    delay *= exponential_base * (1 + (jitter * random.random()))
                    logger.warning(
                        f"{CLI_WARNING_PREFIX} got rate limit error ('{http_err}'). "
                        f"Retrying in {int(delay)}s (attempt {num_retries}/{max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    raise
            except Exception:
                raise
    return wrapper


def _check_token_limit(
    llm_config: LLMConfig, messages: List[Message], functions: Optional[List[Dict[str, Any]]]
) -> None:
    """
    Raises an error if the total tokens (from messages and function definitions)
    exceed the model's context window.
    """
    msgs_oai_fmt = [m.to_openai_dict() for m in messages]
    prompt_tokens = num_tokens_from_messages(msgs_oai_fmt, llm_config.model)
    function_tokens = num_tokens_from_functions(functions, llm_config.model) if functions else 0
    total = prompt_tokens + function_tokens
    if total > llm_config.context_window:
        raise ValueError(
            f"Request exceeds maximum context length ({total} > {llm_config.context_window} tokens)"
        )


def _ensure_model_settings(ms: Optional[Any]) -> ModelSettings:
    """
    Returns provided model_settings or loads the global settings.
    """
    if ms is None:
        from letta.settings import model_settings
        assert isinstance(model_settings, ModelSettings)
        return model_settings
    assert isinstance(ms, ModelSettings)
    return ms


def _check_streaming_supported(
    provider: LLMProvider, requested_stream: bool, supports_streaming: bool
) -> None:
    """
    Raises an error if streaming was requested but the provider does not support it.
    """
    if requested_stream and not supports_streaming:
        raise NotImplementedError(f"Streaming not implemented for provider '{provider}'.")


def execute_with_optional_streaming(
    stream: bool,
    stream_interface: Optional[
        Union[AgentChunkStreamingInterface, AgentRefreshStreamingInterface]
    ],
    func: Callable[[], Any],
) -> Any:
    """
    Helper to execute a function with optional streaming interface management.
    If the stream_interface is an AgentChunkStreamingInterface, call stream_start
    before execution and (if not streaming) call stream_end afterwards.
    """
    if isinstance(stream_interface, AgentChunkStreamingInterface):
        stream_interface.stream_start()
        if not stream:
            try:
                result = func()
            finally:
                stream_interface.stream_end()
            return result
        else:
            return func()
    else:
        return func()


#
# ─────────────────────────────────────────────────────────────
#   5) BASE HANDLER CLASS (for shared behavior)
# ─────────────────────────────────────────────────────────────
#
class BaseHandler(ProviderHandler, ABC):
    def postprocess(
        self, response: ChatCompletionResponse, llm_config: LLMConfig
    ) -> ChatCompletionResponse:
        """
        Default postprocessing: unpack inner thoughts if configured.
        """
        if llm_config.put_inner_thoughts_in_kwargs:
            return unpack_all_inner_thoughts_from_kwargs(response, INNER_THOUGHTS_KWARG)
        return response


#
# ─────────────────────────────────────────────────────────────
#   6) HANDLER IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────
#
class OpenAIHandler(BaseHandler):
    def supports_streaming(self) -> bool:
        return True

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        if llm_config.model_endpoint == "https://api.openai.com/v1":
            return ["openai_api_key"]
        return []

    def build_request(self, params: BuildRequestParams) -> ChatCompletionRequest:
        fc = params.function_call
        if fc is None and params.functions and len(params.functions) > 0:
            fc = "auto" if params.llm_config.model_endpoint == "https://inference.memgpt.ai" else "required"
        return build_openai_chat_completions_request(
            llm_config=params.llm_config,
            messages=params.messages,
            user_id=params.user_id,
            functions=params.functions,
            function_call=fc,
            use_tool_naming=params.use_tool_naming,
            max_tokens=params.max_tokens,
        )

    def invoke_request(
        self, request_obj: Any, invoke_params: InvokeRequestParams
    ) -> ChatCompletionResponse:
        request_obj.stream = invoke_params.stream
        ms = invoke_params.model_settings
        llm_config = invoke_params.llm_config
        stream_interface = invoke_params.stream_interface

        def do_request() -> ChatCompletionResponse:
            if invoke_params.stream:
                return openai_chat_completions_process_stream(
                    url=llm_config.model_endpoint,
                    api_key=ms.openai_api_key,
                    chat_completion_request=request_obj,
                    stream_interface=stream_interface,
                )
            else:
                return openai_chat_completions_request(
                    url=llm_config.model_endpoint,
                    api_key=ms.openai_api_key,
                    chat_completion_request=request_obj,
                )

        return execute_with_optional_streaming(invoke_params.stream, stream_interface, do_request)


class AzureHandler(BaseHandler):
    def supports_streaming(self) -> bool:
        return False

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        return ["azure_api_key", "azure_base_url", "azure_api_version"]

    def build_request(self, params: BuildRequestParams) -> ChatCompletionRequest:
        return build_openai_chat_completions_request(
            llm_config=params.llm_config,
            messages=params.messages,
            user_id=params.user_id,
            functions=params.functions,
            function_call=params.function_call,
            use_tool_naming=params.use_tool_naming,
            max_tokens=params.max_tokens,
        )

    def invoke_request(
        self, request_obj: Any, invoke_params: InvokeRequestParams
    ) -> ChatCompletionResponse:
        _check_streaming_supported(LLMProvider.AZURE, invoke_params.stream, False)
        ms = invoke_params.model_settings
        llm_config = invoke_params.llm_config
        # override model_endpoint using azure_base_url
        llm_config.model_endpoint = ms.azure_base_url
        return azure_openai_chat_completions_request(
            model_settings=ms,
            llm_config=llm_config,
            api_key=ms.azure_api_key,
            chat_completion_request=request_obj,
        )


class GoogleAIHandler(BaseHandler):
    def supports_streaming(self) -> bool:
        return False

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        return ["gemini_api_key"]

    def build_request(self, params: BuildRequestParams) -> Dict[str, Any]:
        if not params.use_tool_naming:
            raise NotImplementedError("Only tool calling on Google AI.")
        tools = None
        if params.functions:
            typed_tools = [Tool(type="function", function=f) for f in params.functions]
            tools = convert_tools_to_google_ai_format(
                typed_tools,
                inner_thoughts_in_kwargs=params.llm_config.put_inner_thoughts_in_kwargs,
            )
        return {
            "contents": [m.to_google_ai_dict() for m in params.messages],
            "tools": tools,
            "generation_config": {"temperature": params.llm_config.temperature},
        }

    def invoke_request(
        self, request_obj: Any, invoke_params: InvokeRequestParams
    ) -> ChatCompletionResponse:
        _check_streaming_supported(LLMProvider.GOOGLE_AI, invoke_params.stream, False)
        llm_config = invoke_params.llm_config
        ms = invoke_params.model_settings
        return google_ai_chat_completions_request(
            base_url=llm_config.model_endpoint,
            model=llm_config.model,
            api_key=ms.gemini_api_key,
            data=request_obj,
            inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs,
        )


class AnthropicHandler(BaseHandler):
    def supports_streaming(self) -> bool:
        return False

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        return []

    def build_request(self, params: BuildRequestParams) -> ChatCompletionRequest:
        if not params.use_tool_naming:
            raise NotImplementedError("Only tool calling on Anthropic.")
        tool_call = None
        if params.force_tool_call:
            if not params.functions:
                raise ValueError("force_tool_call requires non-empty functions.")
            tool_call = {"type": "function", "function": {"name": params.force_tool_call}}
        return ChatCompletionRequest(
            model=params.llm_config.model,
            messages=[cast_message_to_subtype(m.to_openai_dict()) for m in params.messages],
            tools=(
                [{"type": "function", "function": f} for f in params.functions]
                if params.functions else None
            ),
            tool_choice=tool_call,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=params.llm_config.temperature,
        )

    def invoke_request(
        self, request_obj: ChatCompletionRequest, invoke_params: InvokeRequestParams
    ) -> ChatCompletionResponse:
        _check_streaming_supported(LLMProvider.ANTHROPIC, invoke_params.stream, False)
        return anthropic_chat_completions_request(data=request_obj)


class CohereHandler(BaseHandler):
    def supports_streaming(self) -> bool:
        return False

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        return []  # if you had a cohere_api_key, add it here

    def build_request(self, params: BuildRequestParams) -> Any:
        raise NotImplementedError("Cohere not implemented in original code.")

    def invoke_request(
        self, request_obj: Any, invoke_params: InvokeRequestParams
    ) -> ChatCompletionResponse:
        raise NotImplementedError("Cohere not implemented in original code.")


class LocalHandler(BaseHandler):
    def supports_streaming(self) -> bool:
        return False

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        return []

    def build_request(self, params: BuildRequestParams) -> Dict[str, Any]:
        return {
            "model": params.llm_config.model,
            "messages": params.messages,
            "functions": params.functions,
            "functions_python": params.functions_python,
            "function_call": params.function_call,
            "context_window": params.llm_config.context_window,
            "endpoint": params.llm_config.model_endpoint,
            "endpoint_type": params.llm_config.model_endpoint_type,
            "wrapper": params.llm_config.model_wrapper,
            "user_id": params.user_id,
            "first_message": params.first_message,
        }

    def invoke_request(
        self, request_obj: Any, invoke_params: InvokeRequestParams
    ) -> ChatCompletionResponse:
        _check_streaming_supported(LLMProvider.LOCAL, invoke_params.stream, False)
        ms = invoke_params.model_settings
        return get_chat_completion(
            model=request_obj["model"],
            messages=request_obj["messages"],
            functions=request_obj["functions"],
            functions_python=request_obj["functions_python"],
            function_call=request_obj["function_call"],
            context_window=request_obj["context_window"],
            endpoint=request_obj["endpoint"],
            endpoint_type=request_obj["endpoint_type"],
            wrapper=request_obj["wrapper"],
            user=str(request_obj["user_id"]) if request_obj["user_id"] else None,
            first_message=request_obj["first_message"],
            auth_type=ms.openllm_auth_type,
            auth_key=ms.openllm_api_key,
        )


class GroqHandler(BaseHandler):
    def supports_streaming(self) -> bool:
        return False

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        if llm_config.model_endpoint == "https://api.groq.com/openai/v1/chat/completions":
            return ["groq_api_key"]
        return []

    def build_request(self, params: BuildRequestParams) -> ChatCompletionRequest:
        if params.llm_config.put_inner_thoughts_in_kwargs and params.functions:
            params.functions = add_inner_thoughts_to_functions(
                params.functions,
                inner_thoughts_key=INNER_THOUGHTS_KWARG,
                inner_thoughts_description=INNER_THOUGHTS_KWARG_DESCRIPTION,
            )
        tools = None
        if params.functions:
            tools = [{"type": "function", "function": f} for f in params.functions]
        return ChatCompletionRequest(
            model=params.llm_config.model,
            messages=[
                m.to_openai_dict(
                    put_inner_thoughts_in_kwargs=params.llm_config.put_inner_thoughts_in_kwargs
                )
                for m in params.messages
            ],
            tools=tools,
            tool_choice=params.function_call,
            user=str(params.user_id) if params.user_id else None,
        )

    def invoke_request(
        self, request_obj: ChatCompletionRequest, invoke_params: InvokeRequestParams
    ) -> ChatCompletionResponse:
        _check_streaming_supported(LLMProvider.GROQ, invoke_params.stream, False)
        ms = invoke_params.model_settings
        return openai_chat_completions_request(
            api_key=ms.groq_api_key,
            chat_completion_request=request_obj,
        )


class TogetherHandler(BaseHandler):
    def supports_streaming(self) -> bool:
        return False

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        if llm_config.model_endpoint == "https://api.together.ai/v1/completions":
            return ["together_api_key"]
        return []

    def build_request(self, params: BuildRequestParams) -> Dict[str, Any]:
        return {
            "model": params.llm_config.model,
            "messages": params.messages,
            "functions": params.functions,
            "functions_python": params.functions_python,
            "function_call": params.function_call,
            "context_window": params.llm_config.context_window,
            "endpoint": params.llm_config.model_endpoint,
            "endpoint_type": params.llm_config.model_endpoint_type,
            "wrapper": params.llm_config.model_wrapper,
            "user_id": params.user_id,
            "first_message": params.first_message,
        }

    def invoke_request(
        self, request_obj: Any, invoke_params: InvokeRequestParams
    ) -> ChatCompletionResponse:
        _check_streaming_supported(LLMProvider.TOGETHER, invoke_params.stream, False)
        ms = invoke_params.model_settings
        return get_chat_completion(
            model=request_obj["model"],
            messages=request_obj["messages"],
            functions=request_obj["functions"],
            functions_python=request_obj["functions_python"],
            function_call=request_obj["function_call"],
            context_window=request_obj["context_window"],
            endpoint=request_obj["endpoint"],
            endpoint_type=request_obj["endpoint_type"],
            wrapper=request_obj["wrapper"],
            user=str(request_obj["user_id"]) if request_obj["user_id"] else None,
            first_message=request_obj["first_message"],
            auth_type="bearer_token",
            auth_key=ms.together_api_key,
        )


class BedrockHandler(BaseHandler):
    def supports_streaming(self) -> bool:
        return False

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        # AWS credentials are checked in code below.
        return []

    def build_request(self, params: BuildRequestParams) -> ChatCompletionRequest:
        if not params.use_tool_naming:
            raise NotImplementedError("Only tool calling supported on Anthropic (Bedrock).")
        tool_call = None
        if params.force_tool_call:
            if not params.functions:
                raise ValueError("force_tool_call requires non-empty functions.")
            tool_call = {"type": "function", "function": {"name": params.force_tool_call}}
        return ChatCompletionRequest(
            model=params.llm_config.model,
            messages=[cast_message_to_subtype(m.to_openai_dict()) for m in params.messages],
            tools=(
                [{"type": "function", "function": f} for f in params.functions]
                if params.functions else None
            ),
            tool_choice=tool_call,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

    def invoke_request(
        self, request_obj: Any, invoke_params: InvokeRequestParams
    ) -> ChatCompletionResponse:
        _check_streaming_supported(LLMProvider.BEDROCK, invoke_params.stream, False)
        if not has_valid_aws_credentials():
            raise LettaConfigurationError(
                message="Invalid or missing AWS credentials. Please configure valid AWS credentials."
            )
        return anthropic_bedrock_chat_completions_request(data=request_obj)


#
# ─────────────────────────────────────────────────────────────
#   7) PROVIDER REGISTRY & ENTRYPOINT
# ─────────────────────────────────────────────────────────────
#
provider_registry: Dict[LLMProvider, ProviderHandler] = {
    LLMProvider.OPENAI: OpenAIHandler(),
    LLMProvider.AZURE: AzureHandler(),
    LLMProvider.ANTHROPIC: AnthropicHandler(),
    LLMProvider.GOOGLE_AI: GoogleAIHandler(),
    LLMProvider.COHERE: CohereHandler(),  # not implemented
    LLMProvider.LOCAL: LocalHandler(),
    LLMProvider.GROQ: GroqHandler(),
    LLMProvider.TOGETHER: TogetherHandler(),
    LLMProvider.BEDROCK: BedrockHandler(),
}


@retry_with_exponential_backoff
def create(
    llm_config: LLMConfig,
    messages: List[Message],
    user_id: Optional[str] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    functions_python: Optional[Dict[str, Any]] = None,
    function_call: Optional[str] = None,
    first_message: bool = False,
    force_tool_call: Optional[str] = None,
    use_tool_naming: bool = True,
    stream: bool = False,
    stream_interface: Optional[
        Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]
    ] = None,
    max_tokens: Optional[int] = None,
    model_settings: Optional[Any] = None,
) -> ChatCompletionResponse:
    """
    Single entrypoint for building & invoking a chat completion with any registered provider.
    """
    # 1) Token limit check
    _check_token_limit(llm_config, messages, functions)

    # 2) Unify model settings
    typed_model_settings: ModelSettings = _ensure_model_settings(model_settings)

    # 3) Identify provider
    provider_str = llm_config.model_endpoint_type
    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        raise ValueError(f"Unknown provider '{provider_str}'.")
    logger.info(f"making api call to {provider} at endpoint: {llm_config.model_endpoint}")
    logger.debug(f"using model {llm_config.model_endpoint_type}, endpoint: {llm_config.model_endpoint}")

    # 4) Get the handler from registry
    if provider not in provider_registry:
        raise LettaConfigurationError(
            message=f"Provider '{provider}' is not in registry.",
            missing_fields=[provider],
        )
    handler = provider_registry[provider]

    # 5) Check streaming support
    _check_streaming_supported(provider, stream, handler.supports_streaming())

    # 6) Check required environment keys
    needed = handler.required_env_keys(llm_config)
    for key in needed:
        val = getattr(typed_model_settings, key, None)
        if val is None:
            raise LettaConfigurationError(
                message=f"Missing key '{key}' for provider '{provider}'.",
                missing_fields=[key],
            )

    # 7) Build request
    build_params = BuildRequestParams(
        llm_config=llm_config,
        messages=messages,
        user_id=user_id,
        functions=functions,
        functions_python=functions_python,
        function_call=function_call,
        first_message=first_message,
        force_tool_call=force_tool_call,
        use_tool_naming=use_tool_naming,
        max_tokens=max_tokens,
    )
    request_obj = handler.build_request(build_params)

    # 8) Invoke the request
    invoke_params = InvokeRequestParams(
        llm_config=llm_config,
        model_settings=typed_model_settings,
        stream=stream,
        stream_interface=stream_interface,
    )

    logger.info(f"sending request to {provider} model: {llm_config.model}")
    try:
        raw_response = handler.invoke_request(request_obj, invoke_params)
        logger.info(f"successfully received response from {provider}")
    except Exception as e:
        logger.error(f"error calling {provider} API: {str(e)}")
        raise

    # 9) Postprocess the response
    final_response = handler.postprocess(raw_response, llm_config)
    return final_response
