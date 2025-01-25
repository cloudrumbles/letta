import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import requests

from letta.constants import CLI_WARNING_PREFIX
from letta.errors import LettaConfigurationError, RateLimitExceededError
from letta.llm_api.anthropic import anthropic_bedrock_chat_completions_request, anthropic_chat_completions_request
from letta.llm_api.aws_bedrock import has_valid_aws_credentials
from letta.llm_api.azure_openai import azure_openai_chat_completions_request
from letta.llm_api.google_ai import convert_tools_to_google_ai_format, google_ai_chat_completions_request
from letta.llm_api.helpers import add_inner_thoughts_to_functions, unpack_all_inner_thoughts_from_kwargs
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

# from letta.llm_api.cohere import cohere_chat_completions_request  # If you had it


# Add logging configuration after imports
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
#   2) DATA CLASSES TO AVOID LARGE SIGNATURES
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
    stream_interface: Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]]


#
# ─────────────────────────────────────────────────────────────
#   3) HANDLER PROTOCOL TO ENFORCE MYPY COMPLIANCE
# ─────────────────────────────────────────────────────────────
#
class ProviderHandler(Protocol):
    """Protocol ensures each handler has these methods."""

    def supports_streaming(self) -> bool:
        """Return True if this provider supports streaming."""
        ...

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        """
        Return environment keys required for this provider
        (like 'openai_api_key' or 'azure_api_key').
        """
        ...

    def build_request(self, params: BuildRequestParams) -> Any:
        """Given the build params, return the request object to pass."""
        ...

    def invoke_request(self, request_obj: Any, invoke_params: InvokeRequestParams) -> ChatCompletionResponse:
        """Given the request object & invoke params, call the LLM and return."""
        ...

    def postprocess(
        self,
        response: ChatCompletionResponse,
        llm_config: LLMConfig,
    ) -> ChatCompletionResponse:
        """Optional final pass to modify the response (e.g. unpack thoughts)."""
        ...


#
# ─────────────────────────────────────────────────────────────
#   4) UTILITY: RETRY DECORATOR, TOKEN CHECKS, ETC.
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
    Decorator that retries a function call with exponential
    backoff on given HTTP error codes (e.g. 429).
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        delay = initial_delay
        num_retries = 0
        while True:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.HTTPError as http_err:
                resp = getattr(http_err, "response", None)
                if not resp:
                    raise
                if resp.status_code in error_codes:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise RateLimitExceededError(
                            "Maximum retries exceeded",
                            max_retries=max_retries,
                        )
                    delay *= exponential_base * (1 + (jitter * random.random()))
                    print(f"{CLI_WARNING_PREFIX}Got a rate limit error ('{http_err}'). " f"Waiting {int(delay)}s then retrying...")
                    time.sleep(delay)
                else:
                    raise
            except Exception:
                raise

    return wrapper


def _check_token_limit(
    llm_config: LLMConfig,
    messages: List[Message],
    functions: Optional[List[Dict[str, Any]]],
) -> None:
    """Raise if total tokens from messages + function definitions exceed context."""
    msgs_oai_fmt = [m.to_openai_dict() for m in messages]
    prompt_tokens = num_tokens_from_messages(msgs_oai_fmt, llm_config.model)
    function_tokens = 0
    if functions:
        function_tokens = num_tokens_from_functions(functions, llm_config.model)
    total = prompt_tokens + function_tokens
    if total > llm_config.context_window:
        raise ValueError(f"Request exceeds maximum context length " f"({total} > {llm_config.context_window} tokens)")


def _ensure_model_settings(
    ms: Optional[Any],
) -> ModelSettings:
    """If no model_settings, load from the global settings in letta.settings."""
    if ms is None:
        from letta.settings import model_settings

        assert isinstance(model_settings, ModelSettings)
        return model_settings
    assert isinstance(ms, ModelSettings)
    return ms


def _check_streaming_supported(
    provider: LLMProvider,
    requested_stream: bool,
    supports_streaming: bool,
) -> None:
    """Raise if the user requested streaming but the provider doesn't support it."""
    if requested_stream and not supports_streaming:
        raise NotImplementedError(f"Streaming not implemented for provider '{provider}'.")


#
# ─────────────────────────────────────────────────────────────
#   5) HANDLER IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────
#
class OpenAIHandler(ProviderHandler):
    """Implements the methods for the OpenAI provider."""

    def supports_streaming(self) -> bool:
        return True

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        if llm_config.model_endpoint == "https://api.openai.com/v1":
            return ["openai_api_key"]
        return []

    def build_request(self, params: BuildRequestParams) -> ChatCompletionRequest:
        # handle function_call if it's None but we have functions
        fc = params.function_call
        if fc is None and params.functions and len(params.functions) > 0:
            if params.llm_config.model_endpoint == "https://inference.memgpt.ai":
                fc = "auto"
            else:
                fc = "required"

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
        self,
        request_obj: Any,
        invoke_params: InvokeRequestParams,
    ) -> ChatCompletionResponse:
        request_obj.stream = invoke_params.stream
        ms = invoke_params.model_settings
        llm_config = invoke_params.llm_config
        stream_interface = invoke_params.stream_interface
        if invoke_params.stream:
            if isinstance(stream_interface, AgentChunkStreamingInterface):
                stream_interface.stream_start()
            resp = openai_chat_completions_process_stream(
                url=llm_config.model_endpoint,
                api_key=ms.openai_api_key,
                chat_completion_request=request_obj,
                stream_interface=stream_interface,
            )
        else:
            if isinstance(stream_interface, AgentChunkStreamingInterface):
                stream_interface.stream_start()
            try:
                resp = openai_chat_completions_request(
                    url=llm_config.model_endpoint,
                    api_key=ms.openai_api_key,
                    chat_completion_request=request_obj,
                )
            finally:
                if isinstance(stream_interface, AgentChunkStreamingInterface):
                    stream_interface.stream_end()
        return resp

    def postprocess(
        self,
        response: ChatCompletionResponse,
        llm_config: LLMConfig,
    ) -> ChatCompletionResponse:
        if llm_config.put_inner_thoughts_in_kwargs:
            response = unpack_all_inner_thoughts_from_kwargs(response=response, inner_thoughts_key=INNER_THOUGHTS_KWARG)
        return response


class AzureHandler(ProviderHandler):
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

    def invoke_request(self, request_obj: Any, invoke_params: InvokeRequestParams) -> ChatCompletionResponse:
        _check_streaming_supported(LLMProvider.AZURE, invoke_params.stream, False)
        ms = invoke_params.model_settings
        llm_config = invoke_params.llm_config
        # set model_endpoint from azure_base_url
        llm_config.model_endpoint = ms.azure_base_url
        return azure_openai_chat_completions_request(
            model_settings=ms,
            llm_config=llm_config,
            api_key=ms.azure_api_key,
            chat_completion_request=request_obj,
        )

    def postprocess(self, response: ChatCompletionResponse, llm_config: LLMConfig) -> ChatCompletionResponse:
        if llm_config.put_inner_thoughts_in_kwargs:
            response = unpack_all_inner_thoughts_from_kwargs(response, INNER_THOUGHTS_KWARG)
        return response


class GoogleAIHandler(ProviderHandler):
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
            tools = convert_tools_to_google_ai_format(typed_tools, inner_thoughts_in_kwargs=params.llm_config.put_inner_thoughts_in_kwargs)
        return {
            "contents": [m.to_google_ai_dict() for m in params.messages],
            "tools": tools,
            "generation_config": {"temperature": params.llm_config.temperature},
        }

    def invoke_request(self, request_obj: Any, invoke_params: InvokeRequestParams) -> ChatCompletionResponse:
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

    def postprocess(self, response: ChatCompletionResponse, llm_config: LLMConfig) -> ChatCompletionResponse:
        return response


class AnthropicHandler(ProviderHandler):
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
            tools=[{"type": "function", "function": f} for f in params.functions] if params.functions else None,
            tool_choice=tool_call,
            max_tokens=1024,
            temperature=params.llm_config.temperature,
        )

    def invoke_request(self, request_obj: ChatCompletionRequest, invoke_params: InvokeRequestParams) -> ChatCompletionResponse:
        _check_streaming_supported(LLMProvider.ANTHROPIC, invoke_params.stream, False)
        return anthropic_chat_completions_request(data=request_obj)

    def postprocess(self, response: ChatCompletionResponse, llm_config: LLMConfig) -> ChatCompletionResponse:
        return response


class CohereHandler(ProviderHandler):
    def supports_streaming(self) -> bool:
        return False

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        return []  # if you had cohere_api_key, you'd add it here

    def build_request(self, params: BuildRequestParams) -> Any:
        raise NotImplementedError("Cohere not implemented in original code.")

    def invoke_request(self, request_obj: Any, invoke_params: InvokeRequestParams) -> ChatCompletionResponse:
        raise NotImplementedError("Cohere not implemented in original code.")

    def postprocess(self, response: ChatCompletionResponse, llm_config: LLMConfig) -> ChatCompletionResponse:
        return response


class LocalHandler(ProviderHandler):
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

    def invoke_request(self, request_obj: Any, invoke_params: InvokeRequestParams) -> ChatCompletionResponse:
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

    def postprocess(self, response: ChatCompletionResponse, llm_config: LLMConfig) -> ChatCompletionResponse:
        return response


class GroqHandler(ProviderHandler):
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
                m.to_openai_dict(put_inner_thoughts_in_kwargs=params.llm_config.put_inner_thoughts_in_kwargs) for m in params.messages
            ],
            tools=tools,
            tool_choice=params.function_call,
            user=str(params.user_id) if params.user_id else None,
        )

    def invoke_request(self, request_obj: ChatCompletionRequest, invoke_params: InvokeRequestParams) -> ChatCompletionResponse:
        _check_streaming_supported(LLMProvider.GROQ, invoke_params.stream, False)
        ms = invoke_params.model_settings
        return openai_chat_completions_request(
            api_key=ms.groq_api_key,
            chat_completion_request=request_obj,
        )

    def postprocess(self, response: ChatCompletionResponse, llm_config: LLMConfig) -> ChatCompletionResponse:
        if llm_config.put_inner_thoughts_in_kwargs:
            response = unpack_all_inner_thoughts_from_kwargs(response, INNER_THOUGHTS_KWARG)
        return response


class TogetherHandler(ProviderHandler):
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
            "endpoint_type": "vllm",
            "wrapper": params.llm_config.model_wrapper,
            "user_id": params.user_id,
            "first_message": params.first_message,
        }

    def invoke_request(self, request_obj: Any, invoke_params: InvokeRequestParams) -> ChatCompletionResponse:
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

    def postprocess(self, response: ChatCompletionResponse, llm_config: LLMConfig) -> ChatCompletionResponse:
        return response


class BedrockHandler(ProviderHandler):
    def supports_streaming(self) -> bool:
        return False

    def required_env_keys(self, llm_config: LLMConfig) -> List[str]:
        # We'll check AWS creds in code
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
            tools=[{"type": "function", "function": f} for f in params.functions] if params.functions else None,
            tool_choice=tool_call,
            max_tokens=1024,
        )

    def invoke_request(self, request_obj: Any, invoke_params: InvokeRequestParams) -> ChatCompletionResponse:
        _check_streaming_supported(LLMProvider.BEDROCK, invoke_params.stream, False)
        if not has_valid_aws_credentials():
            raise LettaConfigurationError(message="Invalid or missing AWS credentials. " "Please configure valid AWS credentials.")
        return anthropic_bedrock_chat_completions_request(data=request_obj)

    def postprocess(self, response: ChatCompletionResponse, llm_config: LLMConfig) -> ChatCompletionResponse:
        return response


#
# ─────────────────────────────────────────────────────────────
#   6) THE REGISTRY
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


#
# ─────────────────────────────────────────────────────────────
#   7) create() - FINAL ENTRYPOINT
# ─────────────────────────────────────────────────────────────
#
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
    stream_interface: Optional[Union[AgentRefreshStreamingInterface, AgentChunkStreamingInterface]] = None,
    max_tokens: Optional[int] = None,
    model_settings: Optional[Any] = None,
) -> ChatCompletionResponse:
    """
    Single entrypoint for building & invoking a chat completion with
    any registered provider. Compatible with mypy --strict.
    """
    from letta.utils import printd

    # 1) token limit check
    _check_token_limit(llm_config, messages, functions)

    # 2) unify model settings
    typed_model_settings: ModelSettings = _ensure_model_settings(model_settings)

    # 3) identify provider
    provider_str = llm_config.model_endpoint_type
    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        raise ValueError(f"Unknown provider '{provider_str}'.")

    logger.info(f"Making API call to {provider} at endpoint: {llm_config.model_endpoint}")
    printd(f"Using model {llm_config.model_endpoint_type}, " f"endpoint: {llm_config.model_endpoint}")

    # 4) get the handler from registry
    if provider not in provider_registry:
        raise LettaConfigurationError(
            message=(f"Provider '{provider}' is not in registry."),
            missing_fields=[provider],
        )
    handler = provider_registry[provider]

    # 5) check streaming support
    _check_streaming_supported(provider, stream, handler.supports_streaming())

    # 6) check environment keys
    needed = handler.required_env_keys(llm_config)
    for key in needed:
        val = getattr(typed_model_settings, key, None)
        if val is None:
            raise LettaConfigurationError(
                message=f"Missing key '{key}' for provider '{provider}'.",
                missing_fields=[key],
            )

    # 7) build request
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

    # 8) invoke
    invoke_params = InvokeRequestParams(
        llm_config=llm_config,
        model_settings=typed_model_settings,
        stream=stream,
        stream_interface=stream_interface,
    )

    logger.info(f"Sending request to {provider} model: {llm_config.model}")
    try:
        raw_response = handler.invoke_request(request_obj, invoke_params)
        logger.info(f"Successfully received response from {provider}")
    except Exception as e:
        logger.error(f"Error calling {provider} API: {str(e)}")
        raise

    # 9) postprocess
    final_response = handler.postprocess(raw_response, llm_config)
    return final_response
