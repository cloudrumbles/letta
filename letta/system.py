import uuid
from typing import Dict, List, Optional
import warnings

from .constants import (
    INITIAL_BOOT_MESSAGE,
    INITIAL_BOOT_MESSAGE_SEND_MESSAGE_FIRST_MSG,
    INITIAL_BOOT_MESSAGE_SEND_MESSAGE_THOUGHT,
    MESSAGE_SUMMARY_WARNING_STR,
)
from .utils import get_local_time, json_dumps


def get_initial_boot_messages(version: str = "startup") -> List[Dict]:
    """Get initial boot messages based on specified version.

    Args:
        version: Boot message version ("startup", "startup_with_send_message", or "startup_with_send_message_gpt35")

    Returns:
        List of message dictionaries
    """
    if version == "startup":
        return [
            {"role": "assistant", "content": INITIAL_BOOT_MESSAGE},
        ]

    elif version == "startup_with_send_message":
        tool_call_id = str(uuid.uuid4())
        return [
            {
                "role": "assistant",
                "content": INITIAL_BOOT_MESSAGE_SEND_MESSAGE_THOUGHT,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "send_message",
                            "arguments": json_dumps({"message": INITIAL_BOOT_MESSAGE_SEND_MESSAGE_FIRST_MSG}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "send_message",
                "content": package_function_response(True, None),
                "tool_call_id": tool_call_id,
            },
        ]

    elif version == "startup_with_send_message_gpt35":
        tool_call_id = str(uuid.uuid4())
        return [
            {
                "role": "assistant",
                "content": "*inner thoughts* Still waiting on the user. Sending a message with function.",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "send_message",
                            "arguments": json_dumps({"message": "Hi, is anyone there?"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "send_message",
                "content": package_function_response(True, None),
                "tool_call_id": tool_call_id,
            },
        ]

    raise ValueError(f"Unknown boot message version: {version}")


# System event packaging functions
def get_heartbeat(reason: str = "Automated timer", include_location: bool = False, location_name: str = "San Francisco, CA, USA") -> str:
    """Package a heartbeat event message."""
    packaged_message = {
        "type": "heartbeat",
        "reason": reason,
        "time": get_local_time(),
    }

    if include_location:
        packaged_message["location"] = location_name

    return json_dumps(packaged_message)


def get_login_event(
    last_login: str = "Never (first login)", include_location: bool = False, location_name: str = "San Francisco, CA, USA"
) -> str:
    """Package a login event message."""
    packaged_message = {
        "type": "login",
        "last_login": last_login,
        "time": get_local_time(),
    }

    if include_location:
        packaged_message["location"] = location_name

    return json_dumps(packaged_message)


# Message packaging functions
def package_user_message(
    user_message: str,
    time: Optional[str] = None,
    include_location: bool = False,
    location_name: str = "San Francisco, CA, USA",
    name: Optional[str] = None,
) -> str:
    """Package a user message with metadata."""
    packaged_message = {
        "type": "user_message",
        "message": user_message,
        "time": time if time else get_local_time(),
    }

    if include_location:
        packaged_message["location"] = location_name

    if name:
        packaged_message["name"] = name

    return json_dumps(packaged_message)


def package_function_response(was_success: bool, response_string: Optional[str], timestamp: Optional[str] = None) -> str:
    """Package a function response with status and timestamp."""
    return json_dumps(
        {
            "status": "OK" if was_success else "Failed",
            "message": response_string,
            "time": timestamp if timestamp else get_local_time(),
        }
    )


def package_system_message(system_message, message_type="system_alert", time=None):
    # error handling for recursive packaging
    try:
        message_json = json.loads(system_message)
        if "type" in message_json and message_json["type"] == message_type:
            warnings.warn(f"Attempted to pack a system message that is already packed. Not packing: '{system_message}'")
            return system_message
    except:
        pass  # do nothing, expected behavior that the message is not JSON

    formatted_time = time if time else get_local_time()
    packaged_message = {
        "type": message_type,
        "message": system_message,
        "time": formatted_time,
    }



# Summary message packaging
def package_summarize_message(
    summary: str, summary_message_count: int, hidden_message_count: int, total_message_count: int, timestamp: Optional[str] = None
) -> str:
    """Package a conversation summary message."""
    context_message = (
        f"Note: prior messages ({hidden_message_count} of {total_message_count} total messages) have been hidden from view due to conversation memory constraints.\n"
        + f"The following is a summary of the previous {summary_message_count} messages:\n {summary}"
    )

    return json_dumps(
        {
            "type": "system_alert",
            "message": context_message,
            "time": timestamp if timestamp else get_local_time(),
        }
    )


def package_summarize_message_no_summary(hidden_message_count: int, timestamp: Optional[str] = None, message: Optional[str] = None) -> str:
    """Package a message indicating hidden messages without summary."""
    context_message = (
        message
        if message
        else f"Note: {hidden_message_count} prior messages with the user have been hidden from view due to conversation memory constraints. "
        "Older messages are stored in Recall Memory and can be viewed using functions."
    )

    return json_dumps(
        {
            "type": "system_alert",
            "message": context_message,
            "time": timestamp if timestamp else get_local_time(),
        }
    )



def get_token_limit_warning():
    formatted_time = get_local_time()
    packaged_message = {
        "type": "system_alert",
        "message": MESSAGE_SUMMARY_WARNING_STR,
        "time": formatted_time,
    }

    return json_dumps(packaged_message)


def unpack_message(packed_message) -> str:
    """Take a packed message string and attempt to extract the inner message content"""

    try:
        message_json = json.loads(packed_message)
    except:
        warnings.warn(f"Was unable to load message as JSON to unpack: '{packed_message}'")
        return packed_message

    if "message" not in message_json:
        if "type" in message_json and message_json["type"] in ["login", "heartbeat"]:
            # This is a valid user message that the ADE expects, so don't print warning
            return packed_message
        warnings.warn(f"Was unable to find 'message' field in packed message object: '{packed_message}'")
        return packed_message
    else:
        message_type = message_json["type"]
        if message_type != "user_message":
            warnings.warn(f"Expected type to be 'user_message', but was '{message_type}', so not unpacking: '{packed_message}'")
            return packed_message
        return message_json.get("message")
