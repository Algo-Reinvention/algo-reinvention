# tool_utils.py

from typing import List, Dict, Any, Tuple
import json
import re
import time
from .tool_registry import get_tool_executor
from utils.util import timestamped_print



# Regular expressions for matching <tool_call>...</tool_call> blocks.
# They capture the content between the tags using non-greedy matching.
TOOL_CALL_PATTERN_QWEN = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
TOOL_CALL_PATTERN_NEMOTRON_NANO = re.compile(r'<TOOLCALL>\s*(.*?)\s*</TOOLCALL>', re.DOTALL)

def parse_raw_text_for_tool_calls(raw_text: str, model_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    model_path_lower = model_path.lower()
    if "qwen" in model_path_lower or "nemotron-cascade" in model_path_lower:
        return parse_raw_text_for_tool_calls_qwen(raw_text)
    elif "nemotron-nano" in model_path_lower:
        return parse_raw_text_for_tool_calls_nemotron_nano(raw_text)
    elif "ministral" in model_path_lower:
        return parse_raw_text_for_tool_calls_ministral(raw_text)
    else:
        raise ValueError("Not supported Tool Parsing.")

def parse_raw_text_for_tool_calls_qwen(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse tool calls from the model's raw text output.
    **Current logic: keep only the content before the first <tool_call>
    tag, while still parsing all tool calls in the raw output.**
    
    Args:
        raw_text: Raw text output from the model.
        
    Returns:
        Tuple[str, List[Dict[str, Any]]]: (content_before_first_tool_call, tool_calls_list)
        content_before_first_tool_call contains all text before the first tool_call tag.
        tool_calls_list is the structured tool-call list; all tool calls are parsed
        even if later plain text is discarded.
    """
    
    tool_calls_list = []
    
    first_tool_call_start_index = raw_text.find("<tool_call>")
    
    if first_tool_call_start_index != -1:
        content_without_tool_calls = raw_text[:first_tool_call_start_index].strip()
    else:
        content_without_tool_calls = raw_text.strip()

    matches = TOOL_CALL_PATTERN_QWEN.findall(raw_text)
    
    for i, match_content in enumerate(matches):
        try:
            # Try to parse the JSON payload.
            tool_call_json = json.loads(match_content)
            
            # Make sure the parsed JSON contains 'name' and 'arguments'.
            fn_name = tool_call_json.get('name')
            fn_args = tool_call_json.get('arguments')
            
            if not fn_name or fn_args is None:
                # Ignore malformed payloads and preserve the original flow.
                continue

            # Convert the result into an OpenAI/vLLM-compatible shape.
            if not isinstance(fn_args, str):
                 fn_args = json.dumps(fn_args)
                 
            tool_calls_list.append({
                "id": f"call_{fn_name}_{i}_{time.time():.4f}", # Keep the ID unique.
                "function": {
                    "name": fn_name,
                    "arguments": fn_args
                }
            })
            
        except json.JSONDecodeError as e:
            # Ignore malformed JSON and preserve the original flow.
            pass
        except Exception as e:
            # Ignore unexpected parsing issues and preserve the original flow.
            pass

    content_without_tool_calls = content_without_tool_calls.replace('<|im_end|>', '').strip()
    
    return content_without_tool_calls, tool_calls_list

def parse_raw_text_for_tool_calls_nemotron_nano(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse tool calls in the Nemotron-Nano format.
    Example: <TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Berlin"}}]</TOOLCALL>
    """
    tool_calls_list = []
    
    # 1. Extract the text before the first tool-call tag.
    first_tool_call_start_index = raw_text.find("<TOOLCALL>")
    if first_tool_call_start_index != -1:
        content_before = raw_text[:first_tool_call_start_index].strip()
    else:
        content_before = raw_text.strip()

    # 2. Find all tool-call blocks with regex.
    matches = TOOL_CALL_PATTERN_NEMOTRON_NANO.findall(raw_text)
    
    for match_content in matches:
        try:
            # Nemotron usually returns a JSON list string here: [...]
            calls = json.loads(match_content)
            
            # If the model returns a single object instead of a list, wrap it.
            if isinstance(calls, dict):
                calls = [calls]
            
            if isinstance(calls, list):
                for i, tool_call in enumerate(calls):
                    fn_name = tool_call.get('name')
                    fn_args = tool_call.get('arguments')
                    
                    if not fn_name:
                        continue
                        
                    # Ensure arguments are stringified for OpenAI/vLLM compatibility.
                    if not isinstance(fn_args, str):
                        fn_args_str = json.dumps(fn_args, ensure_ascii=False)
                    else:
                        fn_args_str = fn_args
                    
                    tool_calls_list.append({
                        "id": f"call_{fn_name}_{int(time.time()*1000)}_{i}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": fn_args_str
                        }
                    })
        except Exception:
            # Best-effort parsing: ignore malformed JSON payloads.
            continue

    return content_before, tool_calls_list

def parse_raw_text_for_tool_calls_ministral(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse tool calls in the Ministral format.
    See chat_template.jinja:
      [TOOL_CALLS]tool_name[ARGS]{...}
    Supports optional [CALL_ID] and [TOOL_CALLS][{...}] JSON-list style payloads.
    """
    tool_calls_list = []

    first_tool_call_start_index = raw_text.find("[TOOL_CALLS]")
    if first_tool_call_start_index != -1:
        content_before = raw_text[:first_tool_call_start_index].strip()
    else:
        content_before = raw_text.strip()

    if first_tool_call_start_index == -1:
        return content_before.replace("</s>", "").strip(), tool_calls_list

    decoder = json.JSONDecoder()
    segments = raw_text.split("[TOOL_CALLS]")[1:]

    for i, segment in enumerate(segments):
        payload = segment.strip()
        if not payload:
            continue

        # Support the [TOOL_CALLS][{"name": "...", "arguments": {...}}] style.
        if payload.startswith("[") or payload.startswith("{"):
            try:
                parsed_obj, _ = decoder.raw_decode(payload)
                calls = parsed_obj if isinstance(parsed_obj, list) else [parsed_obj]
                for j, tool_call in enumerate(calls):
                    if not isinstance(tool_call, dict):
                        continue
                    fn_name = tool_call.get("name")
                    fn_args = tool_call.get("arguments")
                    if not fn_name:
                        continue
                    if not isinstance(fn_args, str):
                        fn_args_str = json.dumps(fn_args, ensure_ascii=False)
                    else:
                        fn_args_str = fn_args
                    tool_calls_list.append({
                        "id": f"call_{fn_name}_{int(time.time()*1000)}_{i}_{j}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": fn_args_str,
                        },
                    })
                continue
            except Exception:
                pass

        if "[ARGS]" not in payload:
            continue

        name_and_id, args_part = payload.split("[ARGS]", 1)
        name_and_id = name_and_id.strip()
        args_part = args_part.strip()

        if "[CALL_ID]" in name_and_id:
            fn_name, call_id = name_and_id.split("[CALL_ID]", 1)
            fn_name = fn_name.strip()
            call_id = call_id.strip()
        else:
            fn_name = name_and_id
            call_id = ""

        if not fn_name:
            continue

        try:
            parsed_args, _ = decoder.raw_decode(args_part)
            if isinstance(parsed_args, str):
                fn_args_str = parsed_args
            else:
                fn_args_str = json.dumps(parsed_args, ensure_ascii=False)
        except Exception:
            args_clean = args_part.split("</s>", 1)[0].strip()
            if not args_clean:
                args_clean = "{}"
            fn_args_str = args_clean

        tool_calls_list.append({
            "id": call_id or f"call_{fn_name}_{int(time.time()*1000)}_{i}",
            "type": "function",
            "function": {
                "name": fn_name,
                "arguments": fn_args_str,
            },
        })

    content_before = content_before.replace("</s>", "").strip()
    return content_before, tool_calls_list

def process_tool_calls(
    tool_calls: List[Dict[str, Any]], 
    task_config: Dict[str, Any] # <--- Receive the task configuration dictionary.
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generic tool dispatcher: handle tool-call requests returned by the model,
    execute the tools, and prepare tool-result messages.
    Tool-specific argument parsing and config handling are delegated to each _impl function.
    """
    tool_messages = []
    tool_execution_results = {}
    # print(tool_calls)
    
    for tool_call in tool_calls:
        fn_call = tool_call.get('function')
        if not fn_call:
            continue
            
        fn_name: str = fn_call['name']
        tool_call_id: str = tool_call['id']
        fn_args_json_str: str = fn_call['arguments']
        try:
            # Parse the JSON string into a Python dictionary.
            fn_args: Dict[str, Any] = json.loads(fn_args_json_str) 
        except json.JSONDecodeError as e:
            # If the model returned invalid JSON, emit an error message and skip it.
            timestamped_print(f"Tool Call: Failed to parse arguments for {fn_name}: {e}", "ERROR")
            tool_messages.append({
                "role": "tool",
                "content": json.dumps({"status": "error", "message": f"Arguments JSON decoding error: {e}"}),
                "tool_call_id": tool_call_id,
            })
            continue # Move on to the next tool call.

        try:
            executor = get_tool_executor(fn_name)
            
            if not executor:
                result = {"status": "error", "message": f"Tool '{fn_name}' not found in registry."}
            else:
                # Unified calling interface: pass model arguments and task configuration.
                result = executor(
                    fn_args=fn_args, 
                    task_config=task_config
                )
            
            tool_execution_results[fn_name] = result
            
            # Package the result into a role=tool message.
            tool_messages.append({
                "role": "tool",
                "content": json.dumps(result), 
                "tool_call_id": tool_call_id,
            })
            
        except Exception as e:
            timestamped_print(f"Tool Call: Execution failed for {fn_name}: {e}", "ERROR")
            tool_messages.append({
                "role": "tool",
                "content": json.dumps({"status": "error", "message": f"Internal execution error: {e}"}),
                "tool_call_id": tool_call_id,
            })

    return tool_messages, tool_execution_results

if __name__ == "__main__":
    # Simulated raw Nemotron output.
    test_raw_text = """I will check the weather for you.
<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Berlin"}}, {"name": "get_time", "arguments": {"timezone": "UTC"}}]</TOOLCALL>
Hope this helps!"""

    print("=== Testing Nemotron Parser ===")
    model_path = "/models/nemotron-nano-4b"
    
    content, tool_calls = parse_raw_text_for_tool_calls(test_raw_text, model_path)
    
    print(f"Parsed Content: '{content}'")
    print(f"Tool Calls Count: {len(tool_calls)}")
    
    for i, call in enumerate(tool_calls):
        print(f"\nTool Call {i+1}:")
        print(f"  ID: {call['id']}")
        print(f"  Name: {call['function']['name']}")
        print(f"  Args: {call['function']['arguments']} (Type: {type(call['function']['arguments'])})")

    expected_content = "I will check the weather for you."
    assert content == expected_content
    print("\nTest Passed!")
