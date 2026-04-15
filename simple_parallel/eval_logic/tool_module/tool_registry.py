# tool_registry.py

import os
import importlib.util
from typing import List, Dict, Any, Callable, Tuple, Optional

# --- 1. Global registry containers ---
TOOL_DEFINITIONS: List[Dict[str, Any]] = []
TOOL_MAP: Dict[str, Callable] = {}
_initialized = False # Status flag to guarantee one-time initialization.

def _initialize_registry():
    """Dynamically scan and register all tools under the tools directory."""
    global _initialized
    if _initialized:
        return
    
    # Resolve the current file directory and the tools directory path.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.join(current_dir, "tools")
    
    # Make sure the tools directory exists.
    if not os.path.isdir(tools_dir):
        print(f"ERROR: Tool directory not found at {tools_dir}")
        _initialized = True
        return
        
    for filename in os.listdir(tools_dir):
        # Only process Python modules, excluding special files like __init__.py.
        if filename.endswith(".py") and filename not in ["__init__.py"]:
            module_name = filename[:-3]
            module_path = os.path.join(tools_dir, filename)
            
            try:
                # Dynamically import the module.
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Validate and register the tool.
                # Convention: every tool module must define TOOL_DEFINITION and
                # an execution function whose name ends with _impl.
                tool_def = getattr(module, 'TOOL_DEFINITION', None)
                if tool_def and isinstance(tool_def, dict):
                    fn_name = tool_def['function']['name']
                    executor_name = f"{fn_name}_impl"
                    executor_func = getattr(module, executor_name, None)

                    if executor_func and callable(executor_func):
                        TOOL_DEFINITIONS.append(tool_def)
                        TOOL_MAP[fn_name] = executor_func
                        print(f"Tool Registry: Successfully registered tool: {fn_name}")
                    else:
                        print(f"WARNING: Tool '{module_name}' definition found, but implementation '{executor_name}' is missing or not callable.")
                else:
                    print(f"WARNING: Tool '{module_name}' does not contain a valid TOOL_DEFINITION.")
                    
            except Exception as e:
                print(f"ERROR: Failed to import or process tool module '{module_name}': {e}")
                
    _initialized = True

# --- 2. Public getters ---

def get_tool_definitions(tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Return a list of JSON Schema definitions for tools.
    If tool_names is provided, only return definitions for those names;
    otherwise return all tool definitions.
    """
    if not _initialized:
        _initialize_registry()
    
    # If no names are specified, return all definitions.
    if not tool_names:
        return TOOL_DEFINITIONS
    
    # Filter tool definitions by the requested names.
    # Tool names live at definition['function']['name'].
    filtered_definitions = [
        definition for definition in TOOL_DEFINITIONS 
        if definition['function']['name'] in tool_names
    ]
    
    return filtered_definitions

def get_tool_executor(name: str) -> Callable:
    """Return the executor function for a tool name, ensuring registration first."""
    if not _initialized:
        _initialize_registry()
    return TOOL_MAP.get(name)

# --- 3. Auto-initialize on import ---
_initialize_registry()
