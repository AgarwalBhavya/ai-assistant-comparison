import os
import re
import ast
import operator
import platform
import psutil
from datetime import datetime

# Safe Math Parser
_OP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        return _OP_MAP[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        return _OP_MAP[type(node.op)](_safe_eval(node.operand))
    else:
        raise TypeError(f"Unsupported mathematical operation: {type(node).__name__}")

def safe_math(expression):
    """Safely evaluates a basic mathematical expression without raw eval."""
    try:
        # Strip letters or dangerous symbols
        clean_expr = re.sub(r'[^0-9+\-*/().\s**]', '', expression)
        if not clean_expr.strip():
            return "Error: Empty or invalid expression."
        node = ast.parse(clean_expr, mode='eval').body
        result = _safe_eval(node)
        return f"Calculation Result: {expression} = {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"

# Mock Web Search Index for rich simulated responses
MOCK_SEARCH_INDEX = {
    "weather": "Weather update: Currently cloudy, 22°C (72°F) with 60% humidity and a gentle breeze of 12 km/h. Forecast predicts clearing skies by evening.",
    "capital": "Geographical Database: The capital of France is Paris. The capital of Japan is Tokyo. The capital of India is New Delhi. The capital of Brazil is Brasília.",
    "hugging face": "Hugging Face is a collaboration platform and community hub for machine learning, famous for the transformers library, HF Hub, and HF Spaces hosting over 500,000 open-source models.",
    "gemini": "Gemini is Google's highly capable family of multimodal generative AI models, offering advanced reasoning, text generation, and multi-turn chat capacities across various sizes (Ultra, Pro, Flash).",
    "antigravity": "Antigravity is a python library module (`import antigravity`) that opens an easter egg comic in a browser. In this context, Antigravity is a premier, highly capable agentic AI coding assistant designed by Google DeepMind.",
    "ai assistant": "AI Assistant best practices include robust guardrails, conversational memory retention, and low-latency tool-use architectures.",
}

def simulated_search(query):
    """Simulates a web search using a local mock search index."""
    query_lower = query.lower()
    matches = []
    for key, val in MOCK_SEARCH_INDEX.items():
        if key in query_lower or query_lower in key:
            matches.append(val)
            
    if matches:
        return f"Web Search Results for '{query}':\n" + "\n".join([f"- {m}" for m in matches])
    else:
        return f"Web Search Results for '{query}': No matching documents found. Showing generic search index: We found multiple references to tech articles, but none matching '{query}' directly."

def system_stats():
    """Extracts local computer system statistics."""
    try:
        cpu_usage = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_usage = mem.percent
        os_name = f"{platform.system()} {platform.release()}"
        python_ver = platform.python_version()
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return (f"System Information:\n"
                f"- OS: {os_name}\n"
                f"- Python Version: {python_ver}\n"
                f"- CPU Usage: {cpu_usage}%\n"
                f"- Memory Usage: {mem_usage}% ({round(mem.used / (1024**3), 2)}GB / {round(mem.total / (1024**3), 2)}GB)\n"
                f"- Local Time: {curr_time}\n"
                f"- Assistant Status: Operational")
    except Exception as e:
        return f"Error retrieving system stats: {str(e)}"

class ToolRegistry:
    """Registry and execution environment for assistant tools."""
    def __init__(self):
        self.tools = {
            "calculate": {
                "func": safe_math,
                "desc": "Evaluates math expressions. Usage: calculate(\"5 + 5\") or calculate(\"10 * 3.14\")"
            },
            "get_system_info": {
                "func": system_stats,
                "desc": "Retrieves current system statistics. Usage: get_system_info()"
            },
            "search_web": {
                "func": simulated_search,
                "desc": "Searches the web for facts. Usage: search_web(\"query\")"
            }
        }

    def get_tool_descriptions(self):
        """Returns readable descriptions of all registered tools."""
        desc = "Available Tools (You can trigger these tools by including special markup in your response: [TOOL_USE: tool_name(\"arguments\")]):\n"
        for name, info in self.tools.items():
            desc += f"- `{name}`: {info['desc']}\n"
        return desc

    def parse_and_execute(self, text):
        """
        Parses text for tool call patterns: [TOOL_USE: tool_name("argument")]
        Executes matches and returns list of (tool_name, argument, result).
        """
        # Look for [TOOL_USE: tool_name("args")]
        pattern = r'\[TOOL_USE:\s*(\w+)\s*\(\s*["\'`](.*?)["\'`]\s*\)\s*\]'
        matches = re.findall(pattern, text)
        
        results = []
        for tool_name, arg in matches:
            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name]["func"](arg)
                    results.append({
                        "name": tool_name,
                        "argument": arg,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "name": tool_name,
                        "argument": arg,
                        "result": f"Error executing tool {tool_name}: {str(e)}"
                    })
            else:
                results.append({
                    "name": tool_name,
                    "argument": arg,
                    "result": f"Error: Tool '{tool_name}' not found."
                })
                
        # Also support parameterless calls e.g., [TOOL_USE: get_system_info()]
        no_arg_pattern = r'\[TOOL_USE:\s*(\w+)\s*\(\s*\)\s*\]'
        no_arg_matches = re.findall(no_arg_pattern, text)
        for tool_name in no_arg_matches:
            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name]["func"]() if tool_name == "get_system_info" else self.tools[tool_name]["func"]("")
                    results.append({
                        "name": tool_name,
                        "argument": "",
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "name": tool_name,
                        "argument": "",
                        "result": f"Error executing tool {tool_name}: {str(e)}"
                    })
                    
        return results
