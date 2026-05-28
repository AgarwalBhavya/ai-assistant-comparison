import os
import time
import sqlite3
from abc import ABC, abstractmethod
from .memory import ChatMemory
from .tools import ToolRegistry
from .guardrails import GuardrailManager

class BaseAssistant(ABC):
    """
    Abstract Base Class for assistants. Coordinates safety guardrails,
    conversational memory, tool executions, and logs performance metrics
    to a local SQLite database for observability.
    """
    def __init__(self, name, system_prompt="You are a helpful, secure, and professional AI assistant."):
        self.name = name
        self.system_prompt = system_prompt
        self.memory = ChatMemory()
        self.tools = ToolRegistry()
        self.guardrails = GuardrailManager()
        self.db_path = os.environ.get("DATABASE_PATH", "database.db")
        self._init_db()

    def _init_db(self):
        """Initializes SQLite database for observability logs."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    assistant_name TEXT,
                    raw_prompt TEXT,
                    processed_prompt TEXT,
                    response TEXT,
                    latency REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    tools_used TEXT,
                    guardrails_triggered TEXT,
                    memory_state TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Observability Database Init Error: {e}")

    def log_telemetry(self, raw_prompt, processed_prompt, response, latency, input_tok, output_tok, tools, guardrails):
        """Saves telemetry log to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO logs 
                (assistant_name, raw_prompt, processed_prompt, response, latency, input_tokens, output_tokens, tools_used, guardrails_triggered, memory_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.name,
                raw_prompt,
                processed_prompt,
                response,
                latency,
                input_tok,
                output_tok,
                ",".join(tools) if tools else "",
                ",".join(guardrails) if guardrails else "",
                self.memory.to_json()
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Observability Database Log Error: {e}")

    def respond(self, user_prompt, custom_system=None):
        """
        Full lifecycle of an assistant turn:
        1. Input Guardrails (PII & Injection detection)
        2. Context Retrieval (Conversation memory)
        3. Model Invocation
        4. Tool detection and execution (multi-turn if needed)
        5. Output Guardrails (Safety alignment)
        6. Memory update and SQLite observability logging
        """
        start_time = time.time()
        guardrails_triggered = []
        tools_triggered = []
        
        system_instructions = custom_system or self.system_prompt
        # Append tool descriptions to the system instructions so the model knows it can call them
        system_instructions += f"\n\n{self.tools.get_tool_descriptions()}"
        
        # 1. Run Input Guardrails
        processed_prompt, input_flags = self.guardrails.process_input(user_prompt)
        guardrails_triggered.extend(input_flags)
        
        # Check if prompt injection is blocked - if so, immediately refuse safely
        if any("Prompt Injection Shield" in f for f in input_flags):
            response = "I cannot process this request because it violates safety guidelines by attempting to override core system rules."
            latency = time.time() - start_time
            self.memory.add_message("user", processed_prompt)
            self.memory.add_message("assistant", response)
            self.log_telemetry(user_prompt, processed_prompt, response, latency, 0, 0, [], guardrails_triggered)
            return {
                "response": response,
                "latency": latency,
                "tools_used": [],
                "guardrails_triggered": guardrails_triggered
            }
            
        # 2. Add processed message to memory (user turn)
        self.memory.add_message("user", processed_prompt)
        
        # 3. Retrieve conversation messages
        messages = self.memory.get_messages(include_system=True, system_prompt=system_instructions)
        
        # Approximate input tokens
        input_tokens = sum(len(msg["content"].split()) for msg in messages)
        
        # 4. Invoke LLM generation
        try:
            model_response = self._generate(messages)
        except Exception as e:
            model_response = f"Model Execution Error: {str(e)}"
            
        # 5. Check and run Tool Calls
        tool_results = self.tools.parse_and_execute(model_response)
        
        if tool_results:
            # Append tool executions to active trackers
            for r in tool_results:
                tools_triggered.append(f"{r['name']}({r['argument']})")
                
            # Send tool execution results back to the model for final synthesis
            tool_context = f"\n\n[SYSTEM TOOL RESULTS: The following are executions of your requested tools:\n"
            for r in tool_results:
                tool_context += f"Tool: {r['name']}({r['argument']}) -> Result: {r['result']}\n"
            tool_context += "Use this tool data to synthesize your final reply to the user. Do not call additional tools unless necessary.]"
            
            # Formulate secondary call to model with tool output
            self.memory.add_message("assistant", model_response)
            self.memory.add_message("system", tool_context)
            
            secondary_messages = self.memory.get_messages(include_system=True, system_prompt=system_instructions)
            try:
                model_response = self._generate(secondary_messages)
            except Exception as e:
                model_response = f"Synthesis Error: {str(e)}"
                
        # 6. Run Output Guardrails
        processed_response, output_flags = self.guardrails.process_output(model_response)
        guardrails_triggered.extend(output_flags)
        
        latency = time.time() - start_time
        output_tokens = len(processed_response.split())
        
        # 7. Finalize memory (replace mock assistant message with synthesized output)
        self.memory.add_message("assistant", processed_response)
        
        # 8. Save logs to SQLite for observability
        self.log_telemetry(
            raw_prompt=user_prompt,
            processed_prompt=processed_prompt,
            response=processed_response,
            latency=latency,
            input_tok=input_tokens,
            output_tok=output_tokens,
            tools=tools_triggered,
            guardrails=guardrails_triggered
        )
        
        return {
            "response": processed_response,
            "latency": latency,
            "tools_used": tools_triggered,
            "guardrails_triggered": guardrails_triggered,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    @abstractmethod
    def _generate(self, messages):
        """Subclasses implement this to call their respective models."""
        pass
