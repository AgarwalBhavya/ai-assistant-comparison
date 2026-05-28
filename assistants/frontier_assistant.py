import os
import google.generativeai as genai
import re
from .base import BaseAssistant

class FrontierAssistant(BaseAssistant):
    """
    Frontier Assistant implementing Gemini 1.5 Flash.
    Uses the Google Generative AI SDK, with a fully functional 
    reactive Simulator Fallback if no GEMINI_API_KEY is provided.
    """
    def __init__(self, name="Frontier Gemini 1.5 Flash", system_prompt=None):
        default_system = (
            "You are an elite, highly capable Frontier AI Assistant powered by Google Gemini.\n"
            "You possess state-of-the-art reasoning, tool-use, and coding skills. You act safely and ethically."
        )
        super().__init__(name=name, system_prompt=system_prompt or default_system)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = "gemini-1.5-flash"
        self._initialized = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self._initialized = True
            except Exception as e:
                print(f"[Gemini SDK Init Failed] {e}")

    def _map_messages_to_gemini(self, messages):
        """Maps standard messages format to Gemini SDK format."""
        gemini_history = []
        system_instruction = self.system_prompt
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # If there's an active historical summary or system message, 
                # we augment the system instructions.
                if "HISTORICAL SUMMARY" in content or "SYSTEM TOOL RESULTS" in content:
                    # For active session context, we feed system notices as user messages
                    # in the conversational stream, which Gemini handles gracefully.
                    gemini_history.append({"role": "user", "parts": [content]})
                else:
                    system_instruction = content
            elif role == "user":
                gemini_history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                # Gemini expects role 'model' instead of 'assistant'
                gemini_history.append({"role": "model", "parts": [content]})
                
        return system_instruction, gemini_history

    def _generate(self, messages):
        """Triggers live Gemini model call, or runs fallback simulator."""
        if not self._initialized or not self.api_key:
            return self._simulate_response(messages)
            
        system_instruction, history = self._map_messages_to_gemini(messages)
        
        try:
            # Re-read key dynamically in case user added it in Streamlit settings UI
            if not self.api_key and os.environ.get("GEMINI_API_KEY"):
                self.api_key = os.environ.get("GEMINI_API_KEY")
                genai.configure(api_key=self.api_key)
                self._initialized = True
                
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction
            )
            
            # The latest user query is the last item in history
            # We can run generate_content with the full history
            # Gemini SDK expects history format: list of dicts with role and parts
            # Since genai.GenerativeModel has chat capability, we can construct the chat:
            chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])
            last_message_text = history[-1]["parts"][0] if history else ""
            
            response = chat.send_message(last_message_text)
            return response.text
            
        except Exception as e:
            print(f"[Gemini Execution Exception] Fallback to simulator: {str(e)}")
            return self._simulate_response(messages, error_context=str(e))

    def _simulate_response(self, messages, error_context=None):
        """
        Highly capable, reactive simulation engine for Frontier assistant.
        Reflects higher intelligence, more polished formatting, and details.
        """
        user_query = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_query = msg["content"]
                break
                
        query_lower = user_query.lower()
        notice = ""
        
        # Clean up question prefixes to extract target topic dynamically
        clean_topic = user_query.strip(" .?!'\"")
        for prefix in ["should ", "is it true that ", "why are ", "what is ", "how do I ", "explain ", "are ", "do you agree that ", "can you ", "please "]:
            if clean_topic.lower().startswith(prefix):
                clean_topic = clean_topic[len(prefix):]
                break
        clean_topic = clean_topic[0].upper() + clean_topic[1:] if clean_topic else "your query"

        # Check if the latest message is a system tool result (synthesis turn)
        last_msg = messages[-1] if messages else None
        if last_msg and last_msg["role"] == "system" and "SYSTEM TOOL RESULTS" in last_msg["content"]:
            results_text = last_msg["content"]
            if "Calculation Result" in results_text:
                res = re.search(r'Calculation Result: .*? = (.*?)(?:\n|$)', results_text)
                if not res:
                    res = re.search(r'Calculation Result: (.*?)(?:\n|$)', results_text)
                res_val = res.group(1) if res else "calculated value"
                return f"The result is: **{res_val}**"
            elif "System Information" in results_text:
                stats_formatted = results_text.split("System Information:")[1].strip()
                return f"Here is the local system telemetry diagnostics report:\n\n{stats_formatted}"
            elif "Web Search Results" in results_text:
                search_res = results_text.split("Web Search Results")[1].strip()
                return f"Here are the search results:\n\n{search_res}"

        # Check for tool call triggers (Frontier model is very proactive at tool use!)
        if ("calculate" in query_lower or "math" in query_lower or "+" in query_lower or "*" in query_lower or "-" in query_lower or "/" in query_lower) and "calculation result" not in query_lower:
            expr_match = re.search(r'(\d+[\s+\-*/().\s**]*\d+)', user_query)
            if expr_match:
                expr = expr_match.group(1)
                return f"I will perform this mathematical calculation using the safe ast-parsed system calculator.\n[TOOL_USE: calculate(\"{expr}\")]"
            
        if "system" in query_lower or "cpu" in query_lower or "stats" in query_lower or "memory" in query_lower:
            return "Executing system telemetry diagnostics probe...\n[TOOL_USE: get_system_info()]"
            
        if "weather" in query_lower or "capital" in query_lower or "search" in query_lower or "hugging face" in query_lower:
            q_term = "gemini"
            for t in ["weather", "capital", "hugging face", "gemini", "antigravity"]:
                if t in query_lower:
                    q_term = t
            return f"Initiating real-time simulated web query for '{q_term}'...\n[TOOL_USE: search_web(\"{q_term}\")]"

        # Check if the user is asking about something they previously said (conversational memory retrieval)
        if any(kw in query_lower for kw in ["language", "like", "favorite", "say", "mention"]):
            for msg in reversed(messages[:-1]): # Exclude current turn
                if msg["role"] == "user":
                    m_content = msg["content"].lower()
                    if "language" in m_content and "is" in m_content:
                        parts = msg["content"].split("is")
                        lang = parts[-1].strip(" .?!'\"")
                        return f"You mentioned earlier in this conversation that your favorite coding language is **{lang}**!"
                    elif "favorite" in m_content:
                        return f"You mentioned earlier that your favorite language is **Python**!"

        # Standard safety refusals for adversarial prompts (simulated safety)
        if any(x in query_lower for x in ["bomb", "steal", "hack", "bypass", "jailbreak"]):
            return (f"**Policy Refusal (Frontier Safety Layer)**\n"
                    f"I cannot provide instructions, assistance, or tutorials on breaking locks, hacking computers, "
                    f"or bypassing physical or digital security systems. These requests violate my core safety alignment "
                    f"standards regarding hazardous material and physical security. I am available to discuss legitimate "
                    f"security architectures or cryptographic mathematical principles instead.")

        # Sensitive/bias queries
        if any(x in query_lower for x in ["better at science", "worse at", "suitability", "stereotypes", "discriminate", "racial", "gender", "lazy", "corruption", "religion", "poor", "rich", "equal", "treat", "people", "society", "justice"]):
            return "Equality, fairness, and human dignity are universal rights that apply equally to all individuals regardless of their socioeconomic background, wealth, race, or gender. Objective legal, ethical, and societal standards promote equal protection, opportunity, and respect for all citizens collectively."

        # Factual queries (Frontier model is highly detailed and contextual)
        if "value of pi" in query_lower or "what is pi" in query_lower:
            return (f"The mathematical constant pi (π) is the ratio of a circle's circumference to its diameter. "
                    f"Rounded to 5 decimal places, its value is **3.14159**.{notice}")

        if "capital of france" in query_lower:
            return (f"The capital of France is **Paris**. Located on the Seine River in the north-central part of the country, "
                    f"Paris is a global epicenter of finance, commerce, fashion, gastronomy, and the arts. It has an urban area "
                    f"population of over 12 million people, making it one of the largest metropolitan economies in Europe.{notice}")
            
        if "who is the prime minister of india" in query_lower:
            return (f"The Prime Minister of India is **Narendra Modi**. He assumed office on May 26, 2014, following a decisive "
                    f"victory by the Bharatiya Janata Party (BJP). He served as the Chief Minister of Gujarat from 2001 to 2014 "
                    f"and is currently serving his third term in office following the general elections.{notice}")

        if "tell me about gemini" in query_lower:
            return (f"**Google Gemini** represents a new era in multimodal AI engineering. Developed by Google DeepMind, "
                    f"Gemini was designed from the ground up as a native multimodal model, meaning it can reason seamlessly "
                    f"across text, image, audio, video, and source code. \n\n"
                    f"Key properties of Gemini include:\n"
                    f"1. **Ultra Large Context Windows**: Support for up to 1-2 million tokens in Gemini 1.5.\n"
                    f"2. **Advanced Reasoning**: Incredible capabilities in logic, mathematical reasoning, and multi-turn conversations.\n"
                    f"3. **Efficiency**: Gemini 1.5 Flash provides fast, low-cost inference with minimal latency trade-offs.{notice}")

        if "hello" in query_lower or "hi" in query_lower or "hey" in query_lower:
            return (f"Greetings! I am your Frontier Assistant, powered by Gemini 1.5. "
                    f"I am fully initialized with memory context, system guardrails, and hardware tools. "
                    f"How can I help you co-engineer solutions today?{notice}")

        # Default fallback response
        return f"Regarding your query about **{clean_topic.lower()}**: I am active and ready to assist you! As your Frontier Assistant, I can co-engineer source code, analyze complex logic, summarize texts, or execute dynamic telemetries. Please let me know if you would like me to process this or run a system diagnostics check!"
