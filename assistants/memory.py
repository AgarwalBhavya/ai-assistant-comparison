import json

class ChatMemory:
    """
    Manages conversational memory with sliding-window capacity and 
    automatic summarization options to prevent context window overflow.
    """
    def __init__(self, window_size=5, summarization_threshold=10):
        self.window_size = window_size
        self.summarization_threshold = summarization_threshold
        self.history = []  # List of dicts: {"role": "user"|"assistant"|"system", "content": str}
        self.context_summary = ""

    def add_message(self, role, content):
        """Adds a message to the history."""
        self.history.append({"role": role, "content": content})
        self._auto_summarize()

    def get_messages(self, include_system=True, system_prompt="You are a helpful assistant."):
        """
        Returns the formatted message list for LLM context,
        incorporating any system prompts, running summaries, and sliding windows.
        """
        messages = []
        
        # 1. Add base system prompt, augmented with context summary if available
        if include_system:
            augmented_system = system_prompt
            if self.context_summary:
                augmented_system += f"\n\n[HISTORICAL SUMMARY: The following is a summary of the earlier part of this conversation: {self.context_summary}]"
            messages.append({"role": "system", "content": augmented_system})
            
        # 2. Add sliding window messages (including system tool outputs)
        active_messages = self.history
        
        # Take only the last window_size * 2 (turns = user + assistant)
        window_limit = self.window_size * 2
        recent_messages = active_messages[-window_limit:] if len(active_messages) > window_limit else active_messages
        
        messages.extend(recent_messages)
        return messages

    def clear(self):
        """Clears memory state."""
        self.history = []
        self.context_summary = ""

    def _auto_summarize(self):
        """
        If the message history exceeds the threshold, compress the oldest messages
        into a running summary and slide them out of the active window.
        """
        active_messages = self.history
        window_limit = self.window_size * 2
        
        if len(active_messages) > self.summarization_threshold:
            # We need to summarize everything except the most recent messages (sliding window)
            to_summarize = active_messages[:-window_limit]
            
            # Simple heuristic summarizer for robustness (can be overridden by LLM-based summarization)
            summary_parts = []
            for i in range(0, len(to_summarize), 2):
                user_msg = to_summarize[i]["content"] if i < len(to_summarize) else ""
                ast_msg = to_summarize[i+1]["content"] if (i+1) < len(to_summarize) else ""
                
                # Truncate for summary description
                u_trunc = user_msg[:50] + "..." if len(user_msg) > 50 else user_msg
                a_trunc = ast_msg[:50] + "..." if len(ast_msg) > 50 else ast_msg
                
                summary_parts.append(f"User asked: '{u_trunc}'. Assistant answered: '{a_trunc}'.")
            
            # Combine into running summary
            self.context_summary = " ".join(summary_parts)
            
            # Keep history trimmed to only the active window for database logs
            # but note we still maintain the summary context!

    def to_json(self):
        """Serializes current history to JSON string."""
        return json.dumps({
            "history": self.history,
            "context_summary": self.context_summary
        })

    def from_json(self, json_str):
        """Loads state from JSON string."""
        try:
            data = json.loads(json_str)
            self.history = data.get("history", [])
            self.context_summary = data.get("context_summary", "")
        except Exception:
            self.clear()
