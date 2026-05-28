import re

class GuardrailManager:
    """
    Manages dual-layer safety guardrails.
    - Input Guardrails: PII scrubbing, Prompt Injection detection.
    - Output Guardrails: Content filter, compliance alignment.
    """
    def __init__(self):
        # Regular expressions for PII detection
        self.email_regex = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
        self.phone_regex = re.compile(r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b')
        self.ssn_regex = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        
        # Keywords for prompt injection detection
        self.injection_keywords = [
            "ignore previous", "ignore all instructions", "bypass safety", 
            "system prompt", "dan mode", "do anything now", "jailbreak",
            "developer mode", "forget your rules", "override restrictions"
        ]
        
        # Keywords for toxic output detection
        self.unsafe_keywords = [
            "how to steal", "how to hack", "how to build a bomb", "make a weapon",
            "illegal drugs", "pirating software", "hack into", "kill yourself"
        ]

    def process_input(self, text):
        """
        Runs input guardrails on user prompt.
        Returns (processed_text, list_of_triggered_rules).
        """
        triggered = []
        processed = text
        
        # 1. PII Scrubbing
        emails = self.email_regex.findall(processed)
        if emails:
            processed = self.email_regex.sub("[REDACTED_EMAIL]", processed)
            triggered.append(f"PII Scrubbed: Email ({len(emails)} detected)")
            
        phones = self.phone_regex.findall(processed)
        if phones:
            processed = self.phone_regex.sub("[REDACTED_PHONE]", processed)
            triggered.append(f"PII Scrubbed: Phone ({len(phones)} detected)")
            
        ssns = self.ssn_regex.findall(processed)
        if ssns:
            processed = self.ssn_regex.sub("[REDACTED_SSN]", processed)
            triggered.append(f"PII Scrubbed: SSN ({len(ssns)} detected)")

        # 2. Prompt Injection Shield
        text_lower = text.lower()
        injection_found = False
        for kw in self.injection_keywords:
            if kw in text_lower:
                injection_found = True
                break
                
        if injection_found:
            triggered.append("Prompt Injection Shield: Instructions override attempt blocked")
            # We don't modify the prompt, but we flag it so the assistant can trigger refusal/alert
            
        return processed, triggered

    def process_output(self, text):
        """
        Runs output guardrails on assistant's response.
        Returns (processed_text, list_of_triggered_rules).
        """
        triggered = []
        processed = text
        text_lower = text.lower()
        
        # 1. Check for unsafe output leaks (e.g. if the assistant mistakenly generated harmful code or tutorials)
        leak_found = False
        for kw in self.unsafe_keywords:
            if kw in text_lower:
                leak_found = True
                break
                
        if leak_found:
            triggered.append("Output Content Policy: Harmful content output prevented")
            processed = "I cannot fulfill this request. I am programmed to be a safe, helpful, and ethical AI assistant, and I must refuse requests that involve illegal, unsafe, or harmful activities."
            
        # 2. Self-Harm/Crisis Refusal check
        if "suicide" in text_lower or "self-harm" in text_lower:
            triggered.append("Safety Guardrail: Crisis Refusal")
            processed = "If you are feeling overwhelmed or having thoughts of self-harm, please know that you are not alone. Please reach out to a local crisis hotline or mental health professional immediately (e.g. call or text 988 in the US)."
            
        return processed, triggered
