import gradio as gr
import os
import requests
import json

# Public Model Identification
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# Retrieve token from environment for hosting
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

def chat_function(message, history, system_prompt):
    """
    Gradio server handler. Formats the conversational history into ChatML
    and queries Hugging Face Inference API.
    """
    # Initialize headers
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    # Reconstruct ChatML
    formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for human, assistant in history:
        formatted += f"<|im_start|>user\n{human}<|im_end|>\n"
        if assistant:
            formatted += f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
            
    formatted += f"<|im_start|>user\n{message}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"

    payload = {
        "inputs": formatted,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        },
        "options": {"wait_for_model": True}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "")
                return text.replace("<|im_end|>", "").strip()
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"].replace("<|im_end|>", "").strip()
            else:
                return f"Error: Unexpected response structure: {str(result)}"
        else:
            return f"API Error (HTTP {response.status_code}): {response.text}\n\nMake sure to add a valid HF_API_TOKEN secret to your Space's environment variables!"
    except Exception as e:
        return f"Exception occurred during execution: {str(e)}"

# Define the Gradio Chatbot Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🚀 Open-Source Qwen 2.5 (0.5B) Public Deployment")
    gr.Markdown(
        f"This is a public, lightweight API and Chat deployment of **{MODEL_ID}** designed to serve as the "
        f"Open-Source Assistant backend. You can chat here directly, or query this interface using Gradio API endpoints!"
    )
    
    with gr.Accordion("System Settings", open=False):
        sys_prompt = gr.Textbox(
            value="You are a helpful, secure, and professional AI assistant running on Hugging Face Spaces.",
            label="System Prompt"
        )
        
    chat = gr.ChatInterface(
        fn=chat_function,
        additional_inputs=[sys_prompt],
        title="Qwen-2.5 Assistant Chat",
        description="Type your message below to interact with the model in real-time."
    )
    
    gr.Markdown("### How to query this Space programmatically via Python:")
    gr.Code(
        language="python",
        value=(
            "import requests\n\n"
            "# Replace with your deployed space URL\n"
            "URL = 'https://YOUR_SPACE_USERNAME-YOUR_SPACE_NAME.hf.space/gradio_api/call/predict'\n"
            "payload = {\n"
            "  'data': ['Hello!', [], 'You are a helpful assistant.']\n"
            "}\n"
            "res = requests.post(URL, json=payload)\n"
            "print(res.json())"
        )
    )

if __name__ == "__main__":
    # Launch Gradio server
    demo.launch()
