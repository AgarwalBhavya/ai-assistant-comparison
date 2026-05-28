# 🚀 Public Deployment Guide: Qwen 2.5-0.5B-Instruct on Hugging Face Spaces

This guide walks you through deploying the Open-Source Qwen 2.5-0.5B-Instruct model as a free, publicly accessible Gradio API endpoint and chat application on Hugging Face Spaces.

## Quick-Deploy Steps (Under 2 Minutes)

1. **Sign In or Register**:
   - Go to [Hugging Face](https://huggingface.co/) and log in (or create a free account).

2. **Create a New Space**:
   - Navigate to [huggingface.co/new-space](https://huggingface.co/new-space).
   - **Space Name**: e.g., `qwen-2.5-assistant`
   - **SDK**: Select **Gradio**.
   - **Space Hardware**: Choose the **free CPU basic** tier (this is more than enough because the model is loaded serverless!).
   - **Visibility**: Set to **Public** (or Private if you wish, but Public is recommended for the demo link).

3. **Upload Files**:
   - Once the space is created, go to the **Files and versions** tab.
   - Click **Add file** -> **Create a new file** or **Upload files**.
   - Add the following files from your local `deployment/` folder:
     - `app.py`
     - `requirements.txt`
   - Commit the changes directly to the `main` branch.

4. **Add Hugging Face Token Secret (Optional, recommended for speed)**:
   - Go to your Space's **Settings** tab.
   - Under **Variables and secrets**, add a new **Secret**.
   - **Name**: `HF_API_TOKEN`
   - **Value**: Your Hugging Face User Access Token (Get one from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
   - This ensures Hugging Face serverless execution doesn't run into generic rate limits!

5. **Build and Run**:
   - Hugging Face will automatically build and start containerizing the Gradio app!
   - Within 30 seconds, your public chat interface and API will be **Operational** 🚀.

## How to Test the Deployed API

Once the Space is running:
- You can copy the **Public URL** of your Space (e.g. `https://huggingface.co/spaces/your-username/qwen-2.5-assistant`).
- You can plug this URL directly into the client Streamlit application's settings or `.env` to route your OSS queries directly to your publicly hosted instance!
- This gives you a fully functional public deployment bonus for the interview process.
