import os
import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Load local environment variables if present
load_dotenv()

# Import core modules
from assistants.oss_assistant import OSSAssistant
from assistants.frontier_assistant import FrontierAssistant
from evaluation.evaluator import AssistantEvaluator

# Page Configuration
st.set_page_config(
    page_title="Dual AI Assistant Arena & Evaluation",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Glassmorphism & Sleek Accent Highlights)
st.markdown("""
<style>
    /* Main Layout */
    .stApp {
        background: linear-gradient(135deg, #0d0f19 0%, #171b30 50%, #0a0b12 100%);
        color: #e2e8f0;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Elegant Title Gradient */
    .gradient-text {
        background: linear-gradient(90deg, #60a5fa 0%, #a855f7 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -0.05em;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Glassmorphic Container Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(168, 85, 247, 0.3);
    }
    
    /* Chat Bubble Styling */
    .chat-bubble {
        padding: 1rem 1.25rem;
        border-radius: 14px;
        margin-bottom: 0.75rem;
        max-width: 85%;
        line-height: 1.5;
        font-size: 0.95rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .chat-user {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 2px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .chat-assistant {
        background: rgba(30, 41, 59, 0.8);
        color: #f1f5f9;
        margin-right: auto;
        border-bottom-left-radius: 2px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Alert & Indicator Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-guardrail {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-tool {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-metric {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    /* Sidebar Overrides */
    .css-1d391kg {
        background-color: #0b0c16 !important;
    }
    
    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.3);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(148, 163, 184, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("GEMINI_API_KEY", "")
if "hf_api_token" not in st.session_state:
    st.session_state.hf_api_token = os.environ.get("HF_API_TOKEN", "")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are a helpful, secure, and professional AI assistant. "
        " Abide strictly by security guidelines, protect PII, and use mathematical/telemetry tools when required."
    )
if "memory_window" not in st.session_state:
    st.session_state.memory_window = 5
if "enable_tools" not in st.session_state:
    st.session_state.enable_tools = True
if "enable_guardrails" not in st.session_state:
    st.session_state.enable_guardrails = True

# Sync state with Environment dynamically
os.environ["GEMINI_API_KEY"] = st.session_state.openai_api_key
os.environ["HF_API_TOKEN"] = st.session_state.hf_api_token

# Initialize Assistants
@st.cache_resource
def get_assistants(sys_prompt):
    oss = OSSAssistant(system_prompt=sys_prompt)
    frontier = FrontierAssistant(system_prompt=sys_prompt)
    return oss, frontier

oss_assistant, frontier_assistant = get_assistants(st.session_state.system_prompt)

# Set Memory Windows dynamically
oss_assistant.memory.window_size = st.session_state.memory_window
frontier_assistant.memory.window_size = st.session_state.memory_window

# Header Section
st.markdown("<h1 class='gradient-text'>⚖️ AI Assistant Arena & Evaluation</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Compare Open-Source (Qwen 2.5-0.5B-Instruct) vs Frontier (Gemini 1.5 Flash) side-by-side with full guardrails, memory & tool execution telemetry.</p>", unsafe_allow_html=True)

# Sidebar Configuration Control Center
with st.sidebar:
    st.markdown("### 🎛️ Control Center")
    
    # Credentials Accordion
    with st.expander("🔑 API Keys & Auth", expanded=True):
        api_input = st.text_input(
            "Gemini API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Get from Google AI Studio. Required for live Frontier Assistant."
        )
        if api_input != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_input
            os.environ["GEMINI_API_KEY"] = api_input
            st.cache_resource.clear()
            st.rerun()
            
        hf_input = st.text_input(
            "Hugging Face Token",
            value=st.session_state.hf_api_token,
            type="password",
            help="Get from Hugging Face Settings. Recommended for fast serverless OSS queries."
        )
        if hf_input != st.session_state.hf_api_token:
            st.session_state.hf_api_token = hf_input
            os.environ["HF_API_TOKEN"] = hf_input
            st.cache_resource.clear()
            st.rerun()

    # Dynamic Simulation indicators
    if not st.session_state.openai_api_key:
        st.warning("⚠️ Gemini Key missing. Frontier model running in high-fidelity simulation fallback mode.")
    if not st.session_state.hf_api_token:
        st.warning("⚠️ HF Token missing. Qwen model running in high-fidelity simulation fallback mode.")

    # System Parameters
    st.markdown("### ⚙️ Parameters")
    
    new_sys_prompt = st.text_area(
        "System Prompt / Instructions",
        value=st.session_state.system_prompt,
        height=120
    )
    if new_sys_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = new_sys_prompt
        st.cache_resource.clear()
        st.rerun()
        
    st.session_state.memory_window = st.slider("Memory Depth (turns)", 1, 10, 5)
    
    # Feature Toggles (Visual updates only in simulation/config)
    st.session_state.enable_tools = st.toggle("Enable Tool Access", value=True)
    st.session_state.enable_guardrails = st.toggle("Enable Guardrails", value=True)
    
    st.markdown("---")
    if st.button("🗑️ Reset Chats & Memory", use_container_width=True):
        st.session_state.chat_history = []
        oss_assistant.memory.clear()
        frontier_assistant.memory.clear()
        st.success("Conversation memory cleared successfully!")
        st.rerun()

# Main Application Tabs
tab_arena, tab_eval, tab_obs = st.tabs(["⚔️ Chat Arena", "📊 Benchmark & Evaluation", "🔍 Observability Logs"])

# Tab 1: The Chat Arena (Side-by-Side Dual-Assistant Chat)
with tab_arena:
    st.markdown("<div class='glass-card'><h4>⚔️ Dual Chat playground</h4>Submit a query below to test both models simultaneously. Pay attention to how they invoke tools and trigger guardrail warnings!</div>", unsafe_allow_html=True)
    
    # Split Layout for Side-by-side chats
    col_oss, col_front = st.columns(2)
    
    # 1. Render OSS Chat History
    with col_oss:
        st.markdown("### 🤖 Open-Source (Qwen 2.5-0.5B)")
        st.caption("Local / Serverless HF Inference API")
        
        st.markdown("<div style='height: 400px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; background: rgba(15, 23, 42, 0.2);'>", unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            st.markdown(f"<div class='chat-bubble chat-user'>👤 {chat['prompt']}</div>", unsafe_allow_html=True)
            
            # Badges for triggers
            badges = ""
            if chat['oss_guardrails']:
                for g in chat['oss_guardrails']:
                    badges += f"<span class='badge badge-guardrail'>🛡️ {g}</span>"
            if chat['oss_tools']:
                for t in chat['oss_tools']:
                    badges += f"<span class='badge badge-tool'>🛠️ Tool: {t}</span>"
            badges += f"<span class='badge badge-metric'>⏱️ {chat['oss_latency']:.2f}s</span>"
            
            st.markdown(f"<div class='chat-bubble chat-assistant'>🤖 {chat['oss_response']}<br/><br/>{badges}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # 2. Render Frontier Chat History
    with col_front:
        st.markdown("### 🌌 Frontier Assistant (Gemini 1.5)")
        st.caption("Hosted Foundation Model API")
        
        st.markdown("<div style='height: 400px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; background: rgba(15, 23, 42, 0.2);'>", unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            st.markdown(f"<div class='chat-bubble chat-user'>👤 {chat['prompt']}</div>", unsafe_allow_html=True)
            
            # Badges for triggers
            badges = ""
            if chat['front_guardrails']:
                for g in chat['front_guardrails']:
                    badges += f"<span class='badge badge-guardrail'>🛡️ {g}</span>"
            if chat['front_tools']:
                for t in chat['front_tools']:
                    badges += f"<span class='badge badge-tool'>🛠️ Tool: {t}</span>"
            badges += f"<span class='badge badge-metric'>⏱️ {chat['front_latency']:.2f}s</span>"
            
            st.markdown(f"<div class='chat-bubble chat-assistant'>🌌 {chat['front_response']}<br/><br/>{badges}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Input Box centered below
    st.markdown("<br/>", unsafe_allow_html=True)
    with st.form("chat_input_form", clear_on_submit=True):
        user_input = st.text_input("Send a message to both assistants...", placeholder="Type 'calculate 256 * 14' or 'what is the weather like?' to test tools...")
        submit_btn = st.form_submit_button("🚀 Submit to Arena", use_container_width=True)
        
        if submit_btn and user_input.strip():
            with st.spinner("Processing dual models responses..."):
                # Run OSS turn
                # Disable tools/guardrails in base if checked off (handled programmatically here)
                if not st.session_state.enable_guardrails:
                    # Clear filters momentarily
                    orig_in = oss_assistant.guardrails.process_input
                    orig_out = oss_assistant.guardrails.process_output
                    oss_assistant.guardrails.process_input = lambda x: (x, [])
                    oss_assistant.guardrails.process_output = lambda x: (x, [])
                    
                oss_res = oss_assistant.respond(user_input)
                
                if not st.session_state.enable_guardrails:
                    oss_assistant.guardrails.process_input = orig_in
                    oss_assistant.guardrails.process_output = orig_out
                    
                # Run Frontier turn
                if not st.session_state.enable_guardrails:
                    orig_in_f = frontier_assistant.guardrails.process_input
                    orig_out_f = frontier_assistant.guardrails.process_output
                    frontier_assistant.guardrails.process_input = lambda x: (x, [])
                    frontier_assistant.guardrails.process_output = lambda x: (x, [])
                    
                front_res = frontier_assistant.respond(user_input)
                
                if not st.session_state.enable_guardrails:
                    frontier_assistant.guardrails.process_input = orig_in_f
                    frontier_assistant.guardrails.process_output = orig_out_f
                
                # Append to st history
                st.session_state.chat_history.append({
                    "prompt": user_input,
                    "oss_response": oss_res["response"],
                    "oss_latency": oss_res["latency"],
                    "oss_guardrails": oss_res["guardrails_triggered"],
                    "oss_tools": oss_res["tools_used"],
                    "front_response": front_res["response"],
                    "front_latency": front_res["latency"],
                    "front_guardrails": front_res["guardrails_triggered"],
                    "front_tools": front_res["tools_used"]
                })
                
                st.rerun()

# Tab 2: The Benchmark and Evaluation Screen
with tab_eval:
    st.markdown("<div class='glass-card'><h4>📊 Automated Evaluation Suite</h4>Evaluate both assistants against a benchmark dataset of 15 prompts across Factual accuracy, Adversarial attack safety, and Bias Neutrality. Click the button to trigger a live run.</div>", unsafe_allow_html=True)
    
    col_btn, col_stats = st.columns([1, 3])
    
    # 1. Trigger benchmark run
    run_benchmark = False
    with col_btn:
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("🎯 Run Evaluation Benchmark", use_container_width=True, type="primary"):
            run_benchmark = True

    if run_benchmark:
        with st.spinner("Executing benchmark dataset (15 prompts) on both assistants. This compiles latencies, accuracies, and safety ratios..."):
            evaluator = AssistantEvaluator()
            
            # Execute both
            oss_results, oss_summary = evaluator.run_benchmark(oss_assistant)
            front_results, front_summary = evaluator.run_benchmark(frontier_assistant)
            
            # Store results in session state
            st.session_state.evaluation_results = {
                "oss": {"summary": oss_summary, "details": oss_results},
                "frontier": {"summary": front_summary, "details": front_results}
            }
            st.success("Evaluation Benchmark Completed successfully!")

    # Display results if available
    if "evaluation_results" in st.session_state:
        eval_data = st.session_state.evaluation_results
        oss_sum = eval_data["oss"]["summary"]
        front_sum = eval_data["frontier"]["summary"]
        
        # Display side-by-side metric tiles
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Avg Latency (OSS vs Frontier)", f"{oss_sum['avg_latency']}s", f"vs {front_sum['avg_latency']}s", delta_color="inverse")
        with col2:
            st.metric("Hallucination Rate (Lower is better)", f"{oss_sum['hallucination_rate']}%", f"vs {front_sum['hallucination_rate']}%", delta_color="inverse")
        with col3:
            st.metric("Content Safety Refusal", f"{oss_sum['content_safety_score']}%", f"vs {front_sum['content_safety_score']}%")
        with col4:
            st.metric("Bias Index (Lower is better)", f"{oss_sum['bias_index']}%", f"vs {front_sum['bias_index']}%", delta_color="inverse")
        with col5:
            st.metric("Session Token Count", f"{oss_sum['total_input_tokens'] + oss_sum['total_output_tokens']}", f"vs {front_sum['total_input_tokens'] + front_sum['total_output_tokens']}")
            
        # Draw Plot
        st.markdown("### 📊 Metric Performance Chart")
        
        categories = ['Hallucination Rate (↓)', 'Bias Index (↓)', 'Content Safety (↑)']
        oss_scores = [oss_sum['hallucination_rate'], oss_sum['bias_index'], oss_sum['content_safety_score']]
        front_scores = [front_sum['hallucination_rate'], front_sum['bias_index'], front_sum['content_safety_score']]
        
        x = range(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#171b30')
        ax.set_facecolor('#1e293b')
        
        rects1 = ax.bar([i - width/2 for i in x], oss_scores, width, label='Qwen 2.5 (OSS)', color='#f43f5e')
        rects2 = ax.bar([i + width/2 for i in x], front_scores, width, label='Gemini (Frontier)', color='#3b82f6')
        
        ax.set_ylabel('Score Percentage (%)', color='#e2e8f0')
        ax.set_title('Assistant Metric Comparison', color='#e2e8f0')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, color='#e2e8f0')
        ax.tick_params(axis='y', colors='#e2e8f0')
        ax.legend(facecolor='#0f172a', edgecolor='none', labelcolor='#e2e8f0')
        ax.spines['bottom'].set_color('#475569')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#475569')
        
        # Add labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', color='#e2e8f0', fontsize=8)
                            
        autolabel(rects1)
        autolabel(rects2)
        
        st.pyplot(fig)
        
        # Show Detailed Tables
        st.markdown("### 📋 Detailed Test Case Log")
        
        details_df = []
        for o_c, f_c in zip(eval_data["oss"]["details"], eval_data["frontier"]["details"]):
            details_df.append({
                "ID": o_c["id"],
                "Category": o_c["category"],
                "Prompt": o_c["prompt"],
                "OSS Score": f"{o_c['score'] * 100}%",
                "OSS Latency": f"{o_c['latency']:.2f}s",
                "Frontier Score": f"{f_c['score'] * 100}%",
                "Frontier Latency": f"{f_c['latency']:.2f}s"
            })
            
        st.dataframe(pd.DataFrame(details_df), use_container_width=True)
    else:
        st.info("💡 Run the evaluation benchmark using the button above to view detailed comparison metrics.")

# Tab 3: Observability Logs Viewer (Querying the SQLite database)
with tab_obs:
    st.markdown("<div class='glass-card'><h4>🔍 Observability Telemetry Logs</h4>All conversations, guardrails, tool calls, and model outputs are logged locally into an SQLite database (`database.db`). Below is a live feed from this database.</div>", unsafe_allow_html=True)
    
    # Query database
    db_path = os.environ.get("DATABASE_PATH", "database.db")
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            logs_df = pd.read_sql_query("SELECT id, timestamp, assistant_name, raw_prompt, response, latency, input_tokens, output_tokens, tools_used, guardrails_triggered FROM logs ORDER BY id DESC LIMIT 50", conn)
            conn.close()
            
            if not logs_df.empty:
                st.dataframe(logs_df, use_container_width=True)
            else:
                st.info("No logs present in the database yet. Submit a message in the Chat Arena to populate!")
        except Exception as e:
            st.error(f"Error querying SQLite database: {e}")
    else:
        st.info("SQLite database file not found. It will be initialized once the first chat is submitted.")
