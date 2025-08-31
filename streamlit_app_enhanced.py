#!/usr/bin/env python3
"""
Enhanced Streamlit Interface for EV Charging AI Pipeline
Can actually call deployed fine-tuned models via inference server
"""

import streamlit as st
import os
import sys
import requests
import json
from pathlib import Path
from datetime import datetime
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from deployment.model_registry import ModelRegistry

# Configuration
INFERENCE_SERVER_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"  # Change this to your actual API key

def setup_page():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="EV Charging AI Assistant",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚡ EV Charging AI Assistant")
    st.markdown("Ask questions about electric vehicle charging infrastructure and get expert answers from your fine-tuned model!")

def check_inference_server():
    """Check if inference server is running"""
    try:
        response = requests.get(f"{INFERENCE_SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except requests.exceptions.RequestException:
        return False, None

def call_inference_api(question, model_id=None, max_tokens=500, temperature=0.7):
    """Call the inference API to get model response"""
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "question": question,
            "model_id": model_id,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{INFERENCE_SERVER_URL}/predict",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection Error: {str(e)}"}

def get_available_models_api():
    """Get available models from inference server"""
    try:
        response = requests.get(f"{INFERENCE_SERVER_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except requests.exceptions.RequestException:
        return []

def create_chat_interface():
    """Create the main chat interface"""
    st.header("💬 Chat with EV Charging Expert")
    
    # Check server status
    server_running, health_data = check_inference_server()
    
    if not server_running:
        st.error("❌ Inference server is not running!")
        st.info("To start the server, run: `python -m src.deployment.inference_server`")
        st.info("Or use the deployment script: `python deploy.py`")
        return
    
    st.success(f"✅ Connected to inference server (Uptime: {health_data.get('uptime', 0):.1f}s)")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                st.caption(f"Model: {message['metadata'].get('model_id', 'Unknown')} | "
                          f"Response time: {message['metadata'].get('response_time', 0):.2f}s")
    
    # Chat input
    if prompt := st.chat_input("Ask about EV charging..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get sidebar parameters
                max_tokens = st.session_state.get('max_tokens', 500)
                temperature = st.session_state.get('temperature', 0.7)
                selected_model = st.session_state.get('selected_model', None)
                
                # Call inference API
                response = call_inference_api(
                    prompt, 
                    model_id=selected_model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                    ai_response = f"Sorry, I encountered an error: {response['error']}"
                else:
                    ai_response = response.get('answer', 'No response received')
                    
                    # Show response metadata
                    st.caption(f"Model: {response.get('model_id', 'Unknown')} | "
                              f"Response time: {response.get('response_time', 0):.2f}s | "
                              f"Tokens: {response.get('tokens_used', 0)}")
                
                st.markdown(ai_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": ai_response,
            "metadata": {
                "model_id": response.get('model_id', 'Unknown'),
                "response_time": response.get('response_time', 0),
                "tokens_used": response.get('tokens_used', 0)
            }
        })

def show_model_info():
    """Show information about available models"""
    st.header("📋 Model Information")
    
    # Check server status
    if not check_inference_server()[0]:
        st.error("❌ Inference server not running")
        return
    
    # Get models from API
    models = get_available_models_api()
    
    if not models:
        st.info("No models found. Make sure you have deployed models and they are marked as 'deployed' status.")
        return
    
    # Display model information
    for i, model in enumerate(models):
        with st.expander(f"Model {i+1}: {model['model_id']}", expanded=i==0):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Base Model:** {model['base_model']}")
                st.write(f"**Version:** {model['version']}")
                st.write(f"**Status:** {model['status']}")
                st.write(f"**Created:** {model['created_at']}")
            
            with col2:
                st.write(f"**Description:** {model['description']}")
                
                # Show performance metrics if available
                if model['performance_metrics']:
                    st.subheader("Performance Metrics")
                    for metric, value in model['performance_metrics'].items():
                        st.write(f"**{metric}:** {value}")
            
            # Model selection button
            if st.button(f"Select Model {i+1}", key=f"select_{i}"):
                st.session_state.selected_model = model['model_id']
                st.success(f"Selected model: {model['model_id']}")

def show_server_metrics():
    """Show inference server metrics"""
    st.header("📊 Server Metrics")
    
    # Check server status
    server_running, health_data = check_inference_server()
    
    if not server_running:
        st.error("❌ Inference server not running")
        return
    
    # Get metrics
    try:
        response = requests.get(f"{INFERENCE_SERVER_URL}/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Requests", metrics.get('request_count', 0))
            
            with col2:
                st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1f}%")
            
            with col3:
                st.metric("Avg Response Time", f"{metrics.get('average_response_time', 0):.2f}s")
            
            with col4:
                st.metric("Models Available", metrics.get('models_available', 0))
            
            # Detailed metrics
            st.subheader("Detailed Metrics")
            st.json(metrics)
            
        else:
            st.error(f"Could not fetch metrics: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching metrics: {e}")

def show_quick_actions():
    """Show quick action buttons"""
    st.header("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Run Pipeline", help="Execute the complete AI pipeline"):
            st.info("Pipeline execution started! Check the terminal for progress.")
            st.code("python test_complete_pipeline.py")
    
    with col2:
        if st.button("🚀 Deploy Model", help="Deploy the pipeline components"):
            st.info("Deployment started! Check the terminal for progress.")
            st.code("python deploy.py")
    
    with col3:
        if st.button("📊 View Logs", help="View pipeline logs"):
            log_dir = Path("logs")
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                if log_files:
                    st.success(f"Found {len(log_files)} log files")
                    for log_file in log_files:
                        st.write(f"- {log_file.name}")
                else:
                    st.info("No log files found")
            else:
                st.info("Logs directory not found")
    
    # Test connection button
    st.subheader("🔗 Connection Test")
    if st.button("Test Inference Server Connection"):
        server_running, health_data = check_inference_server()
        if server_running:
            st.success("✅ Inference server is running and accessible!")
            st.json(health_data)
        else:
            st.error("❌ Cannot connect to inference server")
            st.info("Make sure the server is running on the correct port")

def show_sidebar():
    """Show sidebar with navigation and settings"""
    st.sidebar.title("🔧 Settings")
    
    # Server status
    st.sidebar.subheader("Server Status")
    server_running, health_data = check_inference_server()
    if server_running:
        st.sidebar.success("🟢 Server Running")
        st.sidebar.info(f"Uptime: {health_data.get('uptime', 0):.1f}s")
    else:
        st.sidebar.error("🔴 Server Offline")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    if server_running:
        models = get_available_models_api()
        if models:
            model_options = [f"{m['model_id']} ({m['status']})" for m in models]
            selected_model = st.sidebar.selectbox(
                "Choose Model:",
                model_options,
                index=0,
                key="model_selector"
            )
            
            # Extract model ID from selection
            if selected_model:
                model_id = selected_model.split(" (")[0]
                st.session_state.selected_model = model_id
                st.sidebar.info(f"Selected: {model_id}")
        else:
            st.sidebar.warning("No models available")
    else:
        st.sidebar.warning("Server offline")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    max_tokens = st.sidebar.slider("Max Tokens", 100, 1000, 500, key="max_tokens_slider")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="temp_slider")
    
    # Store in session state
    st.session_state.max_tokens = max_tokens
    st.session_state.temperature = temperature
    
    st.sidebar.info(f"Max Tokens: {max_tokens}")
    st.sidebar.info(f"Temperature: {temperature}")
    
    # API configuration
    st.sidebar.subheader("API Configuration")
    global API_KEY
    api_key = st.sidebar.text_input(
        "API Key",
        value=API_KEY,
        type="password",
        help="API key for authentication"
    )
    
    if api_key != API_KEY:
        API_KEY = api_key
        st.sidebar.success("✅ API key updated")
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Server info
    st.sidebar.subheader("Server Info")
    st.sidebar.code(f"URL: {INFERENCE_SERVER_URL}")
    if server_running and health_data:
        st.sidebar.info(f"Models: {health_data.get('models_available', 0)}")

def main():
    """Main application function"""
    setup_page()
    
    # Show sidebar
    show_sidebar()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📋 Models", "📊 Metrics", "⚙️ Status", "🚀 Actions"])
    
    with tab1:
        create_chat_interface()
    
    with tab2:
        show_model_info()
    
    with tab3:
        show_server_metrics()
    
    with tab4:
        show_pipeline_status()
    
    with tab5:
        show_quick_actions()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**EV Charging AI Pipeline** | Built with Streamlit | "
        "[View Source](https://github.com/your-repo/ev-charging-ai)"
    )

def show_pipeline_status():
    """Show pipeline execution status"""
    st.header("⚙️ Pipeline Status")
    
    # Check if training data exists
    training_file = Path("data/processed/training_data.jsonl")
    if training_file.exists():
        st.success("✅ Training data available")
        
        # Show file info
        file_size = training_file.stat().st_size / (1024 * 1024)  # MB
        st.info(f"Training file: {training_file.name} ({file_size:.2f} MB)")
        
        # Count training examples
        try:
            with open(training_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                st.info(f"Training examples: {len(lines)}")
        except Exception as e:
            st.warning(f"Could not read training file: {e}")
    else:
        st.warning("⚠️ Training data not found. Run the pipeline first.")
    
    # Check if benchmarks exist
    benchmark_dir = Path("data/benchmark")
    if benchmark_dir.exists() and any(benchmark_dir.glob("*.json")):
        st.success("✅ Benchmarks available")
    else:
        st.info("ℹ️ Benchmarks not found. Run evaluation to create them.")
    
    # Check if evaluation results exist
    eval_dir = Path("data/evaluation")
    if eval_dir.exists() and any(eval_dir.glob("*.json")):
        st.success("✅ Evaluation results available")
    else:
        st.info("ℹ️ Evaluation results not found. Run evaluation to create them.")

if __name__ == "__main__":
    main()
