import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
import os
import re
import tempfile
from collections import Counter
from groq import Groq

# -------------------------------
# Initialize session state
# -------------------------------
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

# -------------------------------
# Function to safely extract and read files
# -------------------------------
def safe_extract_zip(uploaded_file):
    """Safely extract the zip file to a temporary directory."""
    if uploaded_file is None:
        return None
    
    try:
        # Create a temporary file to store the uploaded zip
        temp_zip_path = os.path.join(st.session_state.temp_dir, "uploaded.zip")
        
        # Write the uploaded file to disk
        with open(temp_zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract the zip file
        extract_path = os.path.join(st.session_state.temp_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Find the chat log file
        chat_log_path = None
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.txt'):
                    chat_log_path = os.path.join(root, file)
                    break
        
        return chat_log_path
    
    except Exception as e:
        st.error(f"Error processing zip file: {str(e)}")
        return None

# -------------------------------
# Function to parse the chat log with error handling
# -------------------------------
def parse_chat_log(file_path):
    """Parse the chat log file with robust error handling."""
    if not file_path or not os.path.exists(file_path):
        st.error("Chat log file not found.")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            chats = file.readlines()
    except UnicodeDecodeError:
        # Try alternative encodings if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                chats = file.readlines()
        except Exception as e:
            st.error(f"Error reading file with alternative encoding: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error reading chat log: {str(e)}")
        return None
    
    try:
        total_messages = 0
        user_messages = Counter()
        join_exit_events = []
        messages_data = []
        global_members = set()
        
        message_pattern = re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}) - (.*?): (.*)')
        join_exit_pattern = re.compile(r'(.*) added (.*)|(.+) left')
        
        for line in chats:
            line = line.strip()
            if not line:
                continue
            
            match = message_pattern.match(line)
            if match:
                timestamp, user, message = match.groups()
                # Validate timestamp format
                try:
                    pd.to_datetime(timestamp, format='%d/%m/%y, %H:%M')
                    total_messages += 1
                    user_messages[user] += 1
                    messages_data.append([timestamp, user, message])
                    global_members.add(user)
                except ValueError:
                    continue  # Skip invalid timestamps
            
            event_match = join_exit_pattern.match(line)
            if event_match:
                join_exit_events.append(line)
        
        if not messages_data:
            st.warning("No valid messages found in the chat log.")
            return None
        
        return {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'join_exit_events': join_exit_events,
            'messages_data': messages_data,
            'global_members': sorted(global_members)
        }
    
    except Exception as e:
        st.error(f"Error parsing chat content: {str(e)}")
        return None

# -------------------------------
# Main Streamlit App Layout
# -------------------------------
def main():
    st.title("Structured Chat Log Analyzer")
    
    # Initialize API client if API key exists
    if "API_KEY" in st.secrets:
        client = Groq(api_key=st.secrets["API_KEY"])
    else:
        st.warning("API key not found. LLM summary feature will be disabled.")
        client = None
    
    uploaded_file = st.file_uploader("Upload a zip file containing the chat log", type="zip")
    
    if uploaded_file:
        with st.spinner("Processing chat log..."):
            # Safely extract and process the zip file
            chat_log_path = safe_extract_zip(uploaded_file)
            
            if chat_log_path:
                # Parse the chat log
                stats = parse_chat_log(chat_log_path)
                
                if stats:
                    st.success('Chat log parsed successfully!')
                    
                    # Display statistics and visualizations
                    display_weekly_messages_table(stats['messages_data'], stats['global_members'])
                    display_member_statistics(stats['messages_data'])
                    display_total_messages_chart(stats['user_messages'])
                    
                    # LLM Summary section
                    if client:
                        st.markdown("### LLM Summary of Chat Log")
                        if st.button("Generate Summary"):
                            with st.spinner("Analyzing chat log..."):
                                top_users = dict(stats['user_messages'].most_common(5))
                                snippet_events = stats['join_exit_events'][:20]
                                prompt = (f"Summarize the chat log with these key points:\n"
                                        f"- Top message senders: {top_users}\n"
                                        f"- Group join/exit events (sample): {snippet_events}\n")
                                word_placeholder = st.empty()
                                get_llm_reply(client, prompt, word_placeholder)
                else:
                    st.error("Failed to parse chat log. Please ensure the file format is correct.")
            else:
                st.error("Failed to process the uploaded file. Please try again.")

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    main()
