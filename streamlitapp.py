import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import zipfile
import os
import re
import tempfile
from collections import Counter
from groq import Groq
from dateutil import parser as date_parser
import chardet

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="📱",
    layout="wide"
)

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stat-box {
            padding: 20px;
            border-radius: 5px;
            border: 1px solid #ddd;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Helper Functions
# -------------------------------
def clean_member_name(name):
    """Clean and standardize member names."""
    cleaned = name.strip()
    phone_patterns = [
        r'(\+\d{1,3}\s?)?\d{10,}',
        r'\d{3}[-.]?\d{3}[-.]?\d{4}',
        r'\+\d{1,3}\s\d{1,4}\s\d{4,}'
    ]
    
    for pattern in phone_patterns:
        if re.search(pattern, cleaned):
            digits = re.sub(r'\D', '', cleaned)
            return f"User {digits[-4:]}"
    
    cleaned = re.sub(r'[^\w\s\-\']', '', cleaned)
    return cleaned.strip()

def detect_file_encoding(file_path):
    """Detect file encoding."""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'

# -------------------------------
# File Processing Functions
# -------------------------------
def safe_extract_zip(uploaded_file):
    """Safely extract zip file and return chat log path."""
    if uploaded_file is None:
        return None
    
    try:
        # Create temporary directory if it doesn't exist
        if 'temp_dir' not in st.session_state:
            st.session_state.temp_dir = tempfile.mkdtemp()
        
        temp_zip_path = os.path.join(st.session_state.temp_dir, "uploaded.zip")
        extract_path = os.path.join(st.session_state.temp_dir, "extracted")
        
        # Ensure extract directory exists
        os.makedirs(extract_path, exist_ok=True)
        
        # Write and extract zip file
        with open(temp_zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Find chat log file
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.txt'):
                    return os.path.join(root, file)
        
        return None
    
    except Exception as e:
        st.error(f"Error processing zip file: {str(e)}")
        return None

def parse_chat_log(file_path):
    """Parse WhatsApp chat log with enhanced format detection."""
    if not file_path or not os.path.exists(file_path):
        return None
    
    try:
        encoding = detect_file_encoding(file_path)
        with open(file_path, 'r', encoding=encoding) as file:
            lines = file.readlines()
        
        messages_data = []
        user_messages = Counter()
        join_exit_events = []
        global_members = set()
        
        # Message pattern with flexible date format
        message_pattern = re.compile(
            r'^(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\s*-\s*(.*?):\s(.*)$'
        )
        
        current_message = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = message_pattern.match(line)
            if match:
                if current_message:
                    messages_data.append(current_message)
                
                timestamp_str, user, message = match.groups()
                try:
                    timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                    user = clean_member_name(user)
                    current_message = [timestamp_str, user, message]
                    user_messages[user] += 1
                    global_members.add(user)
                except Exception:
                    continue
            elif current_message:
                current_message[2] += f"\n{line}"
        
        if current_message:
            messages_data.append(current_message)
        
        return {
            'total_messages': len(messages_data),
            'user_messages': user_messages,
            'messages_data': messages_data,
            'global_members': sorted(global_members)
        }
    
    except Exception as e:
        st.error(f"Error parsing chat log: {str(e)}")
        return None

# -------------------------------
# Visualization Functions
# -------------------------------
def display_basic_stats(stats):
    """Display basic statistics in a grid."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Messages", stats['total_messages'])
    
    with col2:
        st.metric("Total Members", len(stats['global_members']))
    
    with col3:
        top_user = max(stats['user_messages'].items(), key=lambda x: x[1])
        st.metric("Most Active Member", f"{top_user[0]} ({top_user[1]} messages)")

def display_message_distribution(user_messages):
    """Display message distribution chart."""
    df = pd.DataFrame(list(user_messages.items()), columns=['Member', 'Messages'])
    df = df.sort_values('Messages', ascending=True)
    
    fig = px.bar(
        df,
        x='Messages',
        y='Member',
        orientation='h',
        title='Message Distribution by Member',
        color='Messages',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=max(400, len(user_messages) * 30),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_member_activity(messages_data):
    """Display member activity over time."""
    if not messages_data:
        return
    
    df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member', 'Message'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    
    daily_messages = df.groupby([df['Timestamp'].dt.date, 'Member']).size().reset_index()
    daily_messages.columns = ['Date', 'Member', 'Messages']
    
    fig = px.line(
        daily_messages,
        x='Date',
        y='Messages',
        color='Member',
        title='Daily Message Activity by Member'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Main App
# -------------------------------
def main():
    st.title("📱 WhatsApp Chat Analyzer")
    
    st.write("""
    Upload your WhatsApp chat export (ZIP file) to analyze the conversation patterns.
    To export your chat:
    1. Open WhatsApp chat
    2. Click ⋮ (three dots)
    3. More > Export chat
    4. Choose 'Without Media'
    5. Save the ZIP file
    """)
    
    uploaded_file = st.file_uploader("Upload WhatsApp Chat Export (ZIP)", type="zip")
    
    if uploaded_file:
        with st.spinner("Processing chat log..."):
            chat_log_path = safe_extract_zip(uploaded_file)
            
            if chat_log_path:
                stats = parse_chat_log(chat_log_path)
                
                if stats:
                    st.success("Chat log processed successfully!")
                    
                    # Display statistics
                    display_basic_stats(stats)
                    
                    # Message distribution
                    st.subheader("Message Distribution")
                    display_message_distribution(stats['user_messages'])
                    
                    # Activity timeline
                    st.subheader("Activity Timeline")
                    display_member_activity(stats['messages_data'])
                    
                else:
                    st.error("Failed to parse chat log. Please ensure it's a valid WhatsApp chat export.")
            else:
                st.error("No chat log found in the ZIP file. Please ensure you're uploading a WhatsApp chat export.")

if __name__ == "__main__":
    main()
