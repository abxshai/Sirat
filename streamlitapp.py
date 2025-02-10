import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import zipfile
import os
import re
import tempfile
from collections import Counter
from groq import Groq
from dateutil import parser as date_parser  # For robust date parsing

# -------------------------------
# Helper Function to Clean Member Names
# -------------------------------
def clean_member_name(name):
    """
    If the provided name is primarily a phone number, return a formatted string
    using the last 4 digits. Otherwise, return the cleaned name.
    """
    cleaned = name.strip()
    digits_only = re.sub(r'\D', '', cleaned)
    if len(cleaned) - len(digits_only) <= 2 and len(digits_only) >= 7:
        return "User " + digits_only[-4:]
    else:
        return cleaned

# -------------------------------
# Initialize session state for temporary directory
# -------------------------------
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

# -------------------------------
# Custom CSS for layout and spacing
# -------------------------------
st.markdown("""
    <style>
        .custom-table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        .custom-table th, .custom-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .custom-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .chart-container {
            margin-top: 20px;
            margin-bottom: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Initialize LLM Client
# -------------------------------
API_KEY = st.secrets["API_KEY"]
client = Groq(api_key=API_KEY)

def get_llm_reply(client, prompt, word_placeholder):
    """Get an LLM summary reply using the Groq API."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": ("Analyze the chat log, and summarize key details such as "
                                "the highest message sender, people who joined the group, "
                                "and joining/exiting trends on a weekly or monthly basis, "
                                "mention the inactive members' names with a message count (0 if none), "
                                "and display everything in a tabular format.")
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        response = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            response += delta
            word_placeholder.write(response)
        return response
    
    except Exception as e:
        st.error(f"Error generating LLM reply: {str(e)}")
        return None

# -------------------------------
# Function to safely extract a ZIP file
# -------------------------------
def safe_extract_zip(uploaded_file):
    """
    Safely extract the zip file to a temporary directory and return the path
    of the most relevant chat log (.txt) file (largest file).
    """
    if uploaded_file is None:
        return None
    
    try:
        temp_zip_path = os.path.join(st.session_state.temp_dir, "uploaded.zip")
        with open(temp_zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        extract_path = os.path.join(st.session_state.temp_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Choose the largest .txt file as the chat log
        chat_log_path = None
        max_size = 0
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    if size > max_size:
                        max_size = size
                        chat_log_path = file_path
        return chat_log_path
    
    except Exception as e:
        st.error(f"Error processing zip file: {str(e)}")
        return None

# -------------------------------
# Function to parse the WhatsApp chat log
# -------------------------------
def parse_chat_log(file_path):
    """
    Parse a WhatsApp chat log file (TXT) with robust error handling and return aggregated data.
    Returns a dictionary with:
      - messages_data: list of dicts with keys 'Timestamp', 'Member Name', 'Message'
      - user_messages: Counter of messages per user
      - global_members: Sorted list of all member names
      - join_exit_events: List of system messages (if any)
    """
    if not file_path or not os.path.exists(file_path):
        st.error("Chat log file not found.")
        return None
    
    try:
        encodings = ['utf-8', 'latin-1', 'utf-16', 'ascii']
        file_content = None
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    file_content = f.read()
                break
            except UnicodeDecodeError:
                continue
        if file_content is None:
            st.error("Unable to read the file with any supported encoding.")
            return None
        
        chats = file_content.splitlines()
    except Exception as e:
        st.error(f"Error reading chat log: {str(e)}")
        return None
    
    try:
        total_messages = 0
        user_messages = Counter()
        join_exit_events = []
        messages_data = []
        global_members = set()
        
        # Regex pattern for WhatsApp messages; accepts optional square brackets around timestamp.
        message_pattern = re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-\s*(.*?):\s(.*)$'
        )
        
        # Patterns for system messages (join/exit events, etc.)
        system_patterns = [
            r'(.+) added (.+)',
            r'(.+) left',
            r'(.+) removed (.+)',
            r'(.+) joined using this group\'s invite link',
            r'(.+) changed the subject from "(.+)" to "(.+)"',
            r'(.+) changed this group\'s icon',
            r'Messages and calls are end-to-end encrypted',
            r'(.+) changed the group description',
            r'(.+) changed their phone number',
            r'You deleted this message',
            r'(.+) created group "(.+)"',
            r'(.+) was added by (.+)'
        ]
        system_pattern = '|'.join(system_patterns)
        
        for line in chats:
            line = line.strip()
            if not line:
                continue
            
            message_found = False
            match = re.match(message_pattern, line)
            if match:
                try:
                    timestamp_str, user, message = match.groups()
                    user = clean_member_name(user)
                    
                    try:
                        parsed_date = date_parser.parse(timestamp_str, fuzzy=True)
                    except Exception:
                        parsed_date = None
                    if parsed_date is not None:
                        total_messages += 1
                        user_messages[user] += 1
                        messages_data.append({'Timestamp': parsed_date, 'Member Name': user, 'Message': message})
                        global_members.add(user)
                        message_found = True
                except Exception as e:
                    st.error(f"Error parsing line: {line} - {str(e)}")
                    continue
            
            if not message_found and re.search(system_pattern, line):
                join_exit_events.append(line)
        
        return {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'join_exit_events': join_exit_events,
            'messages_data': messages_data,
            'global_members': sorted(global_members)
        }
    
    except Exception as e:
        st.error(f"Error parsing chat log data: {str(e)}")
        return None

# -------------------------------
# Display Functions for Tables & Charts
# -------------------------------
def display_weekly_messages_table(messages_data, global_members):
    """
    Create Table 1: Weekly Message Breakdown.
    For each week (Monday to Sunday), list every member (from global_members)
    with their message count (or 0 if inactive).
    """
    try:
        if not messages_data:
            st.write("No messages to display")
            return
        
        df = pd.DataFrame(messages_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)
        
        # Compute week start (Monday) for each message
        df['Week Start'] = (df['Timestamp'] - pd.to_timedelta(df['Timestamp'].dt.weekday, unit='D')).dt.normalize()
        
        if df.empty:
            st.write("No valid messages to display")
            return
        
        min_week_start = df['Week Start'].min()
        max_week_start = df['Week Start'].max()
        weeks = pd.date_range(start=min_week_start, end=max_week_start, freq='W-MON')
        
        rows = []
        week_counter = 1
        
        for week_start in weeks:
            week_end = week_start + pd.Timedelta(days=6)
            week_mask = (df['Week Start'] == week_start)
            week_messages = df[week_mask]
            
            for member in sorted(global_members):
                count = week_messages[week_messages['Member Name'] == member].shape[0] if not week_messages.empty else 0
                rows.append({
                    'Week': f"Week {week_counter}",
                    'Week Duration': f"{week_start.strftime('%d %b %Y')} - {week_end.strftime('%d %b %Y')}",
                    'Member Name': member,
                    'Number of Messages Sent': count
                })
            week_counter += 1
        
        weekly_df = pd.DataFrame(rows)
        st.markdown("### Table 1: Weekly Message Breakdown")
        st.dataframe(weekly_df)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        user_totals = weekly_df.groupby('Member Name')['Number of Messages Sent'].sum().reset_index()
        if not user_totals.empty:
            ax.bar(user_totals['Member Name'], user_totals['Number of Messages Sent'], color='skyblue')
            ax.set_xlabel("Member Name")
            ax.set_ylabel("Total Messages")
            ax.set_title("Total Messages Sent by Each User")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        else:
            st.warning("No message data available to plot.")
    
    except Exception as e:
        st.error(f"Error creating weekly message table: {str(e)}")

def display_member_statistics(messages_data):
    """
    Create Table 2: Member Statistics.
    For each member, show:
      - Unique Member Name
      - Group Activity Status (Active if last message within 30 days)
      - Longest Membership Duration (Weeks)
      - Avg. Weekly Messages
    """
    try:
        if not messages_data:
            st.write("No messages to display")
            return
        
        df = pd.DataFrame(messages_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)
        
        if df.empty:
            st.write("No valid messages to display")
            return
        
        grouped = df.groupby('Member Name').agg(
            first_message=('Timestamp', 'min'),
            last_message=('Timestamp', 'max'),
            total_messages=('Message', 'count')
        ).reset_index()
        
        grouped['Longest Membership Duration (Weeks)'] = ((grouped['last_message'] - grouped['first_message']).dt.days / 7).round().astype(int)
        grouped['Avg. Weekly Messages'] = grouped.apply(
            lambda row: round(row['total_messages'] / max(row['Longest Membership Duration (Weeks)'], 1), 2),
            axis=1
        )
        
        overall_last_date = df['Timestamp'].max()
        grouped['Group Activity Status'] = grouped['last_message'].apply(
            lambda x: 'Active' if (overall_last_date - x).days <= 30 else 'Inactive'
        )
        
        grouped.rename(columns={'Member Name': 'Unique Member Name'}, inplace=True)
        table2 = grouped[['Unique Member Name', 'Group Activity Status', 'Longest Membership Duration (Weeks)', 'Avg. Weekly Messages']]
        
        st.markdown("### Table 2: Member Statistics")
        st.dataframe(table2)
    
    except Exception as e:
        st.error(f"Error creating member statistics: {str(e)}")

def display_total_messages_chart(user_messages):
    """
    Display a bar chart of total messages per user using Plotly Express.
    """
    try:
        if not user_messages:
            st.write("No message data to display")
            return
        
        df = pd.DataFrame(user_messages.items(), columns=['Member Name', 'Messages'])
        if df.empty:
            st.write("No message data to display")
            return
        
        fig = px.bar(
            df, 
            x='Member Name', 
            y='Messages', 
            title='Total Messages Sent by Each User',
            color='Messages'
        )
        st.markdown("<div class='chart-container'></div>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating messages chart: {str(e)}")

# -------------------------------
# Main App Layout
# -------------------------------
st.title("Structured Chat Log Analyzer")
uploaded_file = st.file_uploader("Upload a zip file containing the WhatsApp chat log", type="zip")

if uploaded_file:
    chat_log_path = safe_extract_zip(uploaded_file)
    if chat_log_path:
        stats = parse_chat_log(chat_log_path)
        if stats:
            st.success("Chat log parsed successfully!")
            
            # Display Table 1: Weekly Message Breakdown
            display_weekly_messages_table(stats['messages_data'], stats['global_members'])
            
            # Display Table 2: Member Statistics
            display_member_statistics(stats['messages_data'])
            
            # Display a bar chart for total messages per user
            display_total_messages_chart(stats['user_messages'])
            
            # LLM-based summary component (using aggregated data)
            st.markdown("### LLM Summary of Chat Log")
            if st.button("Generate Summary"):
                with st.spinner("Analyzing chat log..."):
                    # Create a simple aggregation for summary; you can adjust as needed.
                    top_users = {d['Member Name']: 0 for d in stats['messages_data']}\n  
