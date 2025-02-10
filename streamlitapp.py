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
                    "content": "Analyze the chat log, and summarize key details such as the highest message sender, people who joined the group, and joining/exiting trends on a weekly or monthly basis."
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
# Function to safely extract and read files
# -------------------------------
def safe_extract_zip(uploaded_file):
    """Safely extract the zip file to a temporary directory and return the path of the chat log (.txt) file."""
    if uploaded_file is None:
        return None
    
    try:
        # Create a temporary file for the uploaded zip
        temp_zip_path = os.path.join(st.session_state.temp_dir, "uploaded.zip")
        
        # Write the uploaded file to disk
        with open(temp_zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract the zip file to a subdirectory
        extract_path = os.path.join(st.session_state.temp_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Find the chat log file (.txt)
        chat_log_path = None
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.txt'):
                    chat_log_path = os.path.join(root, file)
                    break
            if chat_log_path:
                break
        
        return chat_log_path
    
    except Exception as e:
        st.error(f"Error processing zip file: {str(e)}")
        return None

# -------------------------------
# Function to parse the chat log with robust error handling
# -------------------------------
def parse_chat_log(file_path):
    """Parse any WhatsApp chat log file with robust error handling and return aggregated data."""
    if not file_path or not os.path.exists(file_path):
        st.error("Chat log file not found.")
        return None
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'utf-16', 'ascii']
        file_content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    file_content = file.read()
                break
            except UnicodeDecodeError:
                continue
        
        if file_content is None:
            st.error("Unable to read the file with any supported encoding.")
            return None
        
        # Split into lines
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
        
        # Common WhatsApp date patterns (for extraction)
        date_patterns = [
            r'\[?(\d{1,2}/\d{1,2}/\d{2,4},?\s\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?)\]?',
        ]
        date_pattern = '|'.join(date_patterns)
        
        # Message patterns for different WhatsApp formats
        message_patterns = [
            # Standard format: [timestamp] - name: message
            rf'{date_pattern}\s?-\s?(.*?):\s(.*)',
            # Alternative format: timestamp - name: message
            rf'{date_pattern}\s?(.*?):\s(.*)',
            # Format without dash: timestamp name: message
            rf'{date_pattern}\s?(.*?):\s(.*)',
        ]
        
        # System message patterns
        system_patterns = [
            r'(.+) added (.+)',
            r'(.+) left',
            r'(.+) removed (.+)',
            r'(.+) joined using this group\'s invite link',
            r'(.+) changed the subject from "(.+)" to "(.+)"',
            r'(.+) changed this group\'s icon',
            r'Messages and calls are end-to-end encrypted',
            r'(.+) changed the group description',
            r'(.+) changed their phone number'
        ]
        system_pattern = '|'.join(system_patterns)
        
        for line in chats:
            line = line.strip()
            if not line:
                continue
            
            message_found = False
            for pattern in message_patterns:
                match = re.search(pattern, line)
                if match:
                    try:
                        # Extract timestamp, user, and message from matching groups
                        timestamp = match.group(1)
                        user = match.group(2)
                        message = match.group(3)
                        
                        # Clean username (remove phone numbers if present)
                        user = re.sub(r'\+\d+\s*', '', user).strip()
                        
                        # Try parsing the timestamp using several formats
                        date_formats = [
                            '%d/%m/%y, %H:%M',
                            '%d/%m/%Y, %H:%M',
                            '%d.%m.%y, %H:%M:%S',
                            '%Y-%m-%d %H:%M:%S',
                            '%d/%m/%y, %H:%M:%S',
                            '%d/%m/%y, %I:%M %p',
                        ]
                        parsed_date = None
                        for fmt in date_formats:
                            try:
                                parsed_date = pd.to_datetime(timestamp, format=fmt)
                                break
                            except Exception:
                                continue
                        
                        if parsed_date is not None:
                            total_messages += 1
                            user_messages[user] += 1
                            messages_data.append([timestamp, user, message])
                            global_members.add(user)
                            message_found = True
                            break
                    except Exception as e:
                        st.error(f"Error parsing line: {line} - {str(e)}")
                        continue
            
            # If not a standard message, check if it's a system message
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
        df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
        
        # Parse timestamps with various formats
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y, %H:%M', errors='coerce')
        # Try alternative formats if needed
        mask = df['Timestamp'].isna()
        if mask.any():
            alt_formats = ['%d/%m/%Y, %H:%M', '%d.%m.%y, %H:%M:%S', '%Y-%m-%d %H:%M:%S',
                           '%d/%m/%y, %H:%M:%S', '%d/%m/%y, %I:%M %p']
            for fmt in alt_formats:
                df.loc[mask, 'Timestamp'] = pd.to_datetime(df.loc[mask, 'Timestamp'], format=fmt, errors='coerce')
                mask = df['Timestamp'].isna()
                if not mask.any():
                    break
        
        df.dropna(subset=['Timestamp'], inplace=True)
        
        # Compute week start (Monday) for each message
        df['Week Start'] = (df['Timestamp'] - pd.to_timedelta(df['Timestamp'].dt.weekday, unit='D')).dt.normalize()
        
        if df.empty:
            st.write("No messages to display")
            return
        
        # Determine overall range of weeks
        min_week_start = df['Week Start'].min()
        max_week_start = df['Week Start'].max()
        
        # Create list of Mondays from min to max
        weeks = pd.date_range(start=min_week_start, end=max_week_start, freq='W-MON')
        
        rows = []
        baseline_members = set(global_members)  # All known members
        week_counter = 1
        
        for week_start in weeks:
            week_end = week_start + pd.Timedelta(days=6)
            # Filter messages for this week (normalized week start)
            week_mask = (df['Week Start'] == week_start)
            week_messages = df[week_mask]
            
            # Update baseline with members who messaged in this week
            if not week_messages.empty:
                current_week_members = set(week_messages['Member Name'].unique())
                baseline_members = baseline_members.union(current_week_members)
            
            # For every member in baseline, record their message count for the week (0 if none)
            for member in sorted(baseline_members):
                count = week_messages[week_messages['Member Name'] == member].shape[0]
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
    
    except Exception as e:
        st.error(f"Error creating weekly message table: {str(e)}")

def display_member_statistics(messages_data):
    """
    Create Table 2: Member Statistics.
    For each member, show their first and last message dates,
    total messages, longest membership duration (in weeks),
    average weekly messages, and group activity status.
    """
    try:
        df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y, %H:%M', errors='coerce')
        
        # Try alternative formats if needed
        mask = df['Timestamp'].isna()
        if mask.any():
            alt_formats = ['%d/%m/%Y, %H:%M', '%d.%m.%y, %H:%M:%S', '%Y-%m-%d %H:%M:%S',
                           '%d/%m/%y, %H:%M:%S', '%d/%m/%y, %I:%M %p']
            for fmt in alt_formats:
                df.loc[mask, 'Timestamp'] = pd.to_datetime(df.loc[mask, 'Timestamp'], format=fmt, errors='coerce')
                mask = df['Timestamp'].isna()
                if not mask.any():
                    break
        
        df.dropna(subset=['Timestamp'], inplace=True)
        
        # Group by member to compute statistics
        grouped = df.groupby('Member Name').agg(
            first_message=('Timestamp', 'min'),
            last_message=('Timestamp', 'max'),
            total_messages=('Message', 'count')
        ).reset_index()
        
        # Calculate membership duration in weeks
        grouped['Longest Membership Duration (Weeks)'] = ((grouped['last_message'] - grouped['first_message']).dt.days / 7).round().astype(int)
        
        # Calculate average weekly messages (avoid division by zero)
        grouped['Avg. Weekly Messages'] = grouped.apply(
            lambda row: round(row['total_messages'] / max(row['Longest Membership Duration (Weeks)'], 1), 2),
            axis=1
        )
        
        # Determine activity status: Active if last message within 30 days of overall last message
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
        df = pd.DataFrame(user_messages.items(), columns=['Member Name', 'Messages'])
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
uploaded_file = st.file_uploader("Upload a zip file containing the chat log", type="zip")

if uploaded_file:
    chat_log_path = safe_extract_zip(uploaded_file)
    if chat_log_path:
        stats = parse_chat_log(chat_log_path)
        if stats:
            st.success('Chat log parsed successfully!')
            
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
                    top_users = dict(stats['user_messages'].most_common(5))
                    snippet_events = stats['join_exit_events'][:20]
                    prompt = (f"Summarize the chat log with these key points:\n"
                              f"- Top message senders: {top_users}\n"
                              f"- Group join/exit events (sample): {snippet_events}\n")
                    word_placeholder = st.empty()
                    get_llm_reply(client, prompt, word_placeholder)
        else:
            st.error("Error parsing chat log.")
    else:
        st.error("No chat log file found in the zip archive.")
