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
from collections import Counter
from groq import Groq

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

def get_llm_reply(prompt):
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
        # Stream response and update placeholder in real time.
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            response += delta
            word_placeholder.write(response)
        return response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# -------------------------------
# Function to extract the zip file
# -------------------------------
def extract_zip(zip_file):
    """Extract the contents of the zip file to the 'extracted' folder."""
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("extracted")
        return "extracted"
    except Exception as e:
        st.error(f"Error extracting zip file: {e}")
        return None

# -------------------------------
# Function to parse the chat log.
# -------------------------------
def parse_chat_log(file_path):
    """Parse the chat log file and return message statistics along with global membership."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            chats = file.readlines()
        
        total_messages = 0
        user_messages = Counter()
        join_exit_events = []
        messages_data = []  # Each element: [timestamp_str, user, message]
        global_members = set()
        
        # Regular expression patterns
        message_pattern = re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}) - (.*?): (.*)')
        join_exit_pattern = re.compile(r'(.*) added (.*)|(.+) left')
        
        for line in chats:
            line = line.strip()
            if not line:
                continue
            
            match = message_pattern.match(line)
            if match:
                total_messages += 1
                timestamp, user, message = match.groups()
                user_messages[user] += 1
                messages_data.append([timestamp, user, message])
                global_members.add(user)
            
            event_match = join_exit_pattern.match(line)
            if event_match:
                join_exit_events.append(line)
                # Optionally, you could extract new members from join events here.
        
        return {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'join_exit_events': join_exit_events,
            'messages_data': messages_data,
            'global_members': sorted(global_members)
        }
    
    except Exception as e:
        st.error(f"Error parsing chat log: {e}")
        return None

# -------------------------------
# Table 1: Weekly Message Breakdown
# -------------------------------
def display_weekly_messages_table(messages_data, global_members):
    """
    Create a table showing weekly message breakdown.
    Each week (starting Monday) is displayed with its date range.
    All members (from global_members) are listed for every week;
    if a member didn't send any messages in a week, count is 0.
    """
    df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
    
    # Parse timestamps with expected format; adjust format if needed.
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y, %H:%M', errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)  # Remove invalid timestamps
    
    # Compute week start (Monday) for each message.
    df['Week Start'] = (df['Timestamp'] - pd.to_timedelta(df['Timestamp'].dt.weekday, unit='D')).dt.normalize()
    
    if df.empty:
        st.write("No messages to display")
        return
    
    # Determine overall range of weeks
    min_week_start = df['Week Start'].min()
    max_week_start = df['Week Start'].max()
    
    # Create a list of Mondays from min_week_start to max_week_start
    weeks = pd.date_range(start=min_week_start, end=max_week_start, freq='W-MON')
    
    rows = []
    baseline_members = set(global_members)  # Start with all known members
    week_counter = 1
    
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)
        # Filter messages exactly for this week (based on normalized Week Start)
        week_mask = (df['Week Start'] == week_start)
        week_messages = df[week_mask]
        
        # Update baseline: add any new members who messaged in this week.
        if not week_messages.empty:
            current_week_members = set(week_messages['Member Name'].unique())
            baseline_members = baseline_members.union(current_week_members)
        
        # For every member in baseline, get the message count (0 if none)
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

# -------------------------------
# Table 2: Member Statistics
# -------------------------------
def display_member_statistics(messages_data):
    """
    Display member statistics including:
    - Unique Member Name
    - Group Activity Status (Active if last message within 30 days)
    - Longest Membership Duration (Weeks)
    - Average Weekly Messages
    """
    df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y, %H:%M', errors='coerce')
    
    # Group by member to get first and last message, total messages
    grouped = df.groupby('Member Name').agg(
        first_message=('Timestamp', 'min'),
        last_message=('Timestamp', 'max'),
        total_messages=('Message', 'count')
    ).reset_index()
    
    # Calculate membership duration in weeks
    grouped['Longest Membership Duration (Weeks)'] = ((grouped['last_message'] - grouped['first_message']).dt.days / 7).round().astype(int)
    
    # Calculate average weekly messages (avoid division by zero)
    grouped['Avg. Weekly Messages'] = grouped.apply(
        lambda row: round(row['total_messages'] / row['Longest Membership Duration (Weeks)'], 2)
        if row['Longest Membership Duration (Weeks)'] > 0 else row['total_messages'], axis=1)
    
    # Determine activity status: Active if last message was within 30 days of overall last message
    overall_last_date = df['Timestamp'].max()
    grouped['Group Activity Status'] = grouped['last_message'].apply(
        lambda x: 'Active' if (overall_last_date - x).days <= 30 else 'Inactive')
    
    grouped.rename(columns={'Member Name': 'Unique Member Name'}, inplace=True)
    table2 = grouped[['Unique Member Name', 'Group Activity Status', 'Longest Membership Duration (Weeks)', 'Avg. Weekly Messages']]
    
    st.markdown("### Table 2: Member Statistics")
    st.dataframe(table2)

# -------------------------------
# Bar Chart: Total Messages per User
# -------------------------------
def display_total_messages_chart(user_messages):
    df = pd.DataFrame(user_messages.items(), columns=['Member Name', 'Messages'])
    fig = px.bar(df, x='Member Name', y='Messages', 
                 title='Total Messages Sent by Each User',
                 color='Messages')
    st.markdown("<div class='chart-container'></div>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Main Streamlit App Layout
# -------------------------------
st.title("Structured Chat Log Analyzer")
uploaded_file = st.file_uploader("Upload a zip file containing the chat log", type="zip")

if uploaded_file:
    extract_path = extract_zip(uploaded_file)
    chat_log_path = None
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.txt'):
                chat_log_path = os.path.join(root, file)
                break
    
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
                    get_llm_reply(prompt)
        else:
            st.error("Error parsing chat log.")
    else:
        st.error('No chat log found in the zip file.')
