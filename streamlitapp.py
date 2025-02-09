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

# Custom CSS for structured layout and spacing
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

# Initialize LLM Client
API_KEY = st.secrets["API_KEY"]
client = Groq(api_key=API_KEY)

def get_llm_reply(prompt):
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
        st.error(f"An error occurred: {e}")
        return None

# Function to extract the zip file
def extract_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("extracted")
    return "extracted"

# Function to parse the chat log.
def parse_chat_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        chats = file.readlines()
    
    total_messages = 0
    user_messages = Counter()
    join_exit_events = []
    messages_data = []  # Detailed message records: timestamp, user, message
    message_pattern = re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}) - (.*?): (.*)')
    join_exit_pattern = re.compile(r'(.*) added (.*)|(.+) left')
    
    for line in chats:
        match = message_pattern.match(line)
        if match:
            total_messages += 1
            timestamp, user, message = match.groups()
            user_messages[user] += 1
            messages_data.append([timestamp, user, message])
        
        event_match = join_exit_pattern.match(line)
        if event_match:
            join_exit_events.append(line.strip())
    
    return {
        'total_messages': total_messages,
        'user_messages': user_messages,
        'join_exit_events': join_exit_events,
        'messages_data': messages_data
    }

# Table 1: Weekly Message Breakdown (fix applied)
def display_weekly_messages_table(messages_data):
    df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
    # Parse the timestamp (expected format: "23/02/23, 02:21")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y, %H:%M', errors='coerce')
    # Compute the week start (Monday) and normalize to midnight
    df['Week Start'] = (df['Timestamp'] - pd.to_timedelta(df['Timestamp'].dt.weekday, unit='D')).dt.normalize()
    
    if df.empty:
        st.write("No messages to display")
        return
    
    # Determine the overall range of weeks
    min_week_start = df['Week Start'].min()
    max_week_start = df['Week Start'].max()
    
    # Create a list of Mondays from min to max
    weeks = pd.date_range(start=min_week_start, end=max_week_start, freq='W-MON')
    
    rows = []
    baseline_members = set()  # Running set of members seen so far
    week_counter = 1
    
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)
        # Filter messages for the current week (exact match on normalized dates)
        week_mask = (df['Week Start'] == week_start)
        week_messages = df[week_mask]
        
        # Update baseline if messages exist in the current week
        if not week_messages.empty:
            current_week_members = set(week_messages['Member Name'].unique())
            baseline_members = baseline_members.union(current_week_members)
        
        # List each member from baseline even if they didn't message in the current week
        if baseline_members:
            for member in sorted(baseline_members):
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

# Table 2: Member Statistics
def display_member_statistics(messages_data):
    df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y, %H:%M', errors='coerce')
    
    # For each member: first message, last message, total messages
    grouped = df.groupby('Member Name').agg(
        min_timestamp=('Timestamp', 'min'),
        max_timestamp=('Timestamp', 'max'),
        total_messages=('Message', 'count')
    ).reset_index()
    
    # Membership duration (in weeks)
    grouped['Longest Membership Duration (Weeks)'] = ((grouped['max_timestamp'] - grouped['min_timestamp']).dt.days / 7).round().astype(int)
    
    # Average weekly messages
    grouped['Avg. Weekly Messages'] = grouped.apply(
        lambda row: round(row['total_messages'] / row['Longest Membership Duration (Weeks)'], 2)
        if row['Longest Membership Duration (Weeks)'] > 0 else row['total_messages'], axis=1)
    
    # Activity status: Active if last message was within 30 days of overall last message
    overall_last_date = df['Timestamp'].max()
    grouped['Group Activity Status'] = grouped['max_timestamp'].apply(
        lambda x: 'Active' if (overall_last_date - x).days <= 30 else 'Inactive')
    
    grouped.rename(columns={'Member Name': 'Unique Member Name'}, inplace=True)
    table2 = grouped[['Unique Member Name', 'Group Activity Status', 'Longest Membership Duration (Weeks)', 'Avg. Weekly Messages']]
    
    st.markdown("### Table 2: Member Statistics")
    st.dataframe(table2)

# Bar chart for total messages per user
def display_total_messages_chart(user_messages):
    df = pd.DataFrame(user_messages.items(), columns=['Member Name', 'Messages'])
    fig = px.bar(df, x='Member Name', y='Messages', 
                 title='Total Messages Sent by Each User',
                 color='Messages')
    st.markdown("<div class='chart-container'></div>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

# Streamlit app layout
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
        st.success('Chat log parsed successfully!')
        
        # Display Table 1: Weekly Message Breakdown
        display_weekly_messages_table(stats['messages_data'])
        
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
        st.error('No chat log found in the zip file.')
