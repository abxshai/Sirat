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

def extract_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("extracted")
    return "extracted"

def parse_timestamp(timestamp_str):
    """Try multiple timestamp formats to robustly parse chat timestamps."""
    for fmt in ['%d/%m/%y, %H:%M', '%d/%m/%Y, %I:%M %p']:
        try:
            return pd.to_datetime(timestamp_str, format=fmt)
        except Exception:
            continue
    return pd.NaT

@st.cache_data(show_spinner=False)
def parse_chat_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        chats = file.readlines()
    
    total_messages = 0
    user_messages = Counter()
    join_exit_events = []
    messages_data = []  # Each element: [timestamp_str, user, message]
    global_members = set()
    
    # Regular expressions for messages and join/left events.
    message_pattern = re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}) - (.*?): (.*)')
    join_pattern = re.compile(r'(.*) added (.*)')
    left_pattern = re.compile(r'(.*) left')
    
    for line in chats:
        match = message_pattern.match(line)
        if match:
            total_messages += 1
            timestamp_str, user, message = match.groups()
            user_messages[user] += 1
            messages_data.append([timestamp_str, user, message])
            global_members.add(user)
        
        join_match = join_pattern.match(line)
        if join_match:
            new_member = join_match.group(2).strip()
            global_members.add(new_member)
            join_exit_events.append(line.strip())
        
        left_match = left_pattern.match(line)
        if left_match:
            member = left_match.group(1).strip()
            global_members.add(member)
            join_exit_events.append(line.strip())
    
    return {
        'total_messages': total_messages,
        'user_messages': user_messages,
        'join_exit_events': join_exit_events,
        'messages_data': messages_data,
        'global_members': sorted(global_members)
    }

def display_weekly_messages_table(messages_data, global_members):
    # Check if messages_data exists and is not empty
    if messages_data is None or len(messages_data) == 0:
        st.warning("No data available for weekly messages.")
        return
    
    # Create DataFrame with defined columns
    df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
    
    # Ensure 'Timestamp' exists and convert it safely to datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = df['Timestamp'].apply(parse_timestamp)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)  # Remove rows where conversion failed
        
        # Compute the Week Start (Monday) and normalize to midnight
        df['Week Start'] = (df['Timestamp'] - pd.to_timedelta(df['Timestamp'].dt.weekday, unit='D')).dt.normalize()
    else:
        st.error("Error: 'Timestamp' column is missing from data.")
        return
    
    # Ensure global_members is valid
    if global_members is None or len(global_members) == 0:
        st.warning("No data available for global members.")
        return

    # Add a selectbox filter for Member Name
    member_options = ["All Members"] + sorted(global_members)
    selected_member = st.selectbox("Select Member (or All Members)", member_options)
    
    # Determine the full range of weeks.
    min_week_start = df['Week Start'].min()
    max_week_start = df['Week Start'].max()
    weeks = pd.date_range(start=min_week_start, end=max_week_start, freq='W-MON')
    
    rows = []
    # Use global_members as baseline so every week includes all members.
    baseline_members = set(global_members)
    week_counter = 1
    
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)
        week_mask = (df['Week Start'] == week_start)
        week_messages = df[week_mask]
        
        # Update baseline with any new members messaging this week.
        if not week_messages.empty:
            current_week_members = set(week_messages['Member Name'].unique())
            baseline_members = baseline_members.union(current_week_members)
        
        # If a specific member is selected, filter baseline to only that member.
        members_to_show = sorted(baseline_members)
        if selected_member != "All Members":
            members_to_show = [selected_member]
        
        for member in members_to_show:
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

def display_member_statistics(messages_data):
    df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
    df['Timestamp'] = df['Timestamp'].apply(parse_timestamp)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    # Group by member.
    grouped = df.groupby('Member Name').agg(
        min_timestamp=('Timestamp', 'min'),
        max_timestamp=('Timestamp', 'max'),
        total_messages=('Message', 'count')
    ).reset_index()
    
    # Membership duration in weeks.
    grouped['Longest Membership Duration (Weeks)'] = ((grouped['max_timestamp'] - grouped['min_timestamp']).dt.days / 7).round().astype(int)
    grouped['Avg. Weekly Messages'] = grouped.apply(
        lambda row: round(row['total_messages'] / row['Longest Membership Duration (Weeks)'], 2)
        if row['Longest Membership Duration (Weeks)'] > 0 else row['total_messages'], axis=1)
    
    overall_last_date = df['Timestamp'].max()
    grouped['Group Activity Status'] = grouped['max_timestamp'].apply(
        lambda x: 'Active' if (overall_last_date - x).days <= 30 else 'Inactive')
    
    grouped.rename(columns={'Member Name': 'Unique Member Name'}, inplace=True)
    table2 = grouped[['Unique Member Name', 'Group Activity Status', 'Longest Membership Duration (Weeks)', 'Avg. Weekly Messages']]
    
    st.markdown("### Table 2: Member Statistics")
    st.dataframe(table2)

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
        display_weekly_messages_table(stats['messages_data'], stats['global_members'])
        
        # Display Table 2: Member Statistics
        display_member_statistics(stats['messages_data'])
        
        # Display bar chart for total messages per user.
        display_total_messages_chart(stats['user_messages'])
        
        # LLM-based summary component.
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
