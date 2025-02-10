import streamlit as st
import pandas as pd
import plotly.express as px
import re
from collections import Counter
from datetime import datetime, timedelta
from groq import Groq

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
                "content": "Analyze the chat log, summarize key details like highest message sender, people who joined the group, and joining/exiting trends weekly or monthly."
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

# Function to parse the chat log
def parse_chat_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        chats = file.readlines()
    
    total_messages = 0
    user_messages = Counter()
    join_exit_events = []
    messages_data = []
    
    message_pattern = re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [APap][Mm]) - ([^:]+): (.*)')
    join_exit_pattern = re.compile(r'(.*) added (.*)|(.+) left')
    
    for line in chats:
        match = message_pattern.match(line)
        if match:
            total_messages += 1
            timestamp_str, user, message = match.groups()
            timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y, %I:%M %p")
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

# Function to display weekly message breakdown
def display_weekly_messages(messages_data):
    df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
    df['Week Start'] = df['Timestamp'].apply(lambda x: x - timedelta(days=x.weekday()))
    
    weekly_counts = df.groupby(['Week Start', 'Member Name']).size().reset_index(name='Number of Messages Sent')
    st.markdown("### Table 1: Weekly Message Breakdown")
    st.dataframe(weekly_counts)

def display_member_statistics(messages_data):
    df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    grouped = df.groupby('Member Name').agg(
        first_message=('Timestamp', 'min'),
        last_message=('Timestamp', 'max'),
        total_messages=('Message', 'count')
    ).reset_index()
    
    grouped['Longest Membership Duration (Weeks)'] = ((grouped['last_message'] - grouped['first_message']).dt.days / 7).astype('Int64')
    grouped['Avg. Weekly Messages'] = grouped['total_messages'] / grouped['Longest Membership Duration (Weeks)']
    grouped['Avg. Weekly Messages'] = grouped['Avg. Weekly Messages'].fillna(0).round(2)
    
    last_message_date = df['Timestamp'].max()
    grouped['Group Activity Status'] = grouped['last_message'].apply(lambda x: 'Active' if (last_message_date - x).days <= 30 else 'Inactive')
    
    st.markdown("### Table 2: Member Statistics")
    st.dataframe(grouped[['Member Name', 'Group Activity Status', 'Longest Membership Duration (Weeks)', 'Avg. Weekly Messages']])

# Streamlit app
st.title("Structured Chat Log Analyzer")
uploaded_file = st.file_uploader("Upload a text file containing the chat log", type="txt")

if uploaded_file:
    with open("temp_chat.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    stats = parse_chat_log("temp_chat.txt")
    st.success('Chat log parsed successfully!')
    
    display_weekly_messages(stats['messages_data'])
    display_member_statistics(stats['messages_data'])
    
    # LLM-based summary component
    st.markdown("### LLM Summary of Chat Log")
    if st.button("Generate Summary"):
        with st.spinner("Analyzing chat log..."):
            top_users = dict(stats['user_messages'].most_common(5))
            prompt = f"Summarize the chat log with these key points:\n\n- Top message senders: {top_users}\n- Group join and exit events: {stats['join_exit_events'][:20]}"
            word_placeholder = st.empty()
            get_llm_reply(prompt)
