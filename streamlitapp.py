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

def display_weekly_messages_table(messages_data):
    """Display a table showing the number of messages each member sent each week."""
    df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y, %H:%M', errors='coerce')
    except Exception as e:
        st.error(f"Error parsing dates: {e}")
        return
    
    df['Week Start'] = df['Timestamp'].apply(lambda x: x - pd.Timedelta(days=x.weekday()))
    
    if df.empty:
        st.write("No messages to display")
        return
    
    min_week = df['Week Start'].min()
    max_week = df['Week Start'].max()
    weeks = pd.date_range(start=min_week, end=max_week, freq='W-MON')
    
    all_members = set(df['Member Name'].unique())
    
    rows = []
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)
        week_mask = df['Week Start'] == week_start
        week_df = df[week_mask]
        
        for member in sorted(all_members):
            count = week_df[week_df['Member Name'] == member].shape[0]
            rows.append({
                'Week': f"Week {(week_start - min_week).days // 7 + 1}",
                'Week Duration': f"{week_start.strftime('%d %b %Y')} - {week_end.strftime('%d %b %Y')}",
                'Member Name': member,
                'Number of Messages Sent': count
            })
    
    weekly_df = pd.DataFrame(rows)
    st.markdown("### Table 1: Weekly Message Breakdown")
    st.dataframe(weekly_df)

def display_member_statistics(messages_data):
    """Display statistics for each member including first message, last message, total messages, and activity status."""
    df = pd.DataFrame(messages_data, columns=['Timestamp', 'Member Name', 'Message'])
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%y, %H:%M', errors='coerce')
    except Exception as e:
        st.error(f"Error parsing dates: {e}")
        return
    
    grouped = df.groupby('Member Name').agg(
        first_message=('Timestamp', 'min'),
        last_message=('Timestamp', 'max'),
        total_messages=('Message', 'count')
    ).reset_index()
    
    grouped['membership_duration'] = (grouped['last_message'] - grouped['first_message']).dt.days
    grouped['membership_duration_weeks'] = (grouped['membership_duration'] / 7).round().astype(int)
    
    overall_last_date = df['Timestamp'].max()
    grouped['activity_status'] = grouped['last_message'].apply(
        lambda x: 'Active' if (overall_last_date - x).days <= 30 else 'Inactive'
    )
    
    table = grouped[['Member Name', 'activity_status', 'membership_duration_weeks', 'total_messages']]
    table.columns = ['Member Name', 'Activity Status', 'Membership Duration (Weeks)', 'Total Messages']
    
    st.markdown("### Table 2: Member Statistics")
    st.dataframe(table)

def display_total_messages_chart(user_messages):
    """Display a bar chart showing total messages per user."""
    df = pd.DataFrame(list(user_messages.items()), columns=['Member Name', 'Messages'])
    fig = px.bar(df, x='Member Name', y='Messages', 
                 title='Total Messages Sent by Each User',
                 color='Messages')
    st.plotly_chart(fig, use_container_width=True)

def extract_zip(zip_file):
    """Extract the contents of the zip file to a directory."""
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("extracted")
        return "extracted"
    except Exception as e:
        st.error(f"Error extracting zip file: {e}")
        return None

def parse_chat_log(file_path):
    """Parse the chat log file and return message statistics."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        total_messages = 0
        user_messages = Counter()
        join_exit_events = []
        messages_data = []
        
        message_pattern = re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}) - (.*?): (.*)')
        join_exit_pattern = re.compile(r'(.*) added (.*)|(.+) left')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = message_pattern.match(line)
            if match:
                total_messages += 1
                timestamp, user, message = match.groups()
                user_messages[user] += 1
                messages_data.append([timestamp, user, message])
            
            event_match = join_exit_pattern.match(line)
            if event_match:
                join_exit_events.append(line)
        
        return {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'join_exit_events': join_exit_events,
            'messages_data': messages_data
        }
    
    except Exception as e:
        st.error(f"Error parsing chat log: {e}")
        return None

def get_llm_reply(prompt, API_KEY):
    """Get a response from the Groq LLM using the provided prompt."""
    try:
        client = Groq(api_key=API_KEY)
        response = client.chat.completions.create(
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
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def main():
    st.title("Structured Chat Log Analyzer")
    
    API_KEY = st.secrets["API_KEY"]
    
    uploaded_file = st.file_uploader("Upload a zip file containing the chat log", type="zip")
    
    if uploaded_file:
        extract_path = extract_zip(uploaded_file)
        if extract_path:
            for root, _, files in os.walk(extract_path):
                txt_files = [os.path.join(root, f) for f in files if f.endswith('.txt')]
                if txt_files:
                    chat_log_path = txt_files[0]
                    stats = parse_chat_log(chat_log_path)
                    if stats:
                        st.success('Chat log parsed successfully!')
                        
                        display_weekly_messages_table(stats['messages_data'])
                        display_member_statistics(stats['messages_data'])
                        display_total_messages_chart(stats['user_messages'])
                        
                        st.markdown("### LLM Summary of Chat Log")
                        if st.button("Generate Summary"):
                            with st.spinner("Analyzing chat log..."):
                                top_users = dict(stats['user_messages'].most_common(5))
                                snippet_events = stats['join_exit_events'][:20]
                                prompt = (f"Summarize the chat log with these key points:\n"
                                           f"- Top message senders: {top_users}\n"
                                           f"- Group join/exit events (sample): {snippet_events}\n")
                                summary = get_llm_reply(prompt, API_KEY)
                                st.write(summary)
                    else:
                        st.error('No valid data found in the chat log.')
                else:
                    st.error('No .txt file found in the zip archive.')
            else:
                st.error('No files found in the extracted directory.')

if __name__ == "__main__":
    main()
