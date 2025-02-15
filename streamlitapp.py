import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter
from groq import Groq
from dateutil import parser as date_parser
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import io

##############################
# Basic Functions & Preprocessing
##############################

def clean_member_name(name):
    """Clean member name, format phone numbers consistently."""
    cleaned = name.strip()
    digits_only = re.sub(r'\D', '', cleaned)
    if len(cleaned) - len(digits_only) <= 2 and len(digits_only) >= 7:
        return f"User {digits_only[-4:]}"
    return cleaned

def initialize_llm_client():
    """Initialize Groq client with API key."""
    API_KEY = st.secrets["API_KEY"]
    return Groq(api_key=API_KEY)

def get_llm_reply(client, prompt, word_placeholder):
    """Get LLM summary using Groq API."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Analyze the chat log and summarize: most active users, "
                        "group membership changes, and engagement patterns. "
                        "Present in a clear, tabular format."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True
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

def parse_chat_log_file(uploaded_file):
    """
    Parse WhatsApp chat log file and extract messages, join events, and left events.
    
    We use two patterns for left events:
      1. A strict left pattern that exactly matches lines like:
         [04/04/22, 9:29:17 PM] Renuka Kondhalkar_SW: ‎Renuka Kondhalkar_SW left
         These are stored in strict_exit_events.
      2. A general left pattern as a fallback.
    
    The raw date strings (exactly as they appear) are saved.
    Note: All date parsing uses dayfirst=True.
    
    This version streams the file line by line to handle larger files efficiently.
    """
    try:
        # Stream the uploaded file line by line instead of reading it all into memory
        text_file = io.TextIOWrapper(uploaded_file, encoding="utf-8", errors="replace")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

    messages_data = []
    user_messages = Counter()
    member_status = {}
    exit_events = []         # General left events
    strict_exit_events = []  # Left events that exactly match our strict pattern

    # Pattern for regular messages:
    message_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?):\s(.*)$'
    )
    # Pattern for join events:
    join_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?) joined'
    )
    # General left pattern (fallback)
    left_pattern = re.compile(
        r'(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?).*?(.*?)\s+left'
    )
    # Strict left pattern: must match the entire line exactly.
    # Allows an optional invisible character (\u200e) after the colon.
    strict_left_pattern = re.compile(
        r'^\[(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*[APap][Mm])\]\s*([^:]+):\s*(?:\u200e)?(.*?)\s+left\s*$'
    )

    for line in text_file:
        line = line.strip()
        if not line:
            continue

        # Check for a regular message.
        msg_match = message_pattern.match(line)
        if msg_match:
            timestamp_str, user, message = msg_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=False, dayfirst=True)
                user = clean_member_name(user)
                messages_data.append({
                    "timestamp": timestamp,
                    "user": user,
                    "message": message
                })
                user_messages[user] += 1
                if user not in member_status:
                    member_status[user] = {
                        'first_seen': timestamp,
                        'first_seen_str': timestamp_str
                    }
            except Exception:
                continue
            continue

        # Check for join events.
        join_match = join_pattern.match(line)
        if join_match:
            timestamp_str, user = join_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=False, dayfirst=True)
                user = clean_member_name(user)
                if user not in member_status:
                    member_status[user] = {
                        'first_seen': timestamp,
                        'first_seen_str': timestamp_str
                    }
            except Exception:
                continue
            continue

        # First try the strict left pattern.
        strict_left_match = strict_left_pattern.match(line)
        if strict_left_match:
            raw_date_str, user, left_msg = strict_left_match.groups()
            try:
                timestamp = date_parser.parse(raw_date_str, fuzzy=False, dayfirst=True)
                user = clean_member_name(user)
                strict_exit_events.append({
                    'User': user,
                    'Exact Date/Time': raw_date_str
                })
                # Update member status.
                if user not in member_status:
                    member_status[user] = {
                        'first_seen': timestamp,
                        'first_seen_str': raw_date_str
                    }
                if 'left_times' not in member_status[user]:
                    member_status[user]['left_times'] = []
                    member_status[user]['left_times_str'] = []
                member_status[user]['left_times'].append(timestamp)
                member_status[user]['left_times_str'].append(raw_date_str)
                exit_events.append({
                    'user': user,
                    'timestamp': timestamp,
                    'raw': raw_date_str
                })
            except Exception:
                continue
            continue  # Skip further processing of this line.

        # If strict pattern didn't match, try the general left pattern.
        left_match = left_pattern.search(line)
        if left_match:
            timestamp_str, user = left_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=False, dayfirst=True)
                user = clean_member_name(user)
                if user not in member_status:
                    member_status[user] = {
                        'first_seen': timestamp,
                        'first_seen_str': timestamp_str
                    }
                if 'left_times' not in member_status[user]:
                    member_status[user]['left_times'] = []
                    member_status[user]['left_times_str'] = []
                member_status[user]['left_times'].append(timestamp)
                member_status[user]['left_times_str'].append(timestamp_str)
                exit_events.append({
                    'user': user,
                    'timestamp': timestamp,
                    'raw': timestamp_str
                })
            except Exception:
                continue

    total_members = len(member_status)
    left_members = sum(1 for m in member_status.values() if m.get('left_times'))
    current_members = total_members - left_members

    return {
        'messages_data': messages_data,
        'user_messages': user_messages,
        'member_status': member_status,
        'total_members': total_members,
        'current_members': current_members,
        'left_members': left_members,
        'exit_events': exit_events,
        'strict_exit_events': strict_exit_events
    }

def create_member_timeline(stats):
    """Create a timeline (DataFrame) showing join and left events with running totals."""
    events = []
    for member, status in stats['member_status'].items():
        if status.get('first_seen'):
            events.append({
                'date': status['first_seen'],
                'change': 1,
                'event_type': 'join',
                'member': member
            })
        if 'left_times' in status:
            for lt in status['left_times']:
                events.append({
                    'date': lt,
                    'change': -1,
                    'event_type': 'left',
                    'member': member
                })
    if not events:
        return pd.DataFrame()
    events.sort(key=lambda x: x['date'])
    timeline_data = []
    member_count = 0
    for event in events:
        member_count += event['change']
        timeline_data.append({
            'Date': event['date'],
            'Member Count': member_count,
            'Event': f"{event['member']} {event['event_type']}",
            'Event Type': event['event_type']
        })
    return pd.DataFrame(timeline_data)

def create_weekly_breakdown(stats):
    """Create a weekly breakdown table of messages and member status."""
    if not stats['messages_data']:
        return pd.DataFrame()
    df = pd.DataFrame(stats['messages_data'])
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except ValueError as e:
        st.error(f"Error converting timestamps: {e}")
        return pd.DataFrame()
    df.dropna(subset=['timestamp'], inplace=True)
    if df.empty:
        return pd.DataFrame()
    user_msgs = df.groupby('user').size().reset_index(name='Messages Sent')
    weekly_data = []
    for _, row in user_msgs.iterrows():
        user = row['user']
        status = stats['member_status'].get(user, {})
        weekly_data.append({
            'Member Name': user,
            'Messages Sent': row['Messages Sent'],
            'Current Status': 'Left' if status.get('left_times') else 'Present'
        })
    return pd.DataFrame(weekly_data)

def fetch_stats(stats, df):
    """Compute top-level statistics from the chat data."""
    total_messages = len(stats['messages_data'])
    total_words = sum(len(message.split()) for message in df['message'])
    media_messages = sum(1 for message in df['message'] if "<Media omitted>" in message)
    link_pattern = re.compile(r'https?://\S+')
    links_shared = sum(1 for message in df['message'] if link_pattern.search(message))
    return total_messages, total_words, media_messages, links_shared

def create_wordcloud(selected_user, df):
    """Generate a word cloud image from messages."""
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    text = " ".join(message for message in df['message'])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wordcloud

##############################
# Exit Events Table Function
##############################

def create_exit_events_table(stats):
    """
    Create a separate table for exit events with two columns:
    | Name of Exit Person | Exit Date & Time (exactly from the txt file) |
    
    This table uses the raw timestamp text from the chat log.
    """
    if stats.get('strict_exit_events'):
        df = pd.DataFrame(stats['strict_exit_events'])
        df = df.rename(columns={
            "User": "Name of Exit Person", 
            "Exact Date/Time": "Exit Date & Time"
        })
        return df[["Name of Exit Person", "Exit Date & Time"]]
    elif stats.get('exit_events'):
        df = pd.DataFrame(stats['exit_events'])
        df = df.rename(columns={
            "user": "Name of Exit Person", 
            "raw": "Exit Date & Time"
        })
        return df[["Name of Exit Person", "Exit Date & Time"]]
    else:
        return pd.DataFrame()

def create_member_activity_table(stats):
    """Create a table of member activity with join and left events."""
    activity_data = []
    for member, status in stats['member_status'].items():
        current_status = 'Left' if status.get('left_times') else 'Active'
        activity_data.append({
            'Member Name': member,
            'Message Count': stats['user_messages'].get(member, 0),
            'Exit Events': len(status.get('left_times', [])),
            'Activity Status': current_status,
            'Join Date': status.get('first_seen_str', status['first_seen'].strftime('%d %b %Y')),
            'Left Date': (status.get('left_times_str', [])[0] if status.get('left_times_str') else 'Present')
        })
    df = pd.DataFrame(activity_data)
    if not df.empty:
        df = df.sort_values(by=['Message Count', 'Member Name'], ascending=[False, True])
    return df

##############################
# Main App Function
##############################

def main():
    st.sidebar.title("Chat Log Analyzer")
    uploaded_file = st.sidebar.file_uploader("Upload WhatsApp chat log (TXT format)", type="txt")
    
    if uploaded_file is not None:
        with st.spinner("Parsing chat log..."):
            stats = parse_chat_log_file(uploaded_file)
        if stats is None:
            st.error("Error parsing the file.")
            return
        
        df = pd.DataFrame(stats['messages_data'])
        if df.empty:
            st.error("No messages found.")
            return
        
        user_list = list(stats['user_messages'].keys())
        if "group_notification" in user_list:
            user_list.remove("group_notification")
        user_list.sort()
        user_list.insert(0, "Overall")
        selected_user = st.sidebar.selectbox("Show Analysis with respect to", user_list)
        
        if st.sidebar.button("Show Analysis"):
            st.title("Top Statistics")
            total_messages, total_words, media_messages, links_shared = fetch_stats(stats, df)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.header("Total Messages")
                st.title(total_messages)
            with col2:
                st.header("Total Words")
                st.title(total_words)
            with col3:
                st.header("Media Shared")
                st.title(media_messages)
            with col4:
                st.header("Links Shared")
                st.title(links_shared)
            
            # Existing Exit Events Summary (for reference)
            st.markdown("### Member Exit Summary")
            if stats['exit_events']:
                exit_df = pd.DataFrame(stats['exit_events'])
                exit_df['timestamp'] = pd.to_datetime(exit_df['timestamp'])
                exit_df = exit_df.sort_values('timestamp')
                exit_counts = exit_df['user'].value_counts().reset_index()
                exit_counts.columns = ['User', 'Number of Exits']
                st.markdown("#### Exit Events Timeline")
                st.dataframe(exit_df.assign(
                    exit_date=exit_df['timestamp'].dt.strftime('%d %b %Y')
                )[['user', 'exit_date']])
                st.markdown("#### Exit Events Count")
                st.dataframe(exit_counts)
                st.metric("Total Members Left", stats['left_members'])
            else:
                st.write("No exit events recorded")
            
            # NEW: Separate Exit Events Table for left events.
            st.markdown("### Exit Events Table")
            left_events_table = create_exit_events_table(stats)
            if not left_events_table.empty:
                st.dataframe(left_events_table)
            else:
                st.write("No left events available")
            
            st.markdown("### Group Member Timeline")
            timeline_df = create_member_timeline(stats)
            if not timeline_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timeline_df['Date'],
                    y=timeline_df['Member Count'],
                    mode='lines',
                    name='Member Count',
                    line=dict(color='#2E86C1', width=2)
                ))
                fig.update_layout(
                    title='Group Member Count Timeline',
                    xaxis_title='Date',
                    yaxis_title='Number of Members',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Member Activity Status")
            activity_df = create_member_activity_table(stats)
            st.dataframe(activity_df, use_container_width=True)
            
            st.markdown("### Weekly Message & Member Analysis")
            weekly_df = create_weekly_breakdown(stats)
            st.dataframe(weekly_df)
            
            st.markdown("### Message Distribution")
            message_df = pd.DataFrame(list(stats['user_messages'].items()), columns=['Member', 'Messages'])
            fig = px.bar(message_df, x='Member', y='Messages', title='Messages per Member', color='Messages')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Word Cloud for Frequent Words")
            wc = create_wordcloud(selected_user, df)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            
            st.markdown("### Chat Analysis Summary (LLM)")
            if st.button("Generate LLM Summary"):
                client = initialize_llm_client()
                prompt = (f"Analyze this chat log:\n"
                          f"- Total members: {stats['total_members']}\n"
                          f"- Current members: {stats['current_members']}\n"
                          f"- Top contributors: {dict(Counter(stats['user_messages']).most_common(5))}\n")
                placeholder = st.empty()
                get_llm_reply(client, prompt, placeholder)

if __name__ == "__main__":
    main()
