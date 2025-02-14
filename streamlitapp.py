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
    Parse WhatsApp chat log file and count exit events based on the 'left' keyword.
    Dates are parsed strictly (fuzzy=False) and the raw date string is saved.
    
    We use two patterns for left events:
      1. A strict pattern that must match the entire line (e.g. system left messages).
      2. A general left pattern as a fallback.
    
    The strict matches are stored in the 'strict_exit_events' list for a separate table.
    """
    try:
        content = uploaded_file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

    messages_data = []
    user_messages = Counter()
    member_status = {}
    exit_events = []         # General exit events list
    strict_exit_events = []  # List for strictly matched left events

    # Regular expression patterns for regular messages and join events.
    message_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?):\s(.*)$'
    )
    join_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?) joined'
    )
    # General left pattern.
    left_pattern = re.compile(
        r'(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?).*?(.*?)\s+left'
    )
    # Strict left pattern – it must match the whole line.
    # Here we allow an optional invisible character (e.g., U+200E) after the colon.
    strict_left_pattern = re.compile(
        r'^\[(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*[APap][Mm])\]\s*(.*?):\s*(?:\u200e)?(.*?)\s+left\s*$'
    )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # First, check for a regular message.
        msg_match = message_pattern.match(line)
        if msg_match:
            timestamp_str, user, message = msg_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=False)
                user = clean_member_name(user)
                messages_data.append({
                    "timestamp": timestamp,
                    "user": user,
                    "message": message
                })
                user_messages[user] += 1
                if user not in member_status:
                    member_status[user] = {'first_seen': timestamp, 'first_seen_str': timestamp_str}
            except Exception:
                continue
            continue

        # Check for join events.
        join_match = join_pattern.match(line)
        if join_match:
            timestamp_str, user = join_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=False)
                user = clean_member_name(user)
                if user not in member_status:
                    member_status[user] = {'first_seen': timestamp, 'first_seen_str': timestamp_str}
            except Exception:
                continue
            continue

        # First try the strict left pattern.
        strict_left_match = strict_left_pattern.match(line)
        if strict_left_match:
            raw_date_str, user, message_text = strict_left_match.groups()
            try:
                timestamp = date_parser.parse(raw_date_str, fuzzy=False)
                user = clean_member_name(user)
                strict_exit_events.append({
                    'user': user,
                    'raw': raw_date_str
                })
                if user not in member_status:
                    member_status[user] = {'first_seen': timestamp, 'first_seen_str': raw_date_str}
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
                timestamp = date_parser.parse(timestamp_str, fuzzy=False)
                user = clean_member_name(user)
                if user not in member_status:
                    member_status[user] = {'first_seen': timestamp, 'first_seen_str': timestamp_str}
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
    """Create timeline with explicit join and exit events (each exit event subtracts one member)."""
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
    """Simplified weekly breakdown without date parsing issues."""
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
    Create a separate table for strict exit events (system messages with "left" that exactly match the format),
    showing the exact timing (raw string) and the person who left.
    """
    strict_events = stats.get('strict_exit_events', [])
    if not strict_events:
        return pd.DataFrame()
    df = pd.DataFrame(strict_events)
    df = df.rename(columns={'user': 'User', 'raw': 'Exact Date/Time'})
    return df[['User', 'Exact Date/Time']]

##############################
# Member Activity Table
##############################

def create_member_activity_table(stats):
    """Create activity status table with exit event counts using raw date strings."""
    activity_data = []
    for member, status in stats['member_status'].items():
        if member == "group_notification":
            continue
        message_count = stats['user_messages'].get(member, 0)
        exit_count = len(status.get('left_times', []))
        activity_data.append({
            'Member Name': member,
            'Message Count': message_count,
            'Exit Events': exit_count,
            'Activity Status': 'Left' if status.get('left_times') else 'Active',
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
            
            # Existing Exit Events Summary
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
            
            # NEW: Separate Exit Events Table for strictly matched left messages.
            st.markdown("### Exit Events Table")
            exit_events_table = create_exit_events_table(stats)
            if not exit_events_table.empty:
                st.dataframe(exit_events_table)
            else:
                st.write("No exit events available")
            
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
