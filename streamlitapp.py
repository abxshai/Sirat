import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from groq import Groq
from dateutil import parser as date_parser
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
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
    """Parse WhatsApp chat log file."""
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
    # Patterns for messages, join, and left events
    message_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?):\s(.*)$'
    )

    join_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?) joined'
    )
    left_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?) left'
    )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Standard messages
        msg_match = message_pattern.match(line)
        if msg_match:
            timestamp_str, user, message = msg_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                user = clean_member_name(user)
                messages_data.append({
                    "timestamp": timestamp,
                    "user": user,
                    "message": message
                })
                user_messages[user] += 1
                if user not in member_status:
                    member_status[user] = {'first_seen': timestamp, 'last_left': None}
            except Exception:
                continue
            continue

        # Group join/left events (if any)
        join_match = join_pattern.match(line)
        left_match = left_pattern.match(line)

        if join_match:
            timestamp_str, user = join_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                user = clean_member_name(user)
                if user not in member_status:
                    member_status[user] = {'first_seen': timestamp, 'last_left': None}
            except Exception:
                continue
        elif left_match:
            timestamp_str, user = left_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                user = clean_member_name(user)
                if user in member_status:
                    member_status[user]['last_left'] = timestamp
            except Exception:
                continue

    current_members = sum(1 for m in member_status.values() if not m['last_left'])
    return {
        'messages_data': messages_data,
        'user_messages': user_messages,
        'member_status': member_status,
        'total_members': len(member_status),
        'current_members': current_members
    }

##############################
# Analysis Functions
##############################

def create_member_activity_table(stats):
    """Create a table of member activity."""
    activity_data = []
    for member, status in stats['member_status'].items():
        message_count = stats['user_messages'].get(member, 0)
        activity_data.append({
            'Member Name': member,
            'Message Count': message_count,
            'Activity Status': 'Active' if message_count > 0 else 'Inactive',
            'Join Date': status['first_seen'].strftime('%d %b %Y'),
            'Left Date': status['last_left'].strftime('%d %b %Y') if status['last_left'] else 'Still in group',
            'Current Status': 'Left' if status['last_left'] else 'Present'
        })
    
    return pd.DataFrame(activity_data).sort_values(by=['Message Count', 'Member Name'], ascending=[False, True])

def create_member_timeline(stats):
    """Create a timeline of group member count over time."""
    events = []
    for member, status in stats['member_status'].items():
        if status['first_seen']:
            events.append({'date': status['first_seen'], 'change': 1})
        if status['last_left']:
            events.append({'date': status['last_left'], 'change': -1})
    if not events:
        return pd.DataFrame()
    events.sort(key=lambda x: x['date'])
    date_range = pd.date_range(start=events[0]['date'].date(), end=events[-1]['date'].date() + timedelta(days=1), freq='D')
    member_count = 0
    daily_counts = []
    for date in date_range:
        for event in [e for e in events if e['date'].date() == date.date()]:
            member_count += event['change']
        daily_counts.append({'Date': date, 'Member Count': member_count})
    return pd.DataFrame(daily_counts)

def create_weekly_breakdown(stats):
    """
    Create a weekly message breakdown.
    "Week Duration" is based solely on the actual message dates from the export.
    """
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
    df = df.set_index('timestamp').sort_index()
    weekly_data = []
    week_number = 1
    for week_period, week_df in df.groupby(pd.Grouper(freq='W-MON')):
        if week_df.empty:
            continue
        # Derive actual boundaries from messages in this group:
        week_first_msg = week_df.index.min()
        week_last_msg = week_df.index.max()
        if week_first_msg.date() == week_last_msg.date():
            week_duration_str = week_first_msg.strftime('%d %b %Y')
        else:
            week_duration_str = f"{week_first_msg.strftime('%d %b %Y')} - {week_last_msg.strftime('%d %b %Y')}"
        week_messages = week_df.groupby('user').size().to_dict()
        current_members = set()
        left_members = set()
        for member, status in stats['member_status'].items():
            if status['first_seen'] <= week_last_msg:
                current_members.add(member)
            if status['last_left'] and week_first_msg <= status['last_left'] <= week_last_msg:
                current_members.discard(member)
                left_members.add(member)
        members_to_report = sorted(set(list(week_messages.keys()) + list(current_members)))
        for member in members_to_report:
            messages_sent = week_messages.get(member, 0)
            weekly_data.append({
                'Week': f'Week {week_number}',
                'Week Duration': week_duration_str,
                'Member Name': member,
                'Messages Sent': messages_sent,
                'Total Members': len(current_members),
                'Left Members': len(left_members),
                'Current Members': len(current_members) - len(left_members)
            })
        week_number += 1
    return pd.DataFrame(weekly_data)

def fetch_stats(stats, df):
    """Compute top-level statistics from the chat data."""
    # Total messages from parsed data
    total_messages = len(stats['messages_data'])
    # Total words
    total_words = sum(len(message.split()) for message in df['message'])
    # Count media messages: common WhatsApp export text "<Media omitted>" (adjust if needed)
    media_messages = sum(1 for message in df['message'] if "<Media omitted>" in message)
    # Count links using a simple regex (matches http/https)
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
        
        # Build a DataFrame from messages_data for additional stats and visualizations
        df = pd.DataFrame(stats['messages_data'])
        if df.empty:
            st.error("No messages found.")
            return
        
        # Extract unique users (filter out notifications if any)
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
            
            # Member Count Timeline (using our existing timeline)
            st.markdown("### Group Member Count Over Time")
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
            
            # Member Activity Table
            st.markdown("### Member Activity Status")
            activity_df = create_member_activity_table(stats)
            st.dataframe(activity_df, use_container_width=True)
            
            # Weekly Breakdown Table
            st.markdown("### Weekly Message & Member Analysis")
            weekly_df = create_weekly_breakdown(stats)
            st.dataframe(weekly_df)
            
            # Message Distribution Chart
            st.markdown("### Message Distribution")
            message_df = pd.DataFrame(list(stats['user_messages'].items()), columns=['Member', 'Messages'])
            fig = px.bar(message_df, x='Member', y='Messages', title='Messages per Member', color='Messages')
            st.plotly_chart(fig, use_container_width=True)
            
            # Busiest Users (for overall analysis)
            if selected_user == "Overall":
                st.markdown("### Most Busy Users")
                busy_users = pd.Series(stats['user_messages']).sort_values(ascending=False)
                fig, ax = plt.subplots()
                colors = plt.cm.viridis(np.linspace(0, 1, len(busy_users)))
                ax.bar(busy_users.index, busy_users.values, color=colors)
                plt.xticks(rotation=90)
                plt.title("User Activity")
                st.pyplot(fig)
            
            # Word Cloud
            st.markdown("### Word Cloud for Frequent Words")
            wc = create_wordcloud(selected_user, df)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            
            # LLM Chat Summary (optional)
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
