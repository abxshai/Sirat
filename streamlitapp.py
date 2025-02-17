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
from datetime import datetime
import io
import concurrent.futures
import math

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

def parse_date(date_str):
    """Parse date string with error handling."""
    try:
        return date_parser.parse(date_str, fuzzy=False, dayfirst=True)
    except Exception:
        return None

def process_chunk(chunk, patterns):
    """
    Process a chunk of lines from the chat log.
    First, try matching the message pattern.
    If a match is found, check if the message text (cleaned of invisible characters)
    exactly equals the sender’s name + " left" (case-insensitive).
    If so, record this as an exit event; otherwise, as a regular message.
    Then check for join events and, as fallback, try strict/general left patterns.
    """
    messages = []
    joins = []
    exits = []
    
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue

        # Try message pattern first.
        m = patterns['message'].match(line)
        if m:
            timestamp_str, user, message = m.groups()
            date = parse_date(timestamp_str)
            if date:
                msg_clean = message.strip().replace("\u200e", "").strip()
                if msg_clean.lower() == clean_member_name(user).lower() + " left":
                    exits.append({
                        'timestamp': date,
                        'timestamp_str': timestamp_str,
                        'user': clean_member_name(user)
                    })
                else:
                    messages.append({
                        'timestamp': date,
                        'timestamp_str': timestamp_str,
                        'user': clean_member_name(user),
                        'message': message
                    })
            continue

        # Try join pattern.
        j = patterns['join'].match(line)
        if j:
            timestamp_str, user = j.groups()
            date = parse_date(timestamp_str)
            if date:
                joins.append({
                    'timestamp': date,
                    'timestamp_str': timestamp_str,
                    'user': clean_member_name(user)
                })
            continue

        # Try strict left pattern.
        sl = patterns['strict_left'].match(line)
        if sl:
            raw_date_str, user, left_msg = sl.groups()
            date = parse_date(raw_date_str)
            if date:
                exits.append({
                    'timestamp': date,
                    'timestamp_str': raw_date_str,
                    'user': clean_member_name(user)
                })
            continue

        # Finally, try general left pattern.
        l = patterns['left'].search(line)
        if l:
            timestamp_str, user = l.groups()
            date = parse_date(timestamp_str)
            if date:
                exits.append({
                    'timestamp': date,
                    'timestamp_str': timestamp_str,
                    'user': clean_member_name(user)
                })
            continue

    return messages, joins, exits

def parse_chat_log_file(uploaded_file, lines_per_chunk=1000):
    """
    Parse WhatsApp chat log file with improved performance for large files.
    Reads the file as text, splits it into complete lines, and groups lines into chunks.
    Processes chunks in parallel using regex patterns for messages, joins, and left events.
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

    lines = text.splitlines()
    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk = "\n".join(lines[i:i+lines_per_chunk])
        chunks.append(chunk)
    
    patterns = {
        'message': re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?):\s(.*)$'
        ),
        'join': re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?) joined'
        ),
        'strict_left': re.compile(
            r'^\[(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*[APap][Mm])\]\s*([^:]+):\s*(?:\u200e)?(.*?)\s+left\s*$'
        ),
        'left': re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?) left'
        )
    }
    
    all_messages = []
    all_joins = []
    all_exits = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chunk, patterns) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            messages, joins, exits = future.result()
            all_messages.extend(messages)
            all_joins.extend(joins)
            all_exits.extend(exits)
    
    all_messages.sort(key=lambda x: x['timestamp'])
    all_joins.sort(key=lambda x: x['timestamp'])
    all_exits.sort(key=lambda x: x['timestamp'])
    
    user_messages = Counter(msg['user'] for msg in all_messages)
    member_status = {}
    for event in all_joins + all_messages:
        user = event['user']
        if user not in member_status:
            member_status[user] = {
                'first_seen': event['timestamp'],
                'first_seen_str': event['timestamp_str']
            }
    for exit_event in all_exits:
        user = exit_event['user']
        if user not in member_status:
            member_status[user] = {
                'first_seen': exit_event['timestamp'],
                'first_seen_str': exit_event['timestamp_str']
            }
        if 'left_times' not in member_status[user]:
            member_status[user]['left_times'] = []
            member_status[user]['left_times_str'] = []
        member_status[user]['left_times'].append(exit_event['timestamp'])
        member_status[user]['left_times_str'].append(exit_event['timestamp_str'])
    
    total_members = len(member_status)
    left_members = sum(1 for m in member_status.values() if m.get('left_times'))
    current_members = total_members - left_members
    
    return {
        'messages_data': all_messages,
        'user_messages': user_messages,
        'member_status': member_status,
        'total_members': total_members,
        'current_members': current_members,
        'left_members': left_members,
        'exit_events': all_exits
    }

def create_exit_events_table(stats):
    """
    Create a separate table for exit events with two columns:
    | Name of Exit Person | Exit Date & Time (exactly from the txt file) |
    Filters out exit events for users who sent a message after leaving.
    """
    exit_events = stats.get('exit_events', [])
    if not exit_events:
        return pd.DataFrame()
    # Build max message timestamp per user.
    user_max = {}
    for msg in stats['messages_data']:
        user = msg['user']
        ts = msg['timestamp']
        if user in user_max:
            if ts > user_max[user]:
                user_max[user] = ts
        else:
            user_max[user] = ts
    filtered = []
    for event in exit_events:
        user = event['user']
        if user not in user_max or user_max[user] <= event['timestamp']:
            filtered.append(event)
    df = pd.DataFrame(filtered)
    df = df.rename(columns={
        'user': 'Name of Exit Person',
        'timestamp_str': 'Exit Date & Time (exactly from the txt file)'
    })
    return df[['Name of Exit Person', 'Exit Date & Time (exactly from the txt file)']]

def create_member_timeline(stats):
    """Create a timeline showing join and left events with running totals."""
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

def create_weekly_activity_table(stats):
    """
    Create a weekly activity summary table.
    For each 7-day interval (starting from the earliest message timestamp),
    this table shows:
      - Week Start (the raw timestamp of the first message in that interval)
      - Total Messages in that week
      - Top Messenger and their message count
      - Join events in that week (using the raw 'first_seen_str' from member_status)
      - Exit events in that week (using the raw timestamp strings, after filtering out rejoin cases)
    """
    messages = stats['messages_data']
    if not messages:
        return pd.DataFrame()
    min_time = min(msg['timestamp'] for msg in messages)
    max_time = max(msg['timestamp'] for msg in messages)
    # Ensure the final (partial) week is included.
    total_weeks = math.floor((max_time - min_time).total_seconds() / (7 * 86400)) + 1
    
    # Precompute max message timestamp per user.
    user_max = {}
    for msg in messages:
        user = msg['user']
        ts = msg['timestamp']
        if user in user_max:
            if ts > user_max[user]:
                user_max[user] = ts
        else:
            user_max[user] = ts
    
    weekly_summary = []
    for i in range(total_weeks):
        group_start = min_time + pd.Timedelta(days=7 * i)
        group_end = group_start + pd.Timedelta(days=7)
        week_msgs = [msg for msg in messages if group_start <= msg['timestamp'] < group_end]
        if not week_msgs:
            continue
        total_msgs = len(week_msgs)
        week_start_raw = min(week_msgs, key=lambda x: x['timestamp'])['timestamp_str']
        counter = Counter(msg['user'] for msg in week_msgs)
        top_user, top_count = counter.most_common(1)[0]
        
        joins = []
        for user, status in stats['member_status'].items():
            ts = status['first_seen']
            if group_start <= ts < group_end:
                joins.append(f"{user}: {status['first_seen_str']}")
        
        exits = []
        for event in stats['exit_events']:
            ts = event['timestamp']
            user = event['user']
            if group_start <= ts < group_end:
                if user not in user_max or user_max[user] <= ts:
                    exits.append(f"{user}: {event['timestamp_str']}")
        
        weekly_summary.append({
            "Week Start": week_start_raw,
            "Total Messages": total_msgs,
            "Top Messenger": top_user,
            "Top Messenger Count": top_count,
            "Joins": ", ".join(joins),
            "Exits": ", ".join(exits)
        })
    return pd.DataFrame(weekly_summary)

def create_wordcloud(df):
    """Generate an optimized word cloud image from overall messages."""
    messages = [msg['message'] for msg in df]
    text = " ".join(messages)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        collocations=False
    ).generate(text)
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
        df = pd.DataFrame(stats['messages_data'])
        if df.empty:
            st.error("No messages found.")
            return
        
        # Overall analysis only.
        st.title("Chat Analysis Results")
        total_messages = len(stats['messages_data'])
        total_words = sum(len(msg['message'].split()) for msg in stats['messages_data'])
        media_messages = sum(1 for msg in stats['messages_data'] if "<Media omitted>" in msg['message'])
        links_shared = sum(1 for msg in stats['messages_data'] if "http" in msg['message'].lower())
        
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("Total Messages", total_messages),
            ("Total Words", total_words),
            ("Media Shared", media_messages),
            ("Links Shared", links_shared)
        ]
        for col, (title, value) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.metric(title, value)
        
        st.subheader("Member Exit Analysis")
        exit_df = create_exit_events_table(stats)
        if not exit_df.empty:
            st.dataframe(exit_df)
            st.metric("Total Members Left", stats['left_members'])
        else:
            st.write("No exit events recorded")
        
        st.subheader("Member Timeline")
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
                title='Group Member Count Over Time',
                xaxis_title='Date',
                yaxis_title='Number of Members',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Weekly Activity Summary")
        weekly_df = create_weekly_activity_table(stats)
        if not weekly_df.empty:
            st.dataframe(weekly_df)
        
        st.subheader("Message Distribution")
        message_df = pd.DataFrame(list(stats['user_messages'].items()), columns=['Member', 'Messages'])
        message_df = message_df.sort_values('Messages', ascending=False)
        fig = px.bar(
            message_df,
            x='Member',
            y='Messages',
            title='Messages per Member',
            color='Messages',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title="Member",
            yaxis_title="Number of Messages",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Word Cloud")
        try:
            wc = create_wordcloud(stats['messages_data'])
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating word cloud: {str(e)}")
        
        st.subheader("Chat Analysis Summary (LLM)")
        if st.button("Generate LLM Summary"):
            try:
                client = initialize_llm_client()
                top_contributors = dict(Counter(stats['user_messages']).most_common(5))
                prompt = (
                    f"Analyze this WhatsApp chat log summary:\n"
                    f"- Total members: {stats['total_members']}\n"
                    f"- Currently active members: {stats['current_members']}\n"
                    f"- Members who left: {stats['left_members']}\n"
                    f"- Total messages: {total_messages}\n"
                    f"- Top 5 contributors: {top_contributors}\n"
                    f"- Media messages shared: {media_messages}\n"
                    f"- Links shared: {links_shared}\n\n"
                    "Provide insights about group dynamics, engagement patterns, and member participation."
                )
                placeholder = st.empty()
                get_llm_reply(client, prompt, placeholder)
            except Exception as e:
                st.error(f"Error generating LLM analysis: {str(e)}")
                    
if __name__ == "__main__":
    st.set_page_config(
        page_title="WhatsApp Chat Analyzer",
        page_icon="💬",
        layout="wide"
    )
    main()
