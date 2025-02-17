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
    """Parse date string with error handling using dayfirst=True."""
    try:
        return date_parser.parse(date_str, fuzzy=False, dayfirst=True)
    except Exception:
        return None

def process_chunk(chunk, patterns):
    """
    Process a chunk of lines from the chat log.
    First try matching the general message pattern.
    If a match is found, check whether the message text (after cleaning)
    exactly equals the sender’s name followed by " left". If so, record as an exit event.
    Otherwise, record as a regular message.
    Then process join events and any remaining left events.
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
    Parse WhatsApp chat log file.
    Reads the file, splits into chunks, applies regex patterns,
    and returns messages, join events, exit events, and member status.
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
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?)\s*\bleft\b\s*$'
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

    # Combine join events and messages, sorted by timestamp,
    # to ensure that each member's "first_seen" is truly the earliest event.
    combined_events = sorted(all_joins + all_messages, key=lambda x: x['timestamp'])
    member_status = {}
    for event in combined_events:
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
    
    # Compute permanent exit info:
    # Build a timeline DataFrame; then for each member, take the last event.
    timeline = create_member_timeline({'member_status': member_status, 'join_events': all_joins, 'messages_data': all_messages, 'exit_events': all_exits})
    if not timeline.empty:
        last_events = timeline.sort_values('Date').groupby('Member', as_index=False).last()
        permanent_left_users = set(last_events[last_events['Event Type'] == 'left']['Member'])
    else:
        permanent_left_users = set()
    left_members = len(permanent_left_users)
    total_members = len(member_status)
    current_members = total_members - left_members

    return {
        'messages_data': all_messages,
        'user_messages': Counter(msg['user'] for msg in all_messages),
        'member_status': member_status,
        'total_members': total_members,
        'current_members': current_members,
        'left_members': left_members,
        'exit_events': all_exits,
        'join_events': all_joins
    }

def create_member_timeline(stats):
    """
    Create a timeline DataFrame of join and left events.
    Adds a 'Member' column to indicate the user.
    """
    events = []
    for member, status in stats['member_status'].items():
        if status.get('first_seen'):
            events.append({
                'Date': status['first_seen'],
                'change': 1,
                'Event Type': 'join',
                'Member': member
            })
        if 'left_times' in status:
            for lt in status['left_times']:
                events.append({
                    'Date': lt,
                    'change': -1,
                    'Event Type': 'left',
                    'Member': member
                })
    if not events:
        return pd.DataFrame()
    events.sort(key=lambda x: x['Date'])
    timeline_data = []
    member_count = 0
    for event in events:
        member_count += event['change']
        timeline_data.append({
            'Date': event['Date'],
            'Member Count': member_count,
            'Event': f"{event['Member']} {event['Event Type']}",
            'Event Type': event['Event Type'],
            'Member': event['Member']
        })
    return pd.DataFrame(timeline_data)

def create_exit_events_table(stats):
    """
    Create a table for permanent exit events with two columns:
    | Name of Exit Person | Exit Date & Time (from the txt file) |
    Determined from the timeline: only members whose last event is 'left' are included.
    """
    timeline_df = create_member_timeline(stats)
    if timeline_df.empty:
        return pd.DataFrame()
    last_events = timeline_df.sort_values('Date').groupby('Member', as_index=False).last()
    left_df = last_events[last_events['Event Type'] == 'left']
    left_df = left_df.rename(columns={'Member': 'Name of Exit Person', 'Date': 'Exit Date & Time (from the txt file)'})
    return left_df[['Name of Exit Person', 'Exit Date & Time (from the txt file)']]

def create_member_activity_table(stats):
    """Create an overall Member Activity Analysis table."""
    activity_data = []
    exit_table = create_exit_events_table(stats)
    permanent_left = exit_table['Name of Exit Person'].tolist() if not exit_table.empty else []
    for member, status in stats['member_status'].items():
        current_status = 'Left' if member in permanent_left else 'Active'
        exit_count = len(status.get('left_times', []))
        activity_data.append({
            'Member Name': member,
            'Overall Message Count': stats['user_messages'].get(member, 0),
            'Exit Events': exit_count,
            'Activity Status': current_status,
            'Join Date': status['first_seen'].strftime('%d %b %Y'),
            'Last Exit Date': (status['left_times'][-1].strftime('%d %b %Y')
                               if status.get('left_times') else 'Present')
        })
    df = pd.DataFrame(activity_data)
    if not df.empty:
        df = df.sort_values(by=['Overall Message Count', 'Member Name'], ascending=[False, True])
    return df

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
        
        if st.sidebar.button("Show Analysis"):
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
                st.metric("Total Members Left", len(exit_df))
            else:
                st.write("No exit events recorded")
            
            st.subheader("Member Activity Analysis")
            activity_df = create_member_activity_table(stats)
            if not activity_df.empty:
                st.dataframe(activity_df)
            
            st.subheader("Word Cloud (Overall)")
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
