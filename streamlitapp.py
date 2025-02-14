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
    """Parse WhatsApp chat log file with enhanced exit event tracking."""
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
    exit_events = []  # Track all exit events
    exit_counts = Counter()  # Track count of exits per user

    # Patterns remain the same
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
                exit_counts[user] += 1  # Increment exit count for user
                if user in member_status:
                    member_status[user]['last_left'] = timestamp
                    exit_events.append({  # Track exit event with full details
                        'user': user,
                        'timestamp': timestamp,
                        'exit_count': exit_counts[user]  # Include the running count
                    })
            except Exception:
                continue

    current_members = sum(1 for m in member_status.values() if not m['last_left'])
    left_members = sum(1 for m in member_status.values() if m['last_left'])

    return {
        'messages_data': messages_data,
        'user_messages': user_messages,
        'member_status': member_status,
        'total_members': len(member_status),
        'current_members': current_members,
        'left_members': left_members,
        'exit_events': exit_events,
        'exit_counts': dict(exit_counts)  # Include exit counts in return data
    }

def create_member_timeline(stats):
    """Create timeline with explicit exit events."""
    events = []
    
    # Add join events
    for member, status in stats['member_status'].items():
        if status['first_seen']:
            events.append({
                'date': status['first_seen'],
                'change': 1,
                'event_type': 'join',
                'member': member
            })
        if status['last_left']:
            events.append({
                'date': status['last_left'],
                'change': -1,
                'event_type': 'left',
                'member': member
            })
    
    if not events:
        return pd.DataFrame()
        
    events.sort(key=lambda x: x['date'])
    
    # Create timeline with running totals
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

[... rest of the functions remain the same until main() ...]

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
            
            # Enhanced Exit Events Summary
            st.markdown("### Member Exit Analysis")
            if stats['exit_events']:
                # Timeline of exit events
                exit_df = pd.DataFrame(stats['exit_events'])
                exit_df['timestamp'] = pd.to_datetime(exit_df['timestamp'])
                exit_df = exit_df.sort_values('timestamp')
                
                # Exit counts per user
                exit_counts_df = pd.DataFrame(list(stats['exit_counts'].items()), 
                                           columns=['User', 'Number of Exits'])
                exit_counts_df = exit_counts_df.sort_values('Number of Exits', 
                                                          ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Exit Events Timeline")
                    st.dataframe(exit_df.assign(
                        exit_date=exit_df['timestamp'].dt.strftime('%d %b %Y')
                    )[['user', 'exit_date', 'exit_count']])
                
                with col2:
                    st.markdown("#### Exit Frequency by User")
                    st.dataframe(exit_counts_df)
                
                st.metric("Total Members Left", stats['left_members'])
                
                # Visualization of exit patterns
                st.markdown("#### Exit Patterns Over Time")
                fig = px.scatter(exit_df, 
                               x='timestamp', 
                               y='user',
                               size='exit_count',
                               title='Exit Events Distribution')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No exit events recorded")
            
            [... rest of the main function remains the same ...]

if __name__ == "__main__":
    main()
