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
    """Parse WhatsApp chat log file and count exit events based on the 'left' keyword."""
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
    exit_events = []  # List to track each exit event

    # Regular expression patterns
    message_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?):\s(.*)$'
    )
    join_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?) joined'
    )
    # We'll detect exit events by searching for the keyword "left"
    left_pattern = re.compile(
        r'(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?).*?(.*?)\s+left'
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
                    member_status[user] = {'first_seen': timestamp}
            except Exception:
                continue
            continue

        join_match = join_pattern.match(line)
        left_match = left_pattern.search(line)  # using search to find the keyword "left"

        if join_match:
            timestamp_str, user = join_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                user = clean_member_name(user)
                if user not in member_status:
                    member_status[user] = {'first_seen': timestamp}
            except Exception:
                continue
        elif left_match:
            timestamp_str, user = left_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                user = clean_member_name(user)
                # Initialize user's record if not present.
                if user not in member_status:
                    member_status[user] = {'first_seen': timestamp}
                # Instead of a single 'last_left', we now store a list of exit events.
                if 'left_times' not in member_status[user]:
                    member_status[user]['left_times'] = []
                member_status[user]['left_times'].append(timestamp)
                exit_events.append({  # Track this exit event
                    'user': user,
                    'timestamp': timestamp
                })
            except Exception:
                continue

    total_members = len(member_status)
    # Unique members that have left at least once.
    left_members = sum(1 for m in member_status.values() if m.get('left_times'))
    # For current members, assume that if a member has left at least once, they are not currently in the group.
    current_members = total_members - left_members

    return {
        'messages_data': messages_data,
        'user_messages': user_messages,
        'member_status': member_status,
        'total_members': total_members,
        'current_members': current_members,
        'left_members': left_members,
        'exit_events': exit_events
    }

def create_member_timeline(stats):
    """Create timeline with explicit join and exit events (each exit event subtracts one member)."""
    events = []
    
    # Add join events and each exit event from the member_status.
    for member, status in stats['member_status'].items():
        if status.get('first_seen'):
            events.append({
                'date': status['first_seen'],
                'change': 1,
                'event_type': 'join',
                'member': member
            })
        # Add each exit event if available.
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

    # Group by user and get message counts
    user_msgs = df.groupby('user').size().reset_index(name='Messages Sent')
    
    # Add member status info
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
# Member Activity Table
##############################

def create_member_activity_table(stats):
    """Create activity status table with exit event counts."""
    activity_data = []
    for member, status in stats['member_status'].items():
        # Skip system notifications if present
        if member == "group_notification":
            continue
        message_count = stats['user_messages'].get(member, 0)
        # Count the exit events from left_times if available.
        exit_count = len(status.get('left_times', []))
        activity_data.append({
            'Member Name': member,
            'Message Count': message_count,
            'Exit Events': exit_count,
            'Activity Status': 'Active' if message_count > 0 and not status.get('left_times') else 'Inactive',
            'Join Date': status['first_seen'].strftime('%d %b %Y'),
            'Left Date': (min(status['left_times']).strftime('%d %b %Y') if status.get('left_times') else 'Present'),
            'Current Status': 'Left' if status.get('left_times') else 'Present'
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
            
            # Exit Events Summary
            st.markdown("### Member Exit Summary")
            if stats['exit_events']:
                exit_df = pd.DataFrame(stats['exit_events'])
                exit_df['timestamp'] = pd.to_datetime(exit_df['timestamp'])
                exit_df = exit_df.sort_values('timestamp')
                
                # Count exit events per user
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
            
            # Group Member Timeline
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
            
            # Busiest Users (Overall)
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
