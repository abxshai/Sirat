import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from groq import Groq
from dateutil import parser as date_parser

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
            model="llama3-70b-8192",
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
    """Parse WhatsApp chat log file with precise date parsing."""
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
    join_events = defaultdict(list)
    left_events = defaultdict(list)

    # Improved regex pattern with explicit date component capture
    date_pattern = r'''
        \[?
        (\d{1,2})[\/\.-]      # Day
        (\d{1,2})[\/\.-]      # Month
        (\d{2,4}),\s+         # Year
        (\d{1,2}):(\d{2})     # Hour:Minute
        (?::(\d{2}))?         # Optional seconds
        \s*([APap][Mm])       # AM/PM
        \]?
    '''

    message_re = re.compile(
        rf'^{date_pattern}\s*-?\s*(.*?):\s(.*)$',
        re.VERBOSE
    )

    join_re = re.compile(
        rf'^{date_pattern}\s*-?\s*(.*?) joined',
        re.VERBOSE
    )

    left_re = re.compile(
        rf'^{date_pattern}\s*-?\s*(.*?) left',
        re.VERBOSE
    )

    def parse_datetime(match):
        """Parse datetime components from regex match."""
        day, month, year, hour, minute, sec, ampm = match.groups()
        year = int(year) if len(year) == 4 else int(f"20{year}")
        hour = int(hour)
        if ampm.lower() == 'pm' and hour < 12:
            hour += 12
        return datetime(
            year=year,
            month=int(month),
            day=int(day),
            hour=hour,
            minute=int(minute),
            second=int(sec) if sec else 0
        )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Parse messages
        msg_match = message_re.match(line)
        if msg_match:
            try:
                timestamp = parse_datetime(msg_match)
                user = clean_member_name(msg_match.group(8))
                message = msg_match.group(9)
                messages_data.append({
                    "timestamp": timestamp,
                    "user": user,
                    "message": message
                })
                user_messages[user] += 1
                if user not in member_status:
                    member_status[user] = {
                        'first_seen': timestamp,
                        'last_left': None,
                        'exit_count': 0
                    }
            except Exception as e:
                continue
            continue

        # Parse join/leave events
        for pattern, event_type in [(join_re, "join"), (left_re, "left")]:
            match = pattern.match(line)
            if match:
                try:
                    timestamp = parse_datetime(match)
                    user = clean_member_name(match.group(8))
                    if event_type == "join":
                        join_events[user].append(timestamp)
                        if user not in member_status:
                            member_status[user] = {
                                'first_seen': timestamp,
                                'last_left': None,
                                'exit_count': 0
                            }
                    else:
                        left_events[user].append(timestamp)
                        member_status.setdefault(user, {
                            'first_seen': timestamp,
                            'last_left': None,
                            'exit_count': 0
                        })['exit_count'] += 1
                        member_status[user]['last_left'] = timestamp
                except Exception as e:
                    continue
                break

    current_members = sum(1 for m in member_status.values() if not m['last_left'])

    return {
        'messages_data': messages_data,
        'user_messages': user_messages,
        'member_status': member_status,
        'total_members': len(member_status),
        'current_members': current_members,
        'left_events': left_events
    }

def create_member_activity_table(stats):
    """Create activity status table with exit events."""
    activity_data = []
    for member, status in stats['member_status'].items():
        message_count = stats['user_messages'].get(member, 0)
        exit_count = status.get('exit_count', 0)
        activity_data.append({
            'Member Name': member,
            'Message Count': message_count,
            'Exit Events': exit_count,
            'Activity Status': 'Active' if message_count > 0 else 'Inactive',
            'Join Date': status['first_seen'].strftime('%d %b %Y'),
            'Left Date': status['last_left'].strftime('%d %b %Y') if status['last_left'] else 'Present',
            'Current Status': 'Left' if status['last_left'] else 'Present'
        })
    return pd.DataFrame(activity_data).sort_values(
        by=['Message Count', 'Member Name'],
        ascending=[False, True]
    )

def create_member_timeline(stats):
    """Create member count timeline with accurate date parsing."""
    events = []
    for member, status in stats['member_status'].items():
        events.append({
            'date': status['first_seen'],
            'change': 1,
            'member': member,
            'type': 'join'
        })
        if status['last_left']:
            events.append({
                'date': status['last_left'],
                'change': -1,
                'member': member,
                'type': 'left'
            })
    if not events:
        return pd.DataFrame()
    events.sort(key=lambda x: x['date'])
    timeline = []
    current_count = 0
    current_date = events[0]['date'].date()
    for event in events:
        event_date = event['date'].date()
        # Fill gaps between dates
        while current_date < event_date:
            timeline.append({
                'Date': datetime.combine(current_date, datetime.min.time()),
                'Member Count': current_count
            })
            current_date += timedelta(days=1)
        # Update count for event date
        current_count += event['change']
        timeline.append({
            'Date': event['date'],
            'Member Count': current_count,
            'Event Type': event['type'],
            'Member': event['member']
        })
        current_date = event_date
    return pd.DataFrame(timeline)

def create_weekly_breakdown(stats):
    """Create weekly breakdown with accurate date ranges."""
    if not stats['messages_data']:
        return pd.DataFrame()
    df = pd.DataFrame(stats['messages_data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    if df.empty:
        return pd.DataFrame()
    # Create proper week grouping
    df['week_start'] = df['timestamp'].dt.to_period('W').dt.start_time
    weekly_data = df.groupby(['week_start', 'user']).agg(
        Messages=('timestamp', 'count'),
        First_Date=('timestamp', 'min'),
        Last_Date=('timestamp', 'max')
    ).reset_index()
    weekly_data['Week Duration'] = weekly_data.apply(
        lambda x: f"{x['First_Date'].strftime('%d %b')} - {x['Last_Date'].strftime('%d %b %Y')}",
        axis=1
    )
    # Format final output
    weekly_data = weekly_data.rename(columns={
        'user': 'Member Name',
        'Messages': 'Messages Sent',
        'week_start': 'Week Start'
    })
    weekly_data['Week'] = 'Week ' + (weekly_data.groupby('Week Start').ngroup() + 1).astype(str)
    return weekly_data[['Week', 'Week Duration', 'Member Name', 'Messages Sent']]

def main():
    st.title("Enhanced Chat Log Analyzer")
    uploaded_file = st.file_uploader("Upload WhatsApp chat log (TXT format)", type="txt")
    if uploaded_file:
        with st.spinner("Parsing chat log..."):
            stats = parse_chat_log_file(uploaded_file)
        if stats:
            st.success(
                f"Chat log parsed successfully!\n- Total Members Ever: {stats['total_members']}\n- Currently Active Members: {stats['current_members']}"
            )
            # Member Timeline Chart
            st.markdown("### Group Member Timeline")
            timeline_df = create_member_timeline(stats)
            if not timeline_df.empty:
                fig = px.area(
                    timeline_df,
                    x='Date',
                    y='Member Count',
                    hover_data=['Member', 'Event Type'],
                    title='Member Participation Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)
            # Enhanced Activity Table
            st.markdown("### Member Activity Overview")
            activity_df = create_member_activity_table(stats)
            st.dataframe(
                activity_df.style.format({
                    'Join Date': lambda x: x,
                    'Left Date': lambda x: x
                }),
                use_container_width=True
            )
            # Exit Events Analysis
            st.markdown("### Exit Events Summary")
            exit_df = activity_df[['Member Name', 'Exit Events', 'Left Date']].sort_values('Exit Events', ascending=False)
            st.dataframe(exit_df, use_container_width=True)
            # Weekly Breakdown
            st.markdown("### Weekly Engagement Analysis")
            weekly_df = create_weekly_breakdown(stats)
            if not weekly_df.empty:
                fig = px.bar(
                    weekly_df,
                    x='Week Duration',
                    y='Messages Sent',
                    color='Member Name',
                    barmode='stack',
                    title='Weekly Message Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(weekly_df, use_container_width=True)
            # LLM Summary Section
            st.markdown("### AI-Powered Insights")
            if st.button("Generate Smart Summary"):
                client = initialize_llm_client()
                prompt = f"Analyze this chat log with {stats['total_members']} members. "
                prompt += f"Key stats: {stats['current_members']} current members, "
                prompt += f"Top contributors: {dict(Counter(stats['user_messages']).most_common(5))}"
                word_placeholder = st.empty()
                get_llm_reply(client, prompt, word_placeholder)

if __name__ == "__main__":
    main()
