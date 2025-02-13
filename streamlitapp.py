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

    # Initialize data structures
    messages_data = []
    user_messages = Counter()
    member_status = {}
    join_events = defaultdict(list)
    left_events = defaultdict(list)

    # Compile regex patterns
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

        # Try to match message first
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
                    member_status[user] = {
                        'first_seen': timestamp,
                        'last_left': None
                    }
            except Exception:
                continue
            continue

        # Check for join/left events
        join_match = join_pattern.match(line)
        left_match = left_pattern.match(line)

        if join_match:
            timestamp_str, user = join_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                user = clean_member_name(user)
                join_events[user].append(timestamp)
                if user not in member_status:
                    member_status[user] = {
                        'first_seen': timestamp,
                        'last_left': None
                    }
            except Exception:
                continue

        elif left_match:
            timestamp_str, user = left_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                user = clean_member_name(user)
                left_events[user].append(timestamp)
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

def create_member_activity_table(stats):
    """Create activity status table."""
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
    
    return pd.DataFrame(activity_data).sort_values(
        by=['Message Count', 'Member Name'], 
        ascending=[False, True]
    )

def create_member_timeline(stats):
    """Create member count timeline."""
    events = []
    
    for member, status in stats['member_status'].items():
        if status['first_seen']:
            events.append({
                'date': status['first_seen'],
                'change': 1,
                'member': member
            })
        if status['last_left']:
            events.append({
                'date': status['last_left'],
                'change': -1,
                'member': member
            })
    
    if not events:
        return pd.DataFrame()
        
    events.sort(key=lambda x: x['date'])
    date_range = pd.date_range(
        start=events[0]['date'].date(),
        end=events[-1]['date'].date() + timedelta(days=1),
        freq='D'
    )
    
    member_count = 0
    daily_counts = []
    
    for date in date_range:
        day_events = [e for e in events if e['date'].date() == date.date()]
        for event in day_events:
            member_count += event['change']
        
        daily_counts.append({
            'Date': date,
            'Member Count': member_count
        })
    
    return pd.DataFrame(daily_counts)

def create_weekly_breakdown(stats):
    """Create weekly message breakdown strictly within chat dates."""
    if not stats['messages_data']:
        return pd.DataFrame()

    df = pd.DataFrame(stats['messages_data'])
    
    # Sort messages by timestamp
    df = df.sort_values('timestamp')
    
    # Create a 'date' column without time information
    df['date'] = df['timestamp'].dt.date
    
    # Group messages by date to find dates with actual messages
    dates_with_messages = df['date'].unique()
    
    # Initialize variables
    weekly_data = []
    current_members = set()
    left_members = set()
    week_number = 1
    
    # Process each week based on actual message dates
    current_week_start = None
    current_week_messages = []
    
    for date in dates_with_messages:
        # Convert date to datetime for comparison
        current_date = datetime.combine(date, datetime.min.time())
        
        # Start new week if needed
        if current_week_start is None:
            current_week_start = current_date
            current_week_messages = []
        elif (current_date - current_week_start).days >= 7:
            # Process completed week
            week_end = min(current_week_start + timedelta(days=6), datetime.combine(date, datetime.max.time()))
            
            # Update member status for this week
            for member, status in stats['member_status'].items():
                if status['first_seen'] <= week_end:
                    current_members.add(member)
                if status['last_left'] and current_week_start <= status['last_left'] <= week_end:
                    current_members.discard(member)
                    left_members.add(member)
            
            # Get messages for this week
            week_mask = (df['timestamp'] >= current_week_start) & (df['timestamp'] <= week_end)
            week_df = df[week_mask]
            week_messages = week_df.groupby('user').size().to_dict()
            
            # Create entries only if there are messages
            if not week_df.empty:
                # Find actual first and last message dates in this week
                week_first_msg = week_df['timestamp'].min()
                week_last_msg = week_df['timestamp'].max()
                
                # Create entry for each active member
                active_members_this_week = {user for user in current_members if week_messages.get(user, 0) > 0}
                
                # Add active members first
                for member in sorted(active_members_this_week):
                    weekly_data.append({
                        'Week': f'Week {week_number}',
                        'Week Duration': f"{week_first_msg.strftime('%d %b %Y')} - {week_last_msg.strftime('%d %b %Y')}",
                        'Member Name': member,
                        'Messages Sent': week_messages.get(member, 0),
                        'Total Members': len(current_members),
                        'Left Members': len(left_members),
                        'Current Members': len(current_members) - len(left_members)
                    })
                
                # Then add inactive members
                inactive_members = current_members - active_members_this_week
                for member in sorted(inactive_members):
                    weekly_data.append({
                        'Week': f'Week {week_number}',
                        'Week Duration': f"{week_first_msg.strftime('%d %b %Y')} - {week_last_msg.strftime('%d %b %Y')}",
                        'Member Name': member,
                        'Messages Sent': 0,
                        'Total Members': len(current_members),
                        'Left Members': len(left_members),
                        'Current Members': len(current_members) - len(left_members)
                    })
                
                week_number += 1
            
            # Start new week
            current_week_start = current_date
            current_week_messages = []
        
        current_week_messages.append(current_date)
    
    # Process the last week if it contains messages
    if current_week_messages:
        week_end = datetime.combine(current_week_messages[-1], datetime.max.time())
        
        # Update member status for the last week
        for member, status in stats['member_status'].items():
            if status['first_seen'] <= week_end:
                current_members.add(member)
            if status['last_left'] and current_week_start <= status['last_left'] <= week_end:
                current_members.discard(member)
                left_members.add(member)
        
        # Get messages for the last week
        week_mask = (df['timestamp'] >= current_week_start) & (df['timestamp'] <= week_end)
        week_df = df[week_mask]
        
        if not week_df.empty:
            # Find actual first and last message dates
            week_first_msg = week_df['timestamp'].min()
            week_last_msg = week_df['timestamp'].max()
            
            week_messages = week_df.groupby('user').size().to_dict()
            active_members_this_week = {user for user in current_members if week_messages.get(user, 0) > 0}
            
            # Add active members for last week
            for member in sorted(active_members_this_week):
                weekly_data.append({
                    'Week': f'Week {week_number}',
                    'Week Duration': f"{week_first_msg.strftime('%d %b %Y')} - {week_last_msg.strftime('%d %b %Y')}",
                    'Member Name': member,
                    'Messages Sent': week_messages.get(member, 0),
                    'Total Members': len(current_members),
                    'Left Members': len(left_members),
                    'Current Members': len(current_members) - len(left_members)
                })
            
            # Add inactive members for last week
            inactive_members = current_members - active_members_this_week
            for member in sorted(inactive_members):
                weekly_data.append({
                    'Week': f'Week {week_number}',
                    'Week Duration': f"{week_first_msg.strftime('%d %b %Y')} - {week_last_msg.strftime('%d %b %Y')}",
                    'Member Name': member,
                    'Messages Sent': 0,
                    'Total Members': len(current_members),
                    'Left Members': len(left_members),
                    'Current Members': len(current_members) - len(left_members)
                })
    
    return pd.DataFrame(weekly_data)

def main():
    st.title("Enhanced Chat Log Analyzer")
    
    uploaded_file = st.file_uploader("Upload WhatsApp chat log (TXT format)", type="txt")
    
    if uploaded_file:
        with st.spinner("Parsing chat log..."):
            stats = parse_chat_log_file(uploaded_file)
        
        if stats:
            st.success(f"""Chat log parsed successfully!
            - Total Members Ever: {stats['total_members']}
            - Currently Active Members: {stats['current_members']}""")
            
            # Member count timeline
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
                    hovermode='x unified',
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Member Activity Table
            st.markdown("### Member Activity Status")
            activity_df = create_member_activity_table(stats)
            st.dataframe(activity_df, use_container_width=True)
            
            # Activity statistics
            active_count = len(activity_df[activity_df['Activity Status'] == 'Active'])
            inactive_count = len(activity_df[activity_df['Activity Status'] == 'Inactive'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active Members", active_count)
            with col2:
                st.metric("Inactive Members", inactive_count)
            
            # Weekly breakdown
            st.markdown("### Weekly Message & Member Analysis")
            weekly_df = create_weekly_breakdown(stats)
            st.dataframe(weekly_df)
            
            # Message distribution
            st.markdown("### Message Distribution")
            message_df = pd.DataFrame(list(stats['user_messages'].items()), 
                                    columns=['Member', 'Messages'])
            fig = px.bar(message_df, x='Member', y='Messages',
                        title='Messages per Member',
                        color='Messages')
            st.plotly_chart(fig, use_container_width=True)
            
            # LLM Summary
            st.markdown("### Chat Analysis Summary")
            if st.button("Generate Summary"):
                client = initialize_llm_client()
                prompt = (f"Analyze this chat log:\n"
                         f"- Total members: {stats['total_members']}\n"
                         f"- Current members: {stats['current_members']}\n"
                         f"- Active members: {active_count}\n"
                         f"- Inactive members: {inactive_count}\n"
                         f"- Top contributors: {dict(Counter(stats['user_messages']).most_common(5))}\n")
                word_placeholder = st.empty()
                get_llm_reply(client, prompt, word_placeholder)

if __name__ == "__main__":
    main()
