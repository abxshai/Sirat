import streamlit as st
import pandas as pd
import plotly.express as px
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
    """
    Optimized parser for WhatsApp chat logs with improved accuracy and performance.
    """
    try:
        content = uploaded_file.read()
        text = content.decode("utf-8") if isinstance(content, bytes) else content
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None

    # Improved regex patterns
    message_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?):\s(.*)$'
    )
    join_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?) joined'
    )
    left_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?) left'
    )
    added_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?) added (.*?)$'
    )

    messages_data = []
    user_messages = Counter()
    join_events = defaultdict(list)
    left_events = defaultdict(list)
    added_events = defaultdict(list)
    
    current_msg = {"timestamp": None, "user": None, "message": None}
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Check for message pattern first (most common case)
        msg_match = message_pattern.match(line)
        if msg_match:
            if current_msg["message"]:
                messages_data.append(current_msg.copy())
                user_messages[current_msg["user"]] += 1
            
            timestamp_str, user, message = msg_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                current_msg = {
                    "timestamp": timestamp,
                    "user": clean_member_name(user),
                    "message": message
                }
            except Exception:
                continue
            continue

        # Check for join/left/added events
        join_match = join_pattern.match(line)
        left_match = left_pattern.match(line)
        added_match = added_pattern.match(line)

        if join_match:
            timestamp_str, user = join_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                join_events[clean_member_name(user)].append(timestamp)
            except Exception:
                continue
        elif left_match:
            timestamp_str, user = left_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                left_events[clean_member_name(user)].append(timestamp)
            except Exception:
                continue
        elif added_match:
            timestamp_str, adder, added = added_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                added_events[clean_member_name(added)].append(timestamp)
            except Exception:
                continue
        elif current_msg["message"]:
            current_msg["message"] += "\n" + line

    # Add final message if exists
    if current_msg["message"]:
        messages_data.append(current_msg.copy())
        user_messages[current_msg["user"]] += 1

    # Process member status
    member_status = {}
    for member in set(user_messages.keys()) | set(join_events.keys()) | set(added_events.keys()):
        join_dates = sorted(join_events.get(member, []) + added_events.get(member, []))
        left_dates = sorted(left_events.get(member, []))
        
        if join_dates:
            member_status[member] = {
                'first_seen': join_dates[0],
                'last_left': left_dates[-1] if left_dates else None
            }
        else:
            # For members who appear in messages but have no explicit join event
            first_msg = min((msg['timestamp'] for msg in messages_data if msg['user'] == member), default=None)
            if first_msg:
                member_status[member] = {
                    'first_seen': first_msg,
                    'last_left': left_dates[-1] if left_dates else None
                }

    return {
        'messages_data': messages_data,
        'user_messages': user_messages,
        'member_status': member_status,
        'total_members': len(member_status),
        'current_members': sum(1 for m in member_status.values() if not m['last_left'])
    }

def create_weekly_breakdown(stats):
    """Create weekly breakdown with improved member counting."""
    if not stats['messages_data']:
        return pd.DataFrame()

    # Convert messages to DataFrame
    df = pd.DataFrame(stats['messages_data'])
    
    # Get the first Monday and calculate weeks
    min_date = df['timestamp'].min()
    first_monday = min_date - timedelta(days=min_date.weekday())
    max_date = df['timestamp'].max()
    weeks = pd.date_range(start=first_monday, end=max_date, freq='W-MON')
    
    weekly_data = []
    current_members = set()
    left_members = set()

    for week_start in weeks:
        week_end = week_start + timedelta(days=6)
        week_number = len(weekly_data) + 1  # Start from week 1
        
        # Update member counts
        for member, status in stats['member_status'].items():
            if status['first_seen'] <= week_end:
                current_members.add(member)
            if status['last_left'] and week_start <= status['last_left'] <= week_end:
                current_members.discard(member)
                left_members.add(member)
        
        # Get messages for the week
        week_mask = (df['timestamp'] >= week_start) & (df['timestamp'] <= week_end)
        week_messages = df[week_mask].groupby('user').size().to_dict()
        
        # Add data for each active member
        for member in current_members:
            weekly_data.append({
                'Week': f'Week {week_number}',
                'Week Duration': f"{week_start.strftime('%d %b %Y')} - {week_end.strftime('%d %b %Y')}",
                'Member Name': member,
                'Messages Sent': week_messages.get(member, 0),
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
            
            # Weekly breakdown
            weekly_df = create_weekly_breakdown(stats)
            st.markdown("### Weekly Message & Member Analysis")
            st.dataframe(weekly_df)
            
            # Message distribution chart
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
                         f"- Top contributors: {dict(Counter(stats['user_messages']).most_common(5))}\n")
                word_placeholder = st.empty()
                get_llm_reply(client, prompt, word_placeholder)
        else:
            st.error("Failed to parse chat log.")

if __name__ == "__main__":
    main()
