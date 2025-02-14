import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter
from datetime import datetime
from groq import Groq
from dateutil import parser as date_parser

def clean_member_name(name):
    """Clean member name, format phone numbers consistently."""
    cleaned = name.strip()
    digits_only = re.sub(r'\D', '', cleaned)
    if len(cleaned) - len(digits_only) <= 2 and len(digits_only) >= 7:
        return f"User {digits_only[-4:]}"
    return cleaned

def parse_date(date_str):
    """Parse dates with better handling of various formats."""
    try:
        # Try parsing with dateutil first
        return date_parser.parse(date_str, fuzzy=True)
    except:
        try:
            # Common WhatsApp date formats
            patterns = [
                r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})',  # DD/MM/YY or DD/MM/YYYY
                r'(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})'     # YYYY/MM/DD
            ]
            
            for pattern in patterns:
                match = re.search(pattern, date_str)
                if match:
                    groups = match.groups()
                    if len(groups[2]) == 2:  # Two-digit year
                        year = 2000 + int(groups[2]) if int(groups[2]) < 50 else 1900 + int(groups[2])
                    else:
                        year = int(groups[2])
                    
                    if len(groups[0]) == 4:  # YYYY/MM/DD format
                        return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                    else:  # DD/MM/YYYY format
                        return datetime(year, int(groups[1]), int(groups[0]))
            
            raise ValueError(f"Could not parse date: {date_str}")
        except:
            return None

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
                st.dataframe(exit_df.assign(
                    exit_date=exit_df['timestamp'].dt.strftime('%d %b %Y')
                )[['user', 'exit_date']])
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
def parse_chat_log_file(uploaded_file):
    """Parse WhatsApp chat log file with enhanced event tracking."""
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
    events = []  # Track all events (joins, adds, exits)

    # Enhanced patterns to catch more variations
    message_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?):\s(.*)$'
    )
    
    # Updated patterns to catch more event types
    event_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?)\s(joined|left|added|removed|was added|was removed)'
    )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        msg_match = message_pattern.match(line)
        event_match = event_pattern.match(line)

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
                    member_status[user] = {'first_seen': timestamp, 'last_seen': timestamp}
                else:
                    member_status[user]['last_seen'] = timestamp
            except Exception as e:
                st.error(f"Date parsing error: {e}")
                continue

        elif event_match:
            timestamp_str, user, action = event_match.groups()
            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                user = clean_member_name(user)
                
                # Normalize action types
                if action in ['joined', 'was added', 'added']:
                    event_type = 'join'
                    change = 1
                else:  # left, removed, was removed
                    event_type = 'exit'
                    change = -1
                
                events.append({
                    'timestamp': timestamp,
                    'user': user,
                    'event_type': event_type,
                    'change': change,
                    'original_action': action
                })
                
                # Update member status
                if event_type == 'join':
                    if user not in member_status:
                        member_status[user] = {'first_seen': timestamp, 'last_seen': timestamp}
                else:  # exit
                    if user in member_status:
                        member_status[user]['last_seen'] = timestamp
                        
            except Exception as e:
                st.error(f"Event parsing error: {e}")
                continue

    # Calculate current status
    current_members = sum(1 for m in member_status.values() 
                         if not any(e['user'] == m and e['event_type'] == 'exit' 
                                  for e in reversed(events)))
    
    return {
        'messages_data': messages_data,
        'user_messages': user_messages,
        'member_status': member_status,
        'events': events,
        'total_members': len(member_status),
        'current_members': current_members,
        'left_members': len(member_status) - current_members
    }

def create_member_timeline(stats):
    """Create timeline with all membership events."""
    if not stats['events']:
        return pd.DataFrame()
    
    # Sort events chronologically
    events_df = pd.DataFrame(stats['events'])
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    events_df = events_df.sort_values('timestamp')
    
    # Calculate running member count
    member_count = 0
    timeline_data = []
    
    for _, event in events_df.iterrows():
        member_count += event['change']
        timeline_data.append({
            'Date': event['timestamp'],
            'Member Count': member_count,
            'Event': f"{event['user']} {event['event_type']}",
            'Event Type': event['event_type']
        })
    
    return pd.DataFrame(timeline_data)

def create_member_activity_table(stats):
    """Create activity status table with accurate dates from events."""
    activity_data = []
    
    for member, messages in stats['user_messages'].items():
        # Find first and last appearance from events
        member_events = [e for e in stats['events'] if e['user'] == member]
        first_event = min(member_events, key=lambda x: x['timestamp']) if member_events else None
        last_exit = max((e for e in member_events if e['event_type'] == 'exit'), 
                       key=lambda x: x['timestamp'], default=None)
        
        activity_data.append({
            'Member Name': member,
            'Message Count': messages,
            'Join Date': first_event['timestamp'].strftime('%d %b %Y') if first_event else 'Unknown',
            'Status': 'Left' if last_exit else 'Present',
            'Exit Date': last_exit['timestamp'].strftime('%d %b %Y') if last_exit else '-'
        })
    
    df = pd.DataFrame(activity_data)
    return df.sort_values(by=['Message Count', 'Member Name'], ascending=[False, True])

def display_events_table(stats):
    """Display a separate table for all membership events."""
    if not stats['events']:
        return pd.DataFrame()
    
    events_df = pd.DataFrame(stats['events'])
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    events_df = events_df.sort_values('timestamp')
    
    display_df = events_df.copy()
    display_df['Date'] = display_df['timestamp'].dt.strftime('%d %b %Y')
    display_df['Time'] = display_df['timestamp'].dt.strftime('%H:%M')
    
    return display_df[['Date', 'Time', 'user', 'event_type', 'original_action']]

if __name__ == "__main__":
    main()

