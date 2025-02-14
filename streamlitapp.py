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

def parse_chat_log_file(uploaded_file):
    """Enhanced parser with better date handling and exit event tracking."""
    try:
        content = uploaded_file.read()
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

    messages_data = []
    user_messages = Counter()
    member_status = {}
    exit_events = []

    # Enhanced patterns to catch more variations
    patterns = {
        'message': re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?):\s(.*)$'
        ),
        'join': re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?)\s+(?:joined|was added)'),
        'left': re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?)\s+(?:left|was removed)')
    }

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        timestamp_str = None
        user = None
        event_type = None

        # Check each pattern
        for event_name, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                timestamp_str = match.group(1)
                user = clean_member_name(match.group(2))
                event_type = event_name
                break

        if not timestamp_str or not user:
            continue

        timestamp = parse_date(timestamp_str)
        if not timestamp:
            continue

        if event_type == 'message':
            message = match.group(3)
            messages_data.append({
                "timestamp": timestamp,
                "user": user,
                "message": message,
                "event_type": "message"
            })
            user_messages[user] += 1
            if user not in member_status:
                member_status[user] = {'first_seen': timestamp, 'last_left': None}
        
        elif event_type == 'join':
            messages_data.append({
                "timestamp": timestamp,
                "user": user,
                "message": "joined the group",
                "event_type": "join"
            })
            if user not in member_status:
                member_status[user] = {'first_seen': timestamp, 'last_left': None}
        
        elif event_type == 'left':
            messages_data.append({
                "timestamp": timestamp,
                "user": user,
                "message": "left the group",
                "event_type": "left"
            })
            if user in member_status:
                member_status[user]['last_left'] = timestamp
                exit_events.append({
                    'user': user,
                    'timestamp': timestamp,
                    'event_type': 'left'
                })

    current_members = sum(1 for m in member_status.values() if not m['last_left'])
    left_members = sum(1 for m in member_status.values() if m['last_left'])

    return {
        'messages_data': messages_data,
        'user_messages': user_messages,
        'member_status': member_status,
        'total_members': len(member_status),
        'current_members': current_members,
        'left_members': left_members,
        'exit_events': exit_events
    }

def create_member_timeline(stats):
    """Create enhanced timeline with join and exit events."""
    if not stats['messages_data']:
        return pd.DataFrame()
    
    df = pd.DataFrame(stats['messages_data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for join and left events
    events_df = df[df['event_type'].isin(['join', 'left'])].copy()
    events_df['member_change'] = events_df['event_type'].map({'join': 1, 'left': -1})
    
    # Sort by timestamp and calculate running total
    events_df = events_df.sort_values('timestamp')
    events_df['member_count'] = events_df['member_change'].cumsum()
    
    # Add event annotations
    events_df['Event'] = events_df.apply(
        lambda x: f"{x['user']} {x['event_type']} the group", axis=1
    )
    
    return events_df

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
        
        # Create member timeline visualization
        st.markdown("### Group Member Timeline")
        timeline_df = create_member_timeline(stats)
        if not timeline_df.empty:
            fig = go.Figure()
            
            # Add member count line
            fig.add_trace(go.Scatter(
                x=timeline_df['timestamp'],
                y=timeline_df['member_count'],
                mode='lines+markers',
                name='Member Count',
                line=dict(color='#2E86C1', width=2)
            ))
            
            # Add event markers
            for event_type, color in [('join', 'green'), ('left', 'red')]:
                event_data = timeline_df[timeline_df['event_type'] == event_type]
                if not event_data.empty:
                    fig.add_trace(go.Scatter(
                        x=event_data['timestamp'],
                        y=event_data['member_count'],
                        mode='markers',
                        name=f'{event_type.title()} Events',
                        marker=dict(color=color, size=10),
                        text=event_data['Event'],
                        hoverinfo='text+x+y'
                    ))
            
            fig.update_layout(
                title='Group Member Count Timeline with Events',
                xaxis_title='Date',
                yaxis_title='Number of Members',
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display exit events summary
            st.markdown("### Member Exit Summary")
            if stats['exit_events']:
                exit_df = pd.DataFrame(stats['exit_events'])
                exit_df['timestamp'] = pd.to_datetime(exit_df['timestamp'])
                exit_df = exit_df.sort_values('timestamp')
                st.dataframe(
                    exit_df.assign(
                        exit_date=exit_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                    )[['user', 'exit_date']],
                    use_container_width=True
                )
                st.metric("Total Members Left", stats['left_members'])
            else:
                st.write("No exit events recorded")

        # Message distribution using Plotly
        st.markdown("### Message Distribution")
        message_df = pd.DataFrame(list(stats['user_messages'].items()), 
                                columns=['Member', 'Messages'])
        fig = px.bar(message_df, 
                    x='Member', 
                    y='Messages',
                    title='Messages per Member',
                    color='Messages',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
