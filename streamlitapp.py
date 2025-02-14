import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter, defaultdict
from datetime import datetime
from groq import Groq
from dateutil import parser as date_parser
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_date(date_str):
    """Enhanced date parsing with multiple format support."""
    try:
        date = date_parser.parse(date_str, fuzzy=True)
        return date
    except Exception:
        formats = [
            '%d/%m/%Y, %H:%M:%S',
            '%d/%m/%Y, %H:%M',
            '%m/%d/%Y, %H:%M:%S',
            '%m/%d/%Y, %H:%M',
            '%d.%m.%Y, %H:%M:%S',
            '%d.%m.%Y, %H:%M'
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

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
    events = []  # Track all member events

    patterns = {
        'message': re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?):\s(.*)$'
        ),
        'event': re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(?:You |)(.*?)\s(joined|left|added|removed|was added|was removed)'
        ),
        'add_remove': re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?)\s(?:added|removed)\s(.*?)$'
        )
    }

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        msg_match = patterns['message'].match(line)
        event_match = patterns['event'].match(line)
        add_remove_match = patterns['add_remove'].match(line)

        if msg_match:
            timestamp_str, user, message = msg_match.groups()
            try:
                timestamp = parse_date(timestamp_str)
                if timestamp:
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
            except Exception:
                continue

        elif event_match or add_remove_match:
            match = event_match or add_remove_match
            timestamp_str = match.group(1)
            
            if add_remove_match:
                actor = clean_member_name(match.group(2))
                user = clean_member_name(match.group(3))
                action = 'added' if 'added' in line else 'removed'
            else:
                user = clean_member_name(match.group(2))
                action = match.group(3)

            try:
                timestamp = parse_date(timestamp_str)
                if timestamp:
                    event_type = 'join' if action in ['joined', 'was added', 'added'] else 'exit'
                    change = 1 if event_type == 'join' else -1

                    event_data = {
                        'timestamp': timestamp,
                        'user': user,
                        'event_type': event_type,
                        'change': change,
                        'original_action': action
                    }
                    
                    if add_remove_match:
                        event_data['actor'] = actor

                    events.append(event_data)

                    if event_type == 'join':
                        if user not in member_status:
                            member_status[user] = {'first_seen': timestamp, 'last_seen': timestamp}
                    else:  # exit
                        if user in member_status:
                            member_status[user]['last_seen'] = timestamp
            except Exception:
                continue

    user_last_events = {}
    for event in sorted(events, key=lambda x: x['timestamp']):
        user_last_events[event['user']] = event['event_type']

    current_members = sum(1 for status in user_last_events.values() if status == 'join')

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
    
    events_df = pd.DataFrame(stats['events'])
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    events_df = events_df.sort_values('timestamp')
    
    member_count = 0
    timeline_data = []
    
    for _, event in events_df.iterrows():
        member_count += event['change']
        event_text = f"{event['user']} {event['original_action']}"
        if 'actor' in event:
            event_text = f"{event['actor']} {event['original_action']} {event['user']}"
            
        timeline_data.append({
            'Date': event['timestamp'],
            'Member Count': member_count,
            'Event': event_text,
            'Event Type': event['event_type']
        })
    
    return pd.DataFrame(timeline_data)

def display_events_table(stats):
    """Display a separate table for all membership events."""
    if not stats['events']:
        return pd.DataFrame()
    
    events_df = pd.DataFrame(stats['events'])
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    events_df = events_df.sort_values('timestamp', ascending=False)
    
    display_df = events_df.copy()
    display_df['Date'] = display_df['timestamp'].dt.strftime('%d %b %Y')
    display_df['Time'] = display_df['timestamp'].dt.strftime('%H:%M')
    
    display_df['Event Description'] = display_df.apply(
        lambda x: f"{x['actor']} {x['original_action']} {x['user']}" if 'actor' in x else f"{x['user']} {x['original_action']}", 
        axis=1
    )
    
    return display_df[['Date', 'Time', 'Event Description', 'event_type']]

def main():
    st.title("WhatsApp Chat Analyzer")
    uploaded_file = st.file_uploader("Upload WhatsApp chat log (TXT format)", type="txt")
    
    if uploaded_file is not None:
        stats = parse_chat_log_file(uploaded_file)
        
        if stats:
            df = pd.DataFrame(stats['messages_data'])
            
            # Member Timeline and Events Section
            st.header("Group Membership Analysis")
            
            # Display membership metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Members", stats['total_members'])
            with col2:
                st.metric("Current Members", stats['current_members'])
            with col3:
                st.metric("Left Members", stats['left_members'])
            
            # Membership Events Table
            st.subheader("Membership Events")
            events_df = display_events_table(stats)
            if not events_df.empty:
                # Create tabs for different event types
                tab1, tab2, tab3 = st.tabs(["All Events", "Exits Only", "Joins Only"])
                
                with tab1:
                    st.dataframe(events_df[['Date', 'Time', 'Event Description']])
                
                with tab2:
                    exits_df = events_df[events_df['event_type'] == 'exit']
                    st.dataframe(exits_df[['Date', 'Time', 'Event Description']])
                
                with tab3:
                    joins_df = events_df[events_df['event_type'] == 'join']
                    st.dataframe(joins_df[['Date', 'Time', 'Event Description']])
            
            # Member Timeline Visualization
            st.subheader("Member Count Timeline")
            timeline_df = create_member_timeline(stats)
            if not timeline_df.empty:
                fig = go.Figure()
                
                # Add member count line
                fig.add_trace(go.Scatter(
                    x=timeline_df['Date'],
                    y=timeline_df['Member Count'],
                    mode='lines',
                    name='Member Count',
                    line=dict(color='#2E86C1', width=2)
                ))
                
                # Add exit events as markers
                exits_df = timeline_df[timeline_df['Event Type'] == 'exit']
                if not exits_df.empty:
                    fig.add_trace(go.Scatter(
                        x=exits_df['Date'],
                        y=exits_df['Member Count'],
                        mode='markers',
                        name='Exit Events',
                        marker=dict(
                            color='red',
                            size=8,
                            symbol='diamond'
                        ),
                        hovertemplate="<b>%{text}</b><br>" +
                                    "Date: %{x}<br>" +
                                    "Members: %{y}<extra></extra>",
                        text=exits_df['Event']
                    ))
                
                fig.update_layout(
                    title='Group Member Count Timeline with Exit Events',
                    xaxis_title='Date',
                    yaxis_title='Number of Members',
                    hovermode='x unified',
                    showlegend=True
                )
                st.plotly_chart(fig)
            
            # Message Analysis Section
            st.header("Message Analysis")
            if len(df) > 0:
                message_df = pd.DataFrame(list(stats['user_messages'].items()), 
                                       columns=['Member', 'Messages'])
                fig = px.bar(message_df, x='Member', y='Messages', 
                            title='Messages per Member')
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
