import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter
from datetime import datetime
from dateutil import parser as date_parser
import numpy as np
from io import StringIO, BytesIO

# Configure Streamlit page and memory management
st.set_page_config(layout="wide")
st.cache_data.clear()

@st.cache_data
def parse_date(date_str):
    """Enhanced date parsing with multiple format support."""
    try:
        return date_parser.parse(date_str, fuzzy=True)
    except Exception:
        formats = [
            '%d/%m/%Y, %H:%M:%S',
            '%d/%m/%Y, %H:%M',
            '%m/%d/%Y, %H:%M:%S',
            '%m/%d/%Y, %H:%M'
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

def process_line(line, patterns):
    """Process a single line of chat log."""
    line = line.strip()
    if not line:
        return None
    
    msg_match = patterns['message'].match(line)
    event_match = patterns['event'].match(line)
    add_remove_match = patterns['add_remove'].match(line)
    
    if msg_match:
        timestamp_str, user, message = msg_match.groups()
        timestamp = parse_date(timestamp_str)
        if timestamp:
            user = clean_member_name(user)
            return {
                'type': 'message',
                'data': {
                    "timestamp": timestamp,
                    "user": user,
                    "message": message
                }
            }
    
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
            actor = None
        
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
            
            if actor:
                event_data['actor'] = actor
                
            return {
                'type': 'event',
                'data': event_data
            }
    
    return None

def read_in_chunks(file_object, chunk_size=8192):
    """Generator to read file in chunks."""
    while True:
        chunk = file_object.read(chunk_size)
        if not chunk:
            break
        yield chunk

@st.cache_data
def parse_chat_log_file(uploaded_file):
    """Parse WhatsApp chat log file with improved memory efficiency."""
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

    messages_data = []
    user_messages = Counter()
    member_status = {}
    events = []
    
    # Read file in chunks and process line by line
    buffer = StringIO()
    
    # Create a bytes buffer for the uploaded file
    bytes_data = uploaded_file.getvalue()
    file_obj = BytesIO(bytes_data)
    
    for chunk in read_in_chunks(file_obj):
        try:
            text_chunk = chunk.decode('utf-8')
        except UnicodeDecodeError:
            text_chunk = chunk.decode('latin-1')
        
        buffer.write(text_chunk)
        buffer.seek(0)
        
        lines = buffer.readlines()
        if lines:
            buffer = StringIO(lines[-1])  # Keep incomplete last line
        else:
            buffer = StringIO()
        
        for line in lines[:-1]:  # Process complete lines
            result = process_line(line, patterns)
            if result:
                if result['type'] == 'message':
                    messages_data.append(result['data'])
                    user_messages[result['data']['user']] += 1
                    
                    if result['data']['user'] not in member_status:
                        member_status[result['data']['user']] = {
                            'first_seen': result['data']['timestamp'],
                            'last_seen': result['data']['timestamp']
                        }
                    else:
                        member_status[result['data']['user']]['last_seen'] = result['data']['timestamp']
                
                elif result['type'] == 'event':
                    events.append(result['data'])
    
    # Process any remaining incomplete line
    if buffer.getvalue():
        result = process_line(buffer.getvalue(), patterns)
        if result:
            if result['type'] == 'message':
                messages_data.append(result['data'])
                user_messages[result['data']['user']] += 1
            elif result['type'] == 'event':
                events.append(result['data'])
    
    # Calculate member status
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

@st.cache_data
def create_member_timeline(stats):
    """Create optimized timeline with membership events."""
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

@st.cache_data
def display_events_table(stats):
    """Display paginated membership events table."""
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
    
    # Add file size warning
    st.info("This application can process chat logs up to 200MB. For larger files, please split them into smaller chunks.")
    
    uploaded_file = st.file_uploader("Upload WhatsApp chat log (TXT format)", type="txt")
    
    if uploaded_file is not None:
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        
        if file_size > 200:
            st.error(f"File size ({file_size:.1f}MB) exceeds the 200MB limit. Please upload a smaller file.")
            return
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("Processing chat log... This may take a few minutes for large files.")
        
        try:
            stats = parse_chat_log_file(uploaded_file)
            progress_bar.progress(1.0)
            progress_text.empty()
            
            if not stats['messages_data'] and not stats['events']:
                st.error("No valid messages or events found in the file. Please check the file format.")
                return
            
            # Display membership metrics
            st.header("Group Membership Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Members", stats['total_members'])
            with col2:
                st.metric("Current Members", stats['current_members'])
            with col3:
                st.metric("Left Members", stats['left_members'])
            
            # Membership Events Table with pagination
            st.subheader("Membership Events")
            events_df = display_events_table(stats)
            if not events_df.empty:
                ROWS_PER_PAGE = 50
                total_pages = len(events_df) // ROWS_PER_PAGE + (1 if len(events_df) % ROWS_PER_PAGE > 0 else 0)
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                
                start_idx = (page - 1) * ROWS_PER_PAGE
                end_idx = start_idx + ROWS_PER_PAGE
                
                tab1, tab2, tab3 = st.tabs(["All Events", "Exits Only", "Joins Only"])
                
                with tab1:
                    st.dataframe(events_df[['Date', 'Time', 'Event Description']].iloc[start_idx:end_idx])
                
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
                
                fig.add_trace(go.Scatter(
                    x=timeline_df['Date'],
                    y=timeline_df['Member Count'],
                    mode='lines',
                    name='Member Count',
                    line=dict(color='#2E86C1', width=2)
                ))
                
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
                st.plotly_chart(fig, use_container_width=True)
            
            # Message Analysis
            st.header("Message Analysis")
            if stats['messages_data']:
                message_df = pd.DataFrame(list(stats['user_messages'].items()), 
                                       columns=['Member', 'Messages'])
                message_df = message_df.sort_values('Messages', ascending=False)
                fig = px.bar(message_df, x='Member', y='Messages', 
                            title='Messages per Member')
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return
        finally:
            progress_bar.empty()

if __name__ == "__main__":
    main()
