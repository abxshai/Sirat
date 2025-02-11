import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from groq import Groq
from dateutil import parser as date_parser

# [Previous helper functions remain the same until create_member_activity_table]

def create_member_timeline(stats):
    """Create a timeline of member count changes."""
    # Get all events that affect member count
    events = []
    
    # Add join events
    for member, status in stats['member_status'].items():
        if status['first_seen']:
            events.append({
                'date': status['first_seen'],
                'change': 1,
                'event': 'join',
                'member': member
            })
        if status['last_left']:
            events.append({
                'date': status['last_left'],
                'change': -1,
                'event': 'left',
                'member': member
            })
    
    # Sort events by date
    events.sort(key=lambda x: x['date'])
    
    # Create daily member count
    if not events:
        return pd.DataFrame()
        
    date_range = pd.date_range(
        start=events[0]['date'].date(),
        end=events[-1]['date'].date() + timedelta(days=1),
        freq='D'
    )
    
    member_count = 0
    daily_counts = []
    
    for date in date_range:
        # Process all events for this day
        day_events = [e for e in events if e['date'].date() == date.date()]
        for event in day_events:
            member_count += event['change']
        
        daily_counts.append({
            'Date': date,
            'Member Count': member_count,
        })
    
    return pd.DataFrame(daily_counts)

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
            else:
                st.warning("No member timeline data available")
            
            # Member Activity Table
            st.markdown("### Member Activity Status")
            activity_df = create_member_activity_table(stats)
            st.dataframe(activity_df, use_container_width=True)
            
            # Display activity statistics
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
                         f"- Active members: {active_count}\n"
                         f"- Inactive members: {inactive_count}\n"
                         f"- Top contributors: {dict(Counter(stats['user_messages']).most_common(5))}\n")
                word_placeholder = st.empty()
                get_llm_reply(client, prompt, word_placeholder)
        else:
            st.error("Failed to parse chat log.")

if __name__ == "__main__":
    main()
