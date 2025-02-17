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
from datetime import datetime
import io
import concurrent.futures

##############################
# Basic Functions & Preprocessing
##############################

def clean_member_name(name):
    """Clean member name, format phone numbers consistently."""
    cleaned = name.strip()
    digits_only = re.sub(r'\D', '', cleaned)
    # If it looks like a phone number, replace with 'User ####'
    if len(cleaned) - len(digits_only) <= 2 and len(digits_only) >= 7:
        return f"User {digits_only[-4:]}"
    return cleaned

def initialize_llm_client():
    """Initialize Groq client with API key (if using LLM)."""
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

def parse_date(date_str):
    """
    Parse a WhatsApp-style date string.
    We use dayfirst=True to handle dd/mm/yy formats typical in some exports.
    """
    try:
        return date_parser.parse(date_str, fuzzy=False, dayfirst=True)
    except Exception:
        return None

##############################
# Unified Parsing of Lines
##############################

# One pattern to capture [date/time], remainder:
TIMESTAMP_PATTERN = re.compile(
    r'^\[?('
    r'\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}'
    r'(?::\d{2})?\s*[APap][Mm]?'
    r')\]?\s*(.*)$'
)

def process_line(line):
    """
    Parse a single line from the chat log to determine if it's:
      - A message (with user + message text)
      - A join event
      - An exit event
    Returns a tuple: (timestamp, user, message, event_type)
      event_type can be "message", "join", or "exit"
      For join/exit, message may be empty or None
    If parsing fails, returns None.
    """
    line = line.strip()
    if not line:
        return None

    match = TIMESTAMP_PATTERN.match(line)
    if not match:
        return None  # Line doesn't match the typical "[date] something" pattern

    timestamp_str = match.group(1).strip()
    remainder = match.group(2).strip()
    dt = parse_date(timestamp_str)
    if not dt:
        return None

    # Check if remainder ends with 'joined' or 'left' (case insensitive)
    lower_remainder = remainder.lower()
    if lower_remainder.endswith(" joined"):
        # e.g. "Sohil Badhwar_SW joined"
        user_part = remainder[:-6].strip()  # remove 'joined'
        return dt, clean_member_name(user_part), None, "join"
    elif lower_remainder.endswith(" left"):
        # e.g. "Sonaali left"
        user_part = remainder[:-4].strip()  # remove 'left'
        return dt, clean_member_name(user_part), None, "exit"
    else:
        # It's a message with user + message
        # We split on the LAST occurrence of ": " 
        # because user names can contain colons too.
        splitted = remainder.rsplit(": ", 1)
        if len(splitted) == 2:
            user_part, message_part = splitted
            return dt, clean_member_name(user_part), message_part.strip(), "message"
        else:
            # Can't parse user:message format
            # Possibly a system message or unknown format
            return None

def process_chunk(chunk):
    """
    Process a chunk of lines from the chat log, returning lists of
    messages, joins, and exits. This uses our unified parse approach.
    """
    messages = []
    joins = []
    exits = []

    for line in chunk.splitlines():
        parsed = process_line(line)
        if not parsed:
            continue
        dt, user, message, event_type = parsed
        # Separate out event types
        if event_type == "join":
            joins.append({
                'timestamp': dt,
                'timestamp_str': line,  # or you could store the raw date part
                'user': user
            })
        elif event_type == "exit":
            exits.append({
                'timestamp': dt,
                'timestamp_str': line,
                'user': user
            })
        elif event_type == "message":
            messages.append({
                'timestamp': dt,
                'timestamp_str': timestamp_str_from_line(line),
                'user': user,
                'message': message
            })

    return messages, joins, exits

def timestamp_str_from_line(line):
    """
    Utility to extract just the date/time portion from a line
    (for a cleaner 'timestamp_str' field). Optional convenience.
    """
    m = TIMESTAMP_PATTERN.match(line.strip())
    return m.group(1).strip() if m else line.strip()

##############################
# Parse the Chat Log (File)
##############################

def parse_chat_log_file(uploaded_file, lines_per_chunk=1000):
    """
    Parse WhatsApp chat log file using the unified approach.
    Returns messages, join events, exit events, etc.
    """
    try:
        content = uploaded_file.read()
        # Try UTF-8 decode first, fallback to latin-1
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

    lines = text.splitlines()
    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk = "\n".join(lines[i:i+lines_per_chunk])
        chunks.append(chunk)
    
    all_messages = []
    all_joins = []
    all_exits = []

    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            messages, joins, exits = future.result()
            all_messages.extend(messages)
            all_joins.extend(joins)
            all_exits.extend(exits)

    # Sort them by timestamp
    all_messages.sort(key=lambda x: x['timestamp'])
    all_joins.sort(key=lambda x: x['timestamp'])
    all_exits.sort(key=lambda x: x['timestamp'])

    # Build a simple member status dictionary
    member_status = {}
    for event in all_joins + all_messages:
        user = event['user']
        if user not in member_status:
            member_status[user] = {
                'first_seen': event['timestamp'],
                'first_seen_str': event['timestamp_str']
            }
    for exit_event in all_exits:
        user = exit_event['user']
        if user not in member_status:
            member_status[user] = {
                'first_seen': exit_event['timestamp'],
                'first_seen_str': exit_event['timestamp_str']
            }
        if 'left_times' not in member_status[user]:
            member_status[user]['left_times'] = []
            member_status[user]['left_times_str'] = []
        member_status[user]['left_times'].append(exit_event['timestamp'])
        member_status[user]['left_times_str'].append(exit_event['timestamp_str'])

    total_members = len(member_status)
    left_members = sum(1 for m in member_status.values() if m.get('left_times'))
    current_members = total_members - left_members

    return {
        'messages_data': all_messages,
        'user_messages': Counter(msg['user'] for msg in all_messages),
        'member_status': member_status,
        'total_members': total_members,
        'current_members': current_members,
        'left_members': left_members,
        'exit_events': all_exits,
        'join_events': all_joins
    }

##############################
# Creating Tables & Charts
##############################

def create_final_exit_table(stats):
    """
    Build a DataFrame listing permanent exits only:
      - People who left, but did not rejoin afterwards.
    """
    exit_events = stats.get('exit_events', [])
    join_events = stats.get('join_events', [])

    if not exit_events:
        return pd.DataFrame()

    # For each user, we want the maximum join timestamp if any
    last_join_ts = {}
    for j in join_events:
        usr = j['user']
        tstamp = j['timestamp']
        if usr not in last_join_ts or tstamp > last_join_ts[usr]:
            last_join_ts[usr] = tstamp

    # Filter out exit events where user rejoined after that exit event
    filtered_exits = []
    for e in exit_events:
        usr = e['user']
        exit_time = e['timestamp']
        # If user rejoined later (join time > exit time), skip
        if usr in last_join_ts and last_join_ts[usr] > exit_time:
            continue
        filtered_exits.append(e)

    if not filtered_exits:
        return pd.DataFrame()

    df = pd.DataFrame(filtered_exits)
    df = df.rename(columns={
        'user': 'Name of Exit Person',
        'timestamp_str': 'Exit Date & Time (from the txt file)'
    })
    return df[['Name of Exit Person', 'Exit Date & Time (from the txt file)']]

def create_weekly_activity_table(stats):
    """
    Create a weekly breakdown of overall activity:
      - # of messages
      - Who joined that week
      - Who left that week
    Group by the actual date/time from the txt file (day-first).
    """
    messages = stats.get('messages_data', [])
    join_events = stats.get('join_events', [])
    exit_events = stats.get('exit_events', [])

    # If there's no message data at all, just return empty
    if not messages:
        return pd.DataFrame()

    msg_df = pd.DataFrame(messages)
    join_df = pd.DataFrame(join_events)
    exit_df = pd.DataFrame(exit_events)

    # 1. MESSAGES
    if not msg_df.empty and 'timestamp' in msg_df.columns:
        msg_df['timestamp'] = pd.to_datetime(msg_df['timestamp'])
        msg_df['week'] = msg_df['timestamp'].dt.to_period('W').dt.start_time
        weekly_msgs = msg_df.groupby('week').size().reset_index(name='Messages Sent')
    else:
        weekly_msgs = pd.DataFrame(columns=['week', 'Messages Sent'])

    # 2. JOINS
    if not join_df.empty and 'timestamp' in join_df.columns:
        join_df['timestamp'] = pd.to_datetime(join_df['timestamp'])
        join_df['week'] = join_df['timestamp'].dt.to_period('W').dt.start_time
        weekly_joins = join_df.groupby('week')['user'] \
            .apply(lambda x: ', '.join(sorted(set(x)))).reset_index(name='Joined')
    else:
        weekly_joins = pd.DataFrame(columns=['week', 'Joined'])

    # 3. EXITS
    if not exit_df.empty and 'timestamp' in exit_df.columns:
        exit_df['timestamp'] = pd.to_datetime(exit_df['timestamp'])
        exit_df['week'] = exit_df['timestamp'].dt.to_period('W').dt.start_time
        weekly_exits = exit_df.groupby('week')['user'] \
            .apply(lambda x: ', '.join(sorted(set(x)))).reset_index(name='Left')
    else:
        weekly_exits = pd.DataFrame(columns=['week', 'Left'])

    # Merge them all
    weekly_data = pd.merge(weekly_msgs, weekly_joins, on='week', how='outer')
    weekly_data = pd.merge(weekly_data, weekly_exits, on='week', how='outer')
    weekly_data = weekly_data.fillna('')
    weekly_data = weekly_data.sort_values('week')

    return weekly_data

def create_wordcloud(all_messages):
    """
    Generate a word cloud from all messages.
    """
    texts = [m['message'] for m in all_messages]
    combined_text = " ".join(texts)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        collocations=False
    ).generate(combined_text)
    return wordcloud

##############################
# Main Streamlit App
##############################

def main():
    st.title("WhatsApp Chat Analyzer")
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
        
        # Show overall analysis
        if st.button("Show Analysis"):
            st.header("Overall Chat Analysis")
            
            total_messages = len(stats['messages_data'])
            total_words = sum(len(msg['message'].split()) for msg in stats['messages_data'])
            media_messages = sum(1 for msg in stats['messages_data'] if "<Media omitted>" in msg['message'])
            links_shared = sum(1 for msg in stats['messages_data'] if "http" in msg['message'].lower())

            col1, col2, col3, col4 = st.columns(4)
            metrics = [
                ("Total Messages", total_messages),
                ("Total Words", total_words),
                ("Media Shared", media_messages),
                ("Links Shared", links_shared)
            ]
            for col, (metric_title, metric_value) in zip([col1, col2, col3, col4], metrics):
                with col:
                    st.metric(metric_title, metric_value)

            st.subheader("Permanent Member Exit Analysis")
            exit_df = create_final_exit_table(stats)
            if not exit_df.empty:
                st.dataframe(exit_df)
                st.metric("Total Permanent Leavers", len(exit_df))
            else:
                st.write("No permanent exit events recorded.")

            st.subheader("Weekly Member Activity")
            weekly_df = create_weekly_activity_table(stats)
            if not weekly_df.empty:
                st.dataframe(weekly_df)
            else:
                st.write("No weekly activity data available.")

            st.subheader("Message Distribution")
            message_df = pd.DataFrame(list(stats['user_messages'].items()), columns=['Member', 'Messages'])
            message_df = message_df.sort_values('Messages', ascending=False)
            fig = px.bar(
                message_df,
                x='Member',
                y='Messages',
                title='Messages per Member',
                color='Messages',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis_title="Member",
                yaxis_title="Number of Messages",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Word Cloud (Overall)")
            try:
                wc = create_wordcloud(stats['messages_data'])
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")

            # Optionally, if you want an LLM summary:
            # st.subheader("Chat Analysis Summary (LLM)")
            # if st.button("Generate LLM Summary"):
            #     try:
            #         client = initialize_llm_client()
            #         top_contributors = dict(Counter(stats['user_messages']).most_common(5))
            #         prompt = (
            #             f"Analyze this WhatsApp chat log summary:\n"
            #             f"- Total members: {stats['total_members']}\n"
            #             f"- Currently active members: {stats['current_members']}\n"
            #             f"- Members who left: {stats['left_members']}\n"
            #             f"- Total messages: {total_messages}\n"
            #             f"- Top 5 contributors: {top_contributors}\n"
            #             f"- Media messages shared: {media_messages}\n"
            #             f"- Links shared: {links_shared}\n\n"
            #             "Provide insights about group dynamics, engagement patterns, and member participation."
            #         )
            #         placeholder = st.empty()
            #         get_llm_reply(client, prompt, placeholder)
            #     except Exception as e:
            #         st.error(f"Error generating LLM analysis: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="WhatsApp Chat Analyzer",
        page_icon="💬",
        layout="wide"
    )
    main()
