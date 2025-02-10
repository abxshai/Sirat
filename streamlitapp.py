import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import re
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from groq import Groq
from dateutil import parser as date_parser  # For robust date parsing

# -------------------------------
# Helper Function to Clean Member Names
# -------------------------------
def clean_member_name(name):
    """
    If the provided name is primarily a phone number, return a formatted string
    using the last 4 digits. Otherwise, return the cleaned name.
    """
    cleaned = name.strip()
    digits_only = re.sub(r'\D', '', cleaned)
    if len(cleaned) - len(digits_only) <= 2 and len(digits_only) >= 7:
        return "User " + digits_only[-4:]
    else:
        return cleaned

# -------------------------------
# Initialize LLM Client
# -------------------------------
API_KEY = st.secrets["API_KEY"]
client = Groq(api_key=API_KEY)

def get_llm_reply(client, prompt, word_placeholder):
    """Get an LLM summary reply using the Groq API with model llama-3.3-70b-versatile."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Analyze the chat log, and summarize key details such as the highest message sender, "
                        "people who joined the group, and joining/exiting trends on a weekly or monthly basis, "
                        "mention the inactive members' names with a message count (0 if none), and display everything in a tabular format."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
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

# -------------------------------
# Function to parse the chat log from a TXT file
# -------------------------------
def parse_chat_log_file(uploaded_file):
    """
    Parse a WhatsApp chat log from an uploaded TXT file.
    Returns a dictionary with:
      - messages_data: list of dicts with keys 'Timestamp', 'Member Name', 'Message'
      - user_messages: Counter of messages per user
      - global_members: Sorted list of all member names
      - join_exit_events: List of system messages (if any)
    """
    try:
        file_content = uploaded_file.read()
        try:
            text = file_content.decode("utf-8")
        except Exception:
            text = file_content.decode("latin-1")
        chats = text.splitlines()
    except Exception as e:
        st.error(f"Error reading chat log: {str(e)}")
        return None

    try:
        total_messages = 0
        user_messages = Counter()
        join_exit_events = []
        messages_data = []
        global_members = set()

        # Regex pattern for WhatsApp messages (accepts optional square brackets and optional dash)
        message_pattern = re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?):\s(.*)$'
        )

        # System message patterns (for join/exit events)
        system_patterns = [
            r'(.+) added (.+)',
            r'(.+) left',
            r'(.+) removed (.+)',
            r'(.+) joined using this group\'s invite link',
            r'(.+) changed the subject from "(.+)" to "(.+)"',
            r'(.+) changed this group\'s icon',
            r'Messages and calls are end-to-end encrypted',
            r'(.+) changed the group description',
            r'(.+) changed their phone number'
        ]
        system_pattern = '|'.join(system_patterns)

        for line in chats:
            line = line.strip()
            if not line:
                continue
            message_found = False
            match = re.match(message_pattern, line)
            if match:
                try:
                    timestamp_str, user, message = match.groups()
                    user = clean_member_name(user)
                    try:
                        parsed_date = date_parser.parse(timestamp_str, fuzzy=True)
                    except Exception:
                        parsed_date = None
                    if parsed_date is not None:
                        total_messages += 1
                        user_messages[user] += 1
                        messages_data.append({'Timestamp': parsed_date, 'Member Name': user, 'Message': message})
                        global_members.add(user)
                        message_found = True
                except Exception as e:
                    st.error(f"Error parsing line: {line} - {str(e)}")
                    continue
            if not message_found and re.search(system_pattern, line):
                join_exit_events.append(line)

        return {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'join_exit_events': join_exit_events,
            'messages_data': messages_data,
            'global_members': sorted(global_members)
        }
    except Exception as e:
        st.error(f"Error parsing chat log data: {str(e)}")
        return None

# -------------------------------
# Display Functions for Tables & Charts
# -------------------------------
def display_weekly_messages_table(messages_data, global_members):
    """
    Create Table 1: Weekly Message Breakdown.
    For each week (Monday to Sunday) up to the current week, list each member (only if they joined by that week)
    with their message count (or 0 if inactive) and the cumulative follower count.
    """
    try:
        if not messages_data:
            st.write("No messages to display")
            return

        df = pd.DataFrame(messages_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)

        # Compute the Monday for each message's week
        df['Week Start'] = df['Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)

        if df.empty:
            st.write("No valid messages to display")
            return

        # Limit weeks up to the current week
        current_week_start = datetime.now() - timedelta(days=datetime.now().weekday())
        current_week_start = current_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        first_monday = df['Week Start'].min()
        weeks = pd.date_range(start=first_monday, end=current_week_start, freq='W-MON')

        # For each member, compute join date as the first message timestamp
        member_join_dates = df.groupby('Member Name')['Timestamp'].min().to_dict()

        rows = []
        week_counter = 1
        for week_start in weeks:
            week_end = week_start + timedelta(days=6)
            week_mask = (df['Week Start'] == week_start)
            week_messages = df[week_mask]
            # Only include members who joined on or before the end of the week
            eligible_members = [m for m, join_date in member_join_dates.items() if join_date <= week_end]
            # Follower count: cumulative count of eligible members by this week
            follower_count = len(eligible_members)
            for member in sorted(eligible_members):
                count = week_messages[week_messages['Member Name'] == member].shape[0] if not week_messages.empty else 0
                rows.append({
                    'Week': f"Week {week_counter}",
                    'Week Duration': f"{week_start.strftime('%d %b %Y')} - {week_end.strftime('%d %b %Y')}",
                    'Member Name': member,
                    'Number of Messages Sent': count,
                    'Follower Count': follower_count
                })
            week_counter += 1

        weekly_df = pd.DataFrame(rows)
        st.markdown("### Table 1: Weekly Message Breakdown")
        st.dataframe(weekly_df)

        # Plot a bar chart of total messages per member from the weekly breakdown
        fig, ax = plt.subplots(figsize=(10, 5))
        user_totals = weekly_df.groupby('Member Name')['Number of Messages Sent'].sum().reset_index()
        if not user_totals.empty:
            ax.bar(user_totals['Member Name'], user_totals['Number of Messages Sent'], color='skyblue')
            ax.set_xlabel("Member Name")
            ax.set_ylabel("Total Messages")
            ax.set_title("Total Messages Sent by Each User")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        else:
            st.warning("No message data available to plot.")
    except Exception as e:
        st.error(f"Error creating weekly message table: {str(e)}")

def display_member_statistics(messages_data):
    """
    Create Table 2: Member Statistics.
    For each member, show:
      - Unique Member Name
      - Group Activity Status (Active if total messages > 0, otherwise Inactive)
      - Membership Duration (Weeks) from the first message until the current week
      - Avg. Weekly Messages
    """
    try:
        if not messages_data:
            st.write("No messages to display")
            return

        df = pd.DataFrame(messages_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)
        if df.empty:
            st.write("No valid messages to display")
            return

        grouped = df.groupby('Member Name').agg(
            first_message=('Timestamp', 'min'),
            total_messages=('Message', 'count')
        ).reset_index()

        current_week_start = datetime.now() - timedelta(days=datetime.now().weekday())
        current_week_start = current_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        # Calculate membership duration; clip negative values to 0
        duration_days = (current_week_start - grouped['first_message']).dt.days.clip(lower=0)
        grouped['Membership Duration (Weeks)'] = (duration_days / 7).round().astype('Int64')
        grouped['Avg. Weekly Messages'] = grouped.apply(
            lambda row: round(row['total_messages'] / max(row['Membership Duration (Weeks)'], 1), 2),
            axis=1
        )

        # Revised activity logic: if total_messages > 0 then Active, otherwise Inactive.
        grouped['Group Activity Status'] = grouped['total_messages'].apply(lambda x: 'Active' if x > 0 else 'Inactive')

        grouped.rename(columns={'Member Name': 'Unique Member Name'}, inplace=True)
        table2 = grouped[['Unique Member Name', 'Group Activity Status', 'Membership Duration (Weeks)', 'Avg. Weekly Messages']]
        st.markdown("### Table 2: Member Statistics")
        st.dataframe(table2)
    except Exception as e:
        st.error(f"Error creating member statistics: {str(e)}")

def display_total_messages_chart(user_messages):
    """
    Display a bar chart of total messages per user using Plotly Express.
    """
    try:
        if not user_messages:
            st.write("No message data to display")
            return
        df = pd.DataFrame(user_messages.items(), columns=['Member Name', 'Messages'])
        if df.empty:
            st.write("No message data to display")
            return
        fig = px.bar(
            df,
            x='Member Name',
            y='Messages',
            title='Total Messages Sent by Each User',
            color='Messages'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating messages chart: {str(e)}")

# -------------------------------
# Main App Layout
# -------------------------------
st.title("Structured Chat Log Analyzer")
uploaded_file = st.file_uploader("Upload a TXT file containing the WhatsApp chat log", type="txt")

if uploaded_file:
    stats = parse_chat_log_file(uploaded_file)
    if stats:
        st.success("Chat log parsed successfully!")
        # Display Table 1: Weekly Message Breakdown
        display_weekly_messages_table(stats['messages_data'], stats['global_members'])
        # Display Table 2: Member Statistics
        display_member_statistics(stats['messages_data'])
        # Display a bar chart for total messages per user
        display_total_messages_chart(stats['user_messages'])
        # LLM-based summary component (using aggregated data)
        st.markdown("### LLM Summary of Chat Log")
        if st.button("Generate Summary"):
            with st.spinner("Analyzing chat log..."):
                top_users = dict(stats['user_messages'].most_common(5))
                prompt = (f"Summarize the chat log with these key points:\n\n"
                          f"- Top message senders: {top_users}\n"
                          f"- Group join and exit events: {stats['join_exit_events'][:20]}\n")
                word_placeholder = st.empty()
                get_llm_reply(client, prompt, word_placeholder)
    else:
        st.error("Error parsing chat log.")
else:
    st.info("Please upload a TXT file containing the WhatsApp chat log.")
