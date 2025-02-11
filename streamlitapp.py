import streamlit as st
import pandas as pd
import plotly.express as px
import re
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
# (Optional) Custom CSS for layout and spacing (can be removed)
# -------------------------------
st.markdown("""
    <style>
        .custom-table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        .custom-table th, .custom-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .custom-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .chart-container {
            margin-top: 20px;
            margin-bottom: 40px;
        }
    </style>
""", unsafe_allow_html=True)

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
      - left_dates: Dictionary mapping member name to the timestamp they left (if any)
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
        left_dates = {}

        # Regex pattern for WhatsApp messages (optional square brackets and optional dash)
        message_pattern = re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*?):\s(.*)$'
        )
        # Regex pattern for "left" events
        left_pattern = re.compile(
            r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*-?\s*(.*) left'
        )
        # Other system messages (join, etc.)
        system_patterns = [
            r'(.+) added (.+)',
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

            # Check for "left" events first
            left_match = re.match(left_pattern, line)
            if left_match:
                timestamp_str, member = left_match.groups()
                member = clean_member_name(member)
                try:
                    left_date = date_parser.parse(timestamp_str, fuzzy=True)
                except Exception:
                    left_date = None
                if left_date is not None:
                    # Record earliest left date if multiple exist
                    if member not in left_dates or left_date < left_dates[member]:
                        left_dates[member] = left_date
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
            'global_members': sorted(global_members),
            'left_dates': left_dates
        }
    except Exception as e:
        st.error(f"Error parsing chat log data: {str(e)}")
        return None

# -------------------------------
# Display Functions for Tables & Charts
# -------------------------------
def display_weekly_messages_table(messages_data, global_members, left_dates):
    """
    Create Table 1: Weekly Message Breakdown.
    For each week (Monday to Sunday) up to the current week, list each member (only if they joined by that week and haven't left before the week ends)
    with their message count (or 0 if inactive), cumulative follower count, and an indicator if they left during that week.
    """
    try:
        if not messages_data:
            st.write("No messages to display")
            return

        df = pd.DataFrame(messages_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)

        # Compute the Monday for each message's week (using to_period('W'))
        df['Week Start'] = df['Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
        if df.empty:
            st.write("No valid messages to display")
            return

        # Limit weeks up to the current week (Monday)
        current_week_start = datetime.now() - timedelta(days=datetime.now().weekday())
        current_week_start = current_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        first_monday = df['Week Start'].min()
        # Generate weeks from first_monday to current_week_start
        weeks = pd.date_range(start=first_monday, end=current_week_start, freq='W-MON')

        # Compute each member's join date (first message timestamp)
        member_join_dates = df.groupby('Member Name')['Timestamp'].min().to_dict()

        rows = []
        week_counter = 1
        cumulative_followers = set()
        for week_start in weeks:
            week_end = week_start + timedelta(days=6)
            week_mask = (df['Week Start'] == week_start)
            week_messages = df[week_mask]
            # Eligible members: those whose join date is <= week_end
            eligible_members = [m for m, join_date in member_join_dates.items() if join_date <= week_end]
            # Skip week if no eligible members to avoid empty week numbering
            if not eligible_members:
                continue
            # Update cumulative followers: if a member joins in a week, they remain until they leave
            cumulative_followers.update(eligible_members)
            follower_count = len(cumulative_followers)
            for member in sorted(eligible_members):
                count = week_messages[week_messages['Member Name'] == member].shape[0] if not week_messages.empty else 0
                # Determine if this member left during this week
                left_this_week = ""
                if member in left_dates:
                    if week_start <= left_dates[member] <= week_end:
                        left_this_week = left_dates[member].strftime("%d %b %Y")
                rows.append({
                    'Week': f"Week {week_counter}",
                    'Week Duration': f"{week_start.strftime('%d %b %Y')} - {week_end.strftime('%d %b %Y')}",
                    'Member Name': member,
                    'Number of Messages Sent': count,
                    'Follower Count': follower_count,
                    'Left This Week': left_this_week
                })
            week_counter += 1

        weekly_df = pd.DataFrame(rows)
        st.markdown("### Table 1: Weekly Message Breakdown")
        st.dataframe(weekly_df)

        # Plot a Plotly bar chart (only one chart)
        totals = weekly_df.groupby('Member Name')['Number of Messages Sent'].sum().reset_index()
        fig = px.bar(totals, x='Member Name', y='Number of Messages Sent',
                     title='Total Messages Sent by Each User', color='Number of Messages Sent')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating weekly message table: {str(e)}")

def display_member_statistics(messages_data, left_dates):
    """
    Create Table 2: Member Statistics.
    For each member, show:
      - Unique Member Name
      - Group Activity Status (Active if they haven't left before the current week, otherwise Inactive)
      - Membership Duration (Weeks) from first message until they left (or current week if still active)
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

        # Compute membership duration in days, and clip negative values to 0
        duration_days = (current_week_start - grouped['first_message']).dt.days.clip(lower=0)
        grouped['Membership Duration (Weeks)'] = (duration_days / 7).round().astype('Int64')
        grouped['Avg. Weekly Messages'] = grouped.apply(
            lambda row: round(row['total_messages'] / max(row['Membership Duration (Weeks)'], 1), 2),
            axis=1
        )

        # Activity: if the member has a left date before current_week_start, mark as Inactive, else Active.
        grouped['Group Activity Status'] = grouped['Member Name'].apply(
            lambda m: 'Inactive' if m in left_dates and left_dates[m] < current_week_start else 'Active'
        )

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
        fig = px.bar(df, x='Member Name', y='Messages',
                     title='Total Messages Sent by Each User', color='Messages')
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
        display_weekly_messages_table(stats['messages_data'], stats['global_members'], stats['left_dates'])
        # Display Table 2: Member Statistics
        display_member_statistics(stats['messages_data'], stats['left_dates'])
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
