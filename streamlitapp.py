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
      - total_messages: total number of parsed messages
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

    total_messages = 0
    user_messages = Counter()
    join_exit_events = []
    messages_data = []
    global_members = set()
    left_dates = {}

    # Regex pattern for WhatsApp messages.
    # This pattern accepts an optional leading/trailing bracket around the timestamp.
    message_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap][Mm])?)\]?\s*(.*?):\s(.*)$'
    )
    # Regex pattern for "left" events (e.g., '... left')
    left_pattern = re.compile(
        r'^\[?(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap][Mm])?)\]?\s*(.*?) left'
    )
    
    # Buffer variables for multi-line messages
    current_timestamp = None
    current_user = None
    current_message = None

    for line in chats:
        line = line.strip()
        if not line:
            continue

        # Check if this line indicates a member left
        left_match = left_pattern.match(line)
        if left_match:
            timestamp_str, member = left_match.groups()
            member = clean_member_name(member)
            try:
                left_date = date_parser.parse(timestamp_str, fuzzy=True)
            except Exception:
                left_date = None
            if left_date is not None:
                # Store the latest left date (if multiple exist, the most recent one is used)
                if member not in left_dates or left_date > left_dates[member]:
                    left_dates[member] = left_date
            continue

        # Check if the line matches the message pattern
        match = message_pattern.match(line)
        if match:
            # If we have an ongoing buffered message, save it before starting a new one
            if current_message is not None and current_user is not None and current_timestamp is not None:
                messages_data.append({
                    'Timestamp': current_timestamp,
                    'Member Name': current_user,
                    'Message': current_message
                })
                user_messages[current_user] += 1
                total_messages += 1
            timestamp_str, user, message = match.groups()
            user = clean_member_name(user)
            try:
                parsed_date = date_parser.parse(timestamp_str, fuzzy=True)
            except Exception:
                parsed_date = None
            current_timestamp = parsed_date
            current_user = user
            current_message = message
            global_members.add(user)
        else:
            # If the line does not match, treat it as a continuation of the previous message
            if current_message is not None:
                current_message += "\n" + line
            else:
                join_exit_events.append(line)
    
    # Save any remaining buffered message after processing all lines
    if current_message is not None and current_user is not None and current_timestamp is not None:
        messages_data.append({
            'Timestamp': current_timestamp,
            'Member Name': current_user,
            'Message': current_message
        })
        user_messages[current_user] += 1
        total_messages += 1

    return {
        'total_messages': total_messages,
        'user_messages': user_messages,
        'join_exit_events': join_exit_events,
        'messages_data': messages_data,
        'global_members': sorted(global_members),
        'left_dates': left_dates
    }

# -------------------------------
# Display Functions for Tables & Charts
# -------------------------------
def display_weekly_messages_table(messages_data, global_members, left_dates):
    """
    Create Table 1: Weekly Message Breakdown.
    For each week (Monday to Sunday) up to the current week, list each member (only if they joined by that week and haven't left before the week ends)
    with their message count (or 0 if inactive), the cumulative follower count (net members), and a column indicating if they left that week.
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

        current_week_start = datetime.now() - timedelta(days=datetime.now().weekday())
        current_week_start = current_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        first_monday = df['Week Start'].min()
        weeks = pd.date_range(start=first_monday, end=current_week_start, freq='W-MON')

        # Compute each member's join date (first message timestamp)
        member_join_dates = df.groupby('Member Name')['Timestamp'].min().to_dict()

        rows = []
        week_counter = 0  # Initialize counter to 0 so that the first week becomes Week 1
        cumulative_added = set()
        cumulative_left = set()

        for week_start in weeks:
            week_counter += 1  # Increment counter at the start of each week iteration
            week_end = week_start + timedelta(days=6)
            eligible_members = [m for m, join_date in member_join_dates.items() if join_date <= week_end]
            cumulative_added.update(eligible_members)
            members_left = [m for m, left_date in left_dates.items() if left_date <= week_end]
            cumulative_left.update(members_left)
            net_members = len(cumulative_added) - len(cumulative_left)
            for member in sorted(eligible_members):
                left_this_week = left_dates[member].strftime("%d %b %Y") if member in left_dates and left_dates[member] <= week_end else ""
                count = df[(df['Week Start'] == week_start) & (df['Member Name'] == member)].shape[0]
                rows.append({
                    'Week': f"Week {week_counter}",
                    'Week Duration': f"{week_start.strftime('%d %b %Y')} - {week_end.strftime('%d %b %Y')}",
                    'Member Name': member,
                    'Number of Messages Sent': count,
                    'Net Member Count': net_members,
                    'Left This Week': left_this_week
                })

        weekly_df = pd.DataFrame(rows)
        st.markdown("### Table 1: Weekly Message Breakdown")
        st.dataframe(weekly_df)
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

        def compute_membership_duration(row):
            join_date = row['first_message']
            member = row['Member Name']
            end_date = left_dates.get(member, current_week_start)
            duration = (end_date - join_date).days / 7
            return max(int(round(duration)), 0)

        grouped['Membership Duration (Weeks)'] = grouped.apply(compute_membership_duration, axis=1)
        grouped['Avg. Weekly Messages'] = grouped.apply(
            lambda row: round(row['total_messages'] / max(row['Membership Duration (Weeks)'], 1), 2),
            axis=1
        )
        grouped['Group Activity Status'] = grouped['total_messages'].apply(
            lambda x: 'Active' if x > 0 else 'Inactive'
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
