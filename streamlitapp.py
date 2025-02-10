import streamlit as st
import pandas as pd
import zipfile
import os
import re
import tempfile
import matplotlib.pyplot as plt
from collections import Counter
from dateutil import parser as date_parser

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
# Function to Extract ZIP File
# -------------------------------
def extract_zip(uploaded_file):
    """Extract chat log from the uploaded zip file."""
    if not uploaded_file:
        return None

    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "chat.zip")

        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        for file in os.listdir(temp_dir):
            if file.endswith(".txt"):
                return os.path.join(temp_dir, file)

        st.error("No .txt file found in the zip archive.")
        return None

    except Exception as e:
        st.error(f"Error extracting zip file: {e}")
        return None

# -------------------------------
# Function to Parse WhatsApp Chat
# -------------------------------
def parse_chat_log(file_path):
    """Parse WhatsApp chat log and extract message details."""
    if not file_path or not os.path.exists(file_path):
        st.error("Chat log file not found.")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            chats = file.readlines()

    except Exception as e:
        st.error(f"Error reading chat log: {e}")
        return None

    messages_data = []
    user_messages = Counter()
    global_members = set()

    message_pattern = re.compile(
        r'^(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\s*-\s*(.*?):\s(.*)$'
    )

    for line in chats:
        line = line.strip()
        match = re.match(message_pattern, line)

        if match:
            timestamp_str, user, message = match.groups()
            user = clean_member_name(user)

            try:
                timestamp = date_parser.parse(timestamp_str, fuzzy=True)
                messages_data.append({"Timestamp": timestamp, "Member Name": user, "Message": message})
                user_messages[user] += 1
                global_members.add(user)
            except Exception:
                continue

    return {
        "messages_data": messages_data,
        "user_messages": user_messages,
        "global_members": sorted(global_members),
    }

# -------------------------------
# Function to Display Weekly Message Breakdown (Table 1)
# -------------------------------
def display_weekly_messages_table(messages_data, global_members):
    """Show the weekly breakdown of messages, ensuring inactive members appear with 0 messages."""
    if not messages_data:
        st.warning("No messages found.")
        return

    df = pd.DataFrame(messages_data)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)

    df["Week Start"] = (df["Timestamp"] - pd.to_timedelta(df["Timestamp"].dt.weekday, unit="D")).dt.normalize()
    
    if df.empty:
        st.warning("No messages to display.")
        return

    min_week = df["Week Start"].min()
    max_week = df["Week Start"].max()
    weeks = pd.date_range(start=min_week, end=max_week, freq="W-MON")

    rows = []
    week_counter = 1

    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)
        week_messages = df[df["Week Start"] == week_start]

        for member in sorted(global_members):
            count = week_messages[week_messages["Member Name"] == member].shape[0] if not week_messages.empty else 0
            rows.append({
                "Week": f"Week {week_counter}",
                "Week Duration": f"{week_start.strftime('%d %b %Y')} - {week_end.strftime('%d %b %Y')}",
                "Member Name": member,
                "Number of Messages Sent": count
            })
        week_counter += 1

    weekly_df = pd.DataFrame(rows)
    st.markdown("### Table 1: Weekly Message Breakdown")
    st.dataframe(weekly_df)

# -------------------------------
# Function to Display Bar Chart of Messages Per User
# -------------------------------
def display_total_messages_chart(user_messages):
    """Show total messages sent by each user in a bar chart."""
    if not user_messages:
        st.warning("No messages to display.")
        return

    df = pd.DataFrame(user_messages.items(), columns=["Member Name", "Messages"])
    fig, ax = plt.subplots()
    ax.bar(df["Member Name"], df["Messages"], color="skyblue")
    ax.set_xlabel("Member Name")
    ax.set_ylabel("Message Count")
    ax.set_title("Total Messages Sent by Each User")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# -------------------------------
# Main Streamlit App
# -------------------------------
st.title("WhatsApp Chat Log Analyzer")

uploaded_file = st.file_uploader("Upload a WhatsApp chat zip file", type="zip")

if uploaded_file:
    chat_log_path = extract_zip(uploaded_file)

    if chat_log_path:
        stats = parse_chat_log(chat_log_path)

        if stats:
            st.success("Chat log parsed successfully!")

            # Display Table 1: Weekly Message Breakdown
            display_weekly_messages_table(stats["messages_data"], stats["global_members"])

            # Display Bar Chart
            display_total_messages_chart(stats["user_messages"])
        else:
            st.error("Error parsing chat log.")
    else:
        st.error("No chat log file found in the zip archive.")
