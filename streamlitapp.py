import streamlit as st
import pandas as pd
import plotly.express as px
import re
from collections import Counter
from datetime import datetime
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
    return cleaned

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

    user_messages = Counter()
    messages_data = []
    global_members = set()
    join_exit_events = []
    left_dates = {}
    total_messages = 0

    # Regex pattern for WhatsApp messages (handles both 12-hour and 24-hour formats)
    message_pattern = re.compile(
        r'^(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}),?\s*(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap][Mm])?)?\s*-\s*(.*?):\s(.*)$'
    )
    
    left_pattern = re.compile(r'^(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}),?\s*(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap][Mm])?)?\s*-\s*(.*?) left')

    ongoing_message = None
    ongoing_user = None
    ongoing_timestamp = None
    
    for line in chats:
        line = line.strip()
        if not line:
            continue
        
        left_match = re.match(left_pattern, line)
        if left_match:
            timestamp_str, time_part, member = left_match.groups()
            timestamp_str = f"{timestamp_str} {time_part}" if time_part else timestamp_str
            member = clean_member_name(member)
            try:
                left_date = date_parser.parse(timestamp_str, fuzzy=True)
            except Exception:
                left_date = None
            if left_date and (member not in left_dates or left_date > left_dates[member]):
                left_dates[member] = left_date
            continue
        
        match = re.match(message_pattern, line)
        if match:
            if ongoing_message:
                messages_data.append({
                    "Timestamp": ongoing_timestamp,
                    "Member Name": ongoing_user,
                    "Message": ongoing_message
                })
                user_messages[ongoing_user] += 1
                total_messages += 1
            
            timestamp_str, time_part, user, message = match.groups()
            timestamp_str = f"{timestamp_str} {time_part}" if time_part else timestamp_str
            try:
                parsed_date = date_parser.parse(timestamp_str, fuzzy=True)
            except Exception:
                parsed_date = None
            
            user = clean_member_name(user)
            global_members.add(user)
            ongoing_timestamp, ongoing_user, ongoing_message = parsed_date, user, message
            
        elif ongoing_message is not None:
            ongoing_message += "\n" + line  # Handle multi-line messages
    
    if ongoing_message:
        messages_data.append({
            "Timestamp": ongoing_timestamp,
            "Member Name": ongoing_user,
            "Message": ongoing_message
        })
        user_messages[ongoing_user] += 1
        total_messages += 1
    
    return {
        "messages_data": messages_data,
        "user_messages": user_messages,
        "global_members": sorted(global_members),
        "join_exit_events": join_exit_events,
        "left_dates": left_dates
    }
