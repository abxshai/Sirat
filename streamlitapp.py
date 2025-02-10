import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
import os
import re
import tempfile
from collections import Counter
from groq import Groq
from dateutil import parser as date_parser
from datetime import datetime
import chardet

# -------------------------------
# Enhanced Helper Functions
# -------------------------------
def detect_file_encoding(file_path):
    """
    Detect the encoding of a file using chardet library.
    Returns the most likely encoding.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def clean_member_name(name):
    """
    Enhanced member name cleaning with better phone number detection
    and international format support.
    """
    cleaned = name.strip()
    # Handle various phone number formats
    phone_patterns = [
        r'(\+\d{1,3}\s?)?\d{10,}',  # International numbers
        r'\d{3}[-.]?\d{3}[-.]?\d{4}',  # US format
        r'\+\d{1,3}\s\d{1,4}\s\d{4,}',  # International with spaces
    ]
    
    for pattern in phone_patterns:
        if re.search(pattern, cleaned):
            digits = re.sub(r'\D', '', cleaned)
            return f"User {digits[-4:]}"
    
    # Remove emojis and special characters
    cleaned = re.sub(r'[^\w\s\-\']', '', cleaned)
    return cleaned.strip()

def parse_timestamp(timestamp_str):
    """
    Enhanced timestamp parsing with support for multiple WhatsApp date formats.
    """
    try:
        # Common WhatsApp date formats
        formats = [
            '%d/%m/%y, %H:%M',  # 31/12/23, 14:30
            '%d/%m/%Y, %H:%M',  # 31/12/2023, 14:30
            '%m/%d/%y, %H:%M',  # 12/31/23, 14:30
            '%m/%d/%Y, %H:%M',  # 12/31/2023, 14:30
            '%d.%m.%y, %H:%M',  # 31.12.23, 14:30
            '%Y-%m-%d, %H:%M',  # 2023-12-31, 14:30
            '%d/%m/%y, %I:%M %p',  # 31/12/23, 02:30 PM
            '%d/%m/%Y, %I:%M %p',  # 31/12/2023, 02:30 PM
        ]
        
        # Try exact formats first
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # If exact formats fail, try dateutil parser
        return date_parser.parse(timestamp_str, fuzzy=True)
    
    except Exception:
        return None

# -------------------------------
# Enhanced Message Pattern Detection
# -------------------------------
def detect_message_pattern(first_few_lines):
    """
    Detect the message pattern format from the first few lines of the chat.
    Returns the appropriate regex pattern.
    """
    # Common patterns in different WhatsApp exports
    patterns = [
        # Standard format with 24-hour time
        (r'^(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s(.*)$',
         'standard_24h'),
        # Format with AM/PM
        (r'^(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm]))\s*-\s*([^:]+):\s(.*)$',
         'standard_12h'),
        # Format with different date separator
        (r'^(\d{4}-\d{2}-\d{2},?\s*\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s(.*)$',
         'iso_format')
    ]
    
    for line in first_few_lines:
        for pattern, pattern_type in patterns:
            if re.match(pattern, line.strip()):
                return pattern, pattern_type
    
    # Default to most flexible pattern if no match found
    return patterns[0][0], 'standard_24h'

# -------------------------------
# Enhanced Chat Log Parser
# -------------------------------
def parse_chat_log(file_path):
    """
    Enhanced chat log parser with better format detection and error handling.
    """
    if not file_path or not os.path.exists(file_path):
        st.error("Chat log file not found.")
        return None
    
    try:
        # Detect file encoding
        encoding = detect_file_encoding(file_path)
        if not encoding:
            encoding = 'utf-8'
        
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
            
        # Split into lines and detect message pattern
        lines = content.splitlines()
        first_few_lines = [line for line in lines[:20] if line.strip()]
        message_pattern, pattern_type = detect_message_pattern(first_few_lines)
        
        # Initialize data structures
        messages_data = []
        user_messages = Counter()
        join_exit_events = []
        global_members = set()
        
        # Enhanced system message patterns
        system_patterns = {
            'join': r'(.+) joined using this group\'s invite link',
            'add': r'(.+) added (.+)',
            'leave': r'(.+) left',
            'remove': r'(.+) removed (.+)',
            'subject': r'(.+) changed the subject from "(.+)" to "(.+)"',
            'icon': r'(.+) changed this group\'s icon',
            'description': r'(.+) changed the group description',
            'number': r'(.+) changed their phone number',
            'security': r'Messages and calls are end-to-end encrypted'
        }
        
        current_message = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match as a new message
            match = re.match(message_pattern, line)
            
            if match:
                # Process previous message if exists
                if current_message:
                    messages_data.append(current_message)
                    current_message = []
                
                # Process new message
                timestamp_str, user, message = match.groups()
                timestamp = parse_timestamp(timestamp_str)
                
                if timestamp:
                    user = clean_member_name(user)
                    current_message = [timestamp_str, user, message]
                    user_messages[user] += 1
                    global_members.add(user)
            else:
                # Check if it's a system message
                is_system_message = False
                for event_type, pattern in system_patterns.items():
                    if re.match(pattern, line):
                        join_exit_events.append((event_type, line))
                        is_system_message = True
                        break
                
                # If not a system message, append to current message
                if not is_system_message and current_message:
                    current_message[2] += f"\n{line}"
        
        # Add last message if exists
        if current_message:
            messages_data.append(current_message)
        
        return {
            'total_messages': len(messages_data),
            'user_messages': user_messages,
            'join_exit_events': join_exit_events,
            'messages_data': messages_data,
            'global_members': sorted(global_members),
            'pattern_type': pattern_type
        }
    
    except Exception as e:
        st.error(f"Error parsing chat log: {str(e)}")
        return None

# [Rest of the code remains the same...]
