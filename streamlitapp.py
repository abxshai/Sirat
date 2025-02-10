import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from groq import Groq

def get_llm_reply(client, prompt, word_placeholder):
    """Get an LLM summary reply using the Groq API."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": ("Analyze the chat log, and summarize key details such as "
                                "the highest message sender, people who joined the group, "
                                "and joining/exiting trends on a weekly or monthly basis, mention the inactive members' names with a message count, take zero if none, display everything in a tabular format")
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

def display_weekly_messages_table(messages_data, global_members):
    if not messages_data:
        st.warning("No data available for weekly messages.")
        return
    
    df = pd.DataFrame(messages_data)
    
    required_columns = {'Timestamp', 'Member Name'}
    if not required_columns.issubset(df.columns):
        st.error(f"Error: Missing required columns: {required_columns - set(df.columns)}")
        return
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    
    df['Week Start'] = df['Timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
    
    min_week_start = df['Week Start'].min()
    max_week_start = df['Week Start'].max()
    weeks = pd.date_range(start=min_week_start, end=max_week_start, freq='W-MON')
    
    rows = []
    week_counter = 1
    
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=6)
        week_mask = (df['Week Start'] == week_start)
        week_messages = df[week_mask]
        
        for member in sorted(global_members):
            count = week_messages[week_messages['Member Name'] == member].shape[0] if not week_messages.empty else 0
            rows.append({
                'Week': f"Week {week_counter}",
                'Week Duration': f"{week_start.strftime('%d %b %Y')} - {week_end.strftime('%d %b %Y')}",
                'Member Name': member,
                'Number of Messages Sent': count
            })
        week_counter += 1
    
    weekly_df = pd.DataFrame(rows)
    
    if weekly_df.empty:
        st.warning("No data available for the chart.")
        return
    
    st.markdown("### Weekly Message Breakdown")
    st.dataframe(weekly_df)
    
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

client = Groq(api_key=st.secrets["API_KEY"])
stats = {
    'messages_data': [
        {'Timestamp': '2024-02-01 12:34:56', 'Member Name': 'User1', 'Message': 'Hello'},
        {'Timestamp': '2024-02-02 15:45:30', 'Member Name': 'User2', 'Message': 'How are you?'},
        {'Timestamp': '2024-02-08 09:20:15', 'Member Name': 'User1', 'Message': 'Good morning!'}
    ],
    'global_members': ['User1', 'User2', 'User3']
}

display_weekly_messages_table(stats['messages_data'], stats['global_members'])

st.markdown("### LLM Summary of Chat Log")
if st.button("Generate Summary"):
    with st.spinner("Analyzing chat log..."):
        top_users = {d['Member Name']: 0 for d in stats['messages_data']}
        snippet_events = stats['messages_data'][:20]
        prompt = (f"Summarize the chat log with these key points:\n"
                  f"- Top message senders: {top_users}\n"
                  f"- Sample messages (first 20): {snippet_events}\n")
        word_placeholder = st.empty()
        get_llm_reply(client, prompt, word_placeholder)
