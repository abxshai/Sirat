import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def display_weekly_messages_table(messages_data, global_members):
    # Check if messages_data exists and is not empty
    if not messages_data:
        st.warning("No data available for weekly messages.")
        return
    
    df = pd.DataFrame(messages_data)
    
    # Ensure 'Timestamp' exists and convert it safely to datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)  # Remove rows where conversion failed
        
        # Compute the Week Start
        df['Week Start'] = (df['Timestamp'] - pd.to_timedelta(df['Timestamp'].dt.weekday, unit='D')).dt.normalize()
        
        # Aggregate message counts per week
        weekly_counts = df.groupby('Week Start').size().reset_index(name='Message Count')
    else:
        st.error("Error: 'Timestamp' column is missing from data.")
        return
    
    # Ensure global_members is valid
    if not global_members:
        st.warning("No data available for global members.")
        return
    
    # Display DataFrame in Streamlit
    if weekly_counts.empty:
        st.warning("No data available for Member Statistics.")
    else:
        st.write("### Weekly Message Counts")
        st.dataframe(weekly_counts)
        
        # Plot bar graph
        fig, ax = plt.subplots()
        ax.bar(weekly_counts['Week Start'].dt.strftime('%Y-%m-%d'), weekly_counts['Message Count'], color='skyblue')
        ax.set_xlabel("Week Start")
        ax.set_ylabel("Message Count")
        ax.set_title("Weekly Messages")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Example usage
stats = {
    'messages_data': [
        {'Timestamp': '2024-02-01 12:34:56', 'Message': 'Hello'},
        {'Timestamp': '2024-02-02 15:45:30', 'Message': 'How are you?'},
        {'Timestamp': '2024-02-08 09:20:15', 'Message': 'Good morning!'}
    ],
    'global_members': ['User1', 'User2']
}

display_weekly_messages_table(stats['messages_data'], stats['global_members'])
