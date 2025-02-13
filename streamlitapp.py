import pandas as pd
import streamlit as st
from PIL import Image
import helper
import preprocessor
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components  # Import Streamlit

image1 = Image.open('download.png')
st.sidebar.image(image1,width = 80)
st.sidebar.title("Whatsapp Chat Analyzer")


uploaded_file = st.sidebar.file_uploader("Choose a file ")

if uploaded_file is None:
    key = '<p style="font-family:Roboto; color:#fff; font-size: 50px; font-weight: bold; text-align: center">Upload your chats (format : DD/MM/YY format) to view analysis</p>'
    st.markdown(key, unsafe_allow_html = True)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    st.dataframe(df)
 
    st.download_button(
        label="Download data as CSV",
        data=df.to_csv(),
        file_name='whatsapp_data_output.csv',
        mime='text/csv',
        help = "When You Click On Download Button Your WhatsApp Text Data Will Be Converted To Clean Downloadable CSV File"
    )
    # fetch unique user
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, 'Overall')

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)
    
    
    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        key = '<u><p style="font-family:Roboto; color:#5e76c4; font-size: 50px; font-weight: bold">Top Chat Statistics</p></u>'
        st.markdown(key, unsafe_allow_html = True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            key = '<p style="font-family:Roboto; color:#12ca7d; font-size: 30px; font-weight: bold">Total Messages</p>'
            st.markdown(key, unsafe_allow_html = True)
            st.title(num_messages)
        with col2:
            key = '<p style="font-family:Roboto; color:#12ca7d; font-size: 30px; font-weight: bold">Total<br>Words</p>'
            st.markdown(key, unsafe_allow_html = True)
            st.title(words)
        with col3:
            key = '<p style="font-family:Roboto; color:#12ca7d; font-size: 30px; font-weight: bold">Media Shared</p>'
            st.markdown(key, unsafe_allow_html = True)
            st.title(num_media_messages)
        with col4:
            key = '<p style="font-family:Roboto; color:#12ca7d; font-size: 30px; font-weight: bold">Linked Shared</p>'
            st.markdown(key, unsafe_allow_html = True)
            st.title(num_links)

        # monthly timeline
        timeline = helper.monthly_timeline(selected_user, df)
        monthly_time = '<br><br><u><p style="font-family:Roboto; color:#ca124d; font-size: 40px; font-weight: bold">Monthly Timeline</p></u>'
        st.markdown(monthly_time, unsafe_allow_html = True)
        st.markdown('<p style="font-family:Roboto fax; color:#ffa81a; font-size: 15px; font-weight: bold">This status shows total number of chats per month in group or personal.</p>',unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize = (10,4))
        plt.plot(timeline['time'], timeline['message'], color='green', linewidth = 3.0)
        plt.title("Monthly Chats")
        plt.xlabel("Months")
        plt.ylabel("Number of chats")
        plt.xticks(rotation='vertical')
        for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
                ax.spines[axis].set_color("orange")
                ax.spines[axis].set_zorder(5)
        st.pyplot(fig)
        

        # daily timeline
        daily_timeline = helper.daily_timeline(selected_user, df)
        daily_time = '<br><br><u><p style="font-family:georgia; color:#ca124d; font-size: 40px; font-weight: bold">Daily Timeline</p></u>'
        st.markdown(daily_time, unsafe_allow_html = True)
        st.markdown('<p style="font-family:lucida fax; color:#ffa81a; font-size: 15px; font-weight: bold">This stats shows total number of chats per day in group or personal.</p>',unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize = (10,4))
        plt.plot(daily_timeline['only_date'], daily_timeline['message'], color='red', linewidth = 2.5)
        plt.title("No. of chats per month")
        plt.xlabel("Days")
        plt.ylabel("Number of chats")
        plt.xticks(rotation='vertical')
        for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
                ax.spines[axis].set_color("orange")
                ax.spines[axis].set_zorder(5)
        st.pyplot(fig)

        # weekly activity
        activity_map = '<br><br><u><p style="font-family:Roboto; color:#ca124d; font-size: 40px; font-weight: bold">Activity Map</p></u>'
        st.markdown(activity_map, unsafe_allow_html = True)
        st.markdown('<p style="font-family:Roboto fax; color:#ffa81a; font-size: 15px; font-weight: bold">This stats shows which day and month of your chats are busy</p>',unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            most_busy_day = '<p style="font-family:georgia; color:#12ca7d; font-size: 30px; font-weight: bold">Most Busy Day</p>'
            st.markdown(most_busy_day, unsafe_allow_html = True)
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values,color = 'pink')
            plt.xticks(rotation='vertical')
            plt.title("Most busy day in a week")
            plt.xlabel("Days")
            plt.ylabel("Number of chats")
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
                ax.spines[axis].set_color("orange")
                ax.spines[axis].set_zorder(5)
            st.pyplot(fig)

        with col2:
            most_busy_month = '<p style="font-family:Roboto; color:#12ca7d; font-size: 30px; font-weight: bold">Most Busy Month</p>'
            st.markdown(most_busy_month, unsafe_allow_html = True)
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='purple')
            plt.xticks(rotation='vertical')
            plt.title("Most busy month")
            plt.xlabel("Months")
            plt.ylabel("Number of chats")
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
                ax.spines[axis].set_color("orange")
                ax.spines[axis].set_zorder(5)
            st.pyplot(fig)

        # Weekly hours' activity map
        weekly_hours_activity_map = '<br><br><u><p style="font-family:Roboto; color:#ca124d; font-size: 40px; font-weight: bold">Weekly Hours Activity Map</p></u>'
        st.markdown(weekly_hours_activity_map, unsafe_allow_html = True)
        st.markdown('<p style="font-family:Roboto fax; color:#ffa81a; font-size: 15px; font-weight: bold">This map shows the active hours of the  group across the week</p>',unsafe_allow_html=True)
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots(figsize= (8,4))
        ax = sns.heatmap(user_heatmap)
        plt.title("Weekly Hours Activity Map")
        plt.xlabel("Time")
        plt.ylabel("Number of chats")
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
            ax.spines[axis].set_color("orange")
            ax.spines[axis].set_zorder(5)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            most_busy_user = '<br><br><u><p style="font-family:Roboto; color:#ca124d; font-size: 40px; font-weight: bold">Most Busy Users</p></u>'
            st.markdown(most_busy_user, unsafe_allow_html = True)
            st.markdown('<p style="font-family:Roboto; color:#ffa81a; font-size: 15px; font-weight: bold">This stats shows who is the most busy user in the group</p>',unsafe_allow_html=True)
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots(figsize= (8,5))

            # col1, col2 = st.columns(2)

            # with col1:
            ax.bar(x.index, x.values, color='cyan')
            plt.xticks(rotation='vertical')
            plt.title("Monthly Timeline")
            plt.xlabel("users")
            plt.ylabel("Number")
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
                ax.spines[axis].set_color("orange")
                ax.spines[axis].set_zorder(5)
            st.pyplot(fig)
            # with col2:
                # st.dataframe(new_df)

        # Word Cloud

        word_cloud = '<br><br><u><p style="font-family:Roboto; color:#ca124d; font-size: 40px; font-weight: bold">Word Cloud</p></u>'
        st.markdown(word_cloud, unsafe_allow_html = True)
        st.markdown('<p style="font-family:Roboto; color:#ffa81a; font-size: 15px; font-weight: bold">This image shows frequently used words in the chats by user or a group</p>',unsafe_allow_html=True)
        df_wc = helper.create_worldcloud(selected_user, df)
        fig, ax = plt.subplots(figsize = (8,4))
        ax.imshow(df_wc)
        plt.title("Word Cloud")
        for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
                ax.spines[axis].set_color("orange")
                ax.spines[axis].set_zorder(5)
        st.pyplot(fig)

        # most common words
        
        most_used_words = '<br><br><u><p style="font-family:Roboto; color:#ca124d; font-size: 40px; font-weight: bold">Most used words</p></u>'
        st.markdown(most_used_words, unsafe_allow_html = True)
        st.markdown('<p style="font-family:Roboto fax; color:#ffa81a; font-size: 15px; font-weight: bold">This stats show most used words in the chats.</p>',unsafe_allow_html=True)
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots(figsize= (8,4))

        ax.bar(most_common_df[0], most_common_df[1],color="brown")
        plt.xticks(rotation='vertical')
        plt.title("Most words used")
        plt.xlabel("Total no. of words")
        plt.ylabel("Words")
        for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(3)
                ax.spines[axis].set_color("orange")
                ax.spines[axis].set_zorder(5)
        st.pyplot(fig)

   
st.sidebar.markdown('<br><br>', unsafe_allow_html=True)

if st.sidebar.button("About us"):
    st.title('Creator : ')
    key = '<br><p style="font-family:Roboto; color:#ca124d; font-size: 40px; font-weight: bold">HETANSI PATEL</p>'
    st.markdown(key, unsafe_allow_html = True)
    st.markdown('<p style="font-family:Roboto; color:blue; font-size: 25px; font-weight: bold">Id : 20DIT056 <br> From : Charotar University of Science and Technology<br> Python Project</p>',unsafe_allow_html = True)
    
