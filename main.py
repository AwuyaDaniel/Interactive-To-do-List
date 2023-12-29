# import time
import streamlit as st
from DataBase import DBConnector
from app import ChatBot, LLMChatBot
import os
# # Add a placeholder
# # latest_iteration = st.empty()
# # bar = st.progress(0)
# #
# # for i in range(100):
# #   # Update the progress bar with each iteration.
# #   latest_iteration.text(f'Iteration {i+1}')
# #   bar.progress(i + 1)
# #   time.sleep(0.1)

messages = []

if __name__ == "__main__":
    # Add a slider to the sidebar:

    # App framework
    st.title('🦜🔗 An Interactive AI To-do List')
    st.write("an AI chatbots that creates and adjusts a to-do list with a calendar system.")
    st.write("This AI chatbot is able to create and add items to a to-do list, as well as adjust the to-do list as necessary.")
    # prompt = st.text_input('Plug in your prompt here')

    # Show stuff to the screen if there's a prompt
    add_slider = st.sidebar.button(
        "To Do Schedule"
    )
    add_slider_2 = st.sidebar.button(
        "Ai Chat bot"
    )
    d_slider_3 = st.sidebar.button(
        "Update"
    )
    data_base = DBConnector()
    data_base.main_connection_to_db('aidb')
    # data_base.create_message_table()
    # data_base.create_response_table()
    # data_base.create_todos_table()
    # data_base.create_todo_item_table()
    if d_slider_3:
        ChatBot().update_todo(data_base)
    if add_slider:
        st.write("Here Is a List Of All Your todo")
        ChatBot().get_todo(data_base)

    else:
        st.write("This is a chatbot that helps create a todo list from your input")
        ChatBot().main(data_base)

