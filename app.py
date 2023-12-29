# Bring in deps
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.llms import HuggingFaceHub
import re
from LoadModel import get_prediction

load_dotenv()  # Load variables from the .env file
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACE_API_TOKEN')


class ChatBot:
    def __init__(self):
        self.input_message = ''
        self.messages = []

    def reformat_task(self, text):
        pattern = r'[a-zA-Z].*'  # Regex pattern to match from the first alphabet to the end of the string
        matches = re.findall(pattern, text)
        if matches:
            list_item = matches[0].strip()  # Get the first match and remove extra spaces
            return list_item

    def create_todo_list(self, database, todo_name, todo_items):
        todo = database.create_todo_record(todo_name)

        for i in todo_items.split('\n'):
            if i is not None and i != "None" and i != '':
                text = self.reformat_task(i)
                data = LLMChatBot().prompt_template(f"'{text}', is the following statement a command, give response only in yes or no format with no definition")
                if "yes" in data[0].lower():
                    database.create_todo_item_record(text, todo[0])

    def get_bot_response(self, user_input):
        # Add your logic to generate bot responses based on user input
        if user_input:
            data = LLMChatBot().prompt_template(user_input)
            return data

    def show_messages(self, data_base):
        st.title("Chatbot")
        all_messages = """SELECT * FROM messages ORDER BY sent_at DESC"""
        responses = """SELECT * FROM responses ORDER BY responded_at DESC"""
        all_messages = data_base.run_customer_query(all_messages)
        responses = data_base.run_customer_query(responses)

        st.write("--- Chat History ---")
        for msg, res in zip(all_messages, responses):
            st.write(f"User: {msg[1]}")
            st.write(f"Response: {res[1]}")
        st.write("--- End of Chat ---")

    def main(self, data_base):
        self.input_message = st.text_input("Enter your message")
        if self.input_message:
            if get_prediction(self.input_message) == "List-related":
                self.update_todo(data_base)
            else:
                message = data_base.create_message_record(self.input_message)
                # print(message)
                bot_response = self.get_bot_response(self.input_message)
                data_base.create_response_record(bot_response[0], message[0])
                self.create_todo_list(data_base, self.input_message, bot_response[0])
                self.show_messages(data_base)

    def get_todo(self, data_base):
        all_todo = """SELECT * FROM Todos ORDER BY created_at DESC"""
        todo_items = """SELECT * FROM TodosItem ORDER BY created_at DESC"""
        all_messages = data_base.run_customer_query(all_todo)
        responses = data_base.run_customer_query(todo_items)
        for i in all_messages:
            st.title(i[1])
            for j in responses:
                if j[2] == i[0]:
                    st.write(j[1])

    def update_todo(self, data_base,):
        command = """
                SELECT * 
                FROM Todos 
                ORDER BY created_at DESC 
                LIMIT 1
                """
        # id = command[0][0]
        current = data_base.run_customer_query(command)
        # todo_items = f"""SELECT * FROM TodosItem WHERE todo_id = {current[0][0]}"""
        # todo_items = data_base.run_customer_query(todo_items)
        bot_response = self.get_bot_response(self.input_message)
        for i in bot_response[0].split('\n'):
            if i is not None and i != "None" and i != '':
                text = self.reformat_task(i)
                data = LLMChatBot().prompt_template(f"'{text}', is the following statement a command, give response only in yes or no format with no definition")
                if "yes" in data[0].lower():
                    data_base.create_todo_item_record(text, current[0][0])



class LLMChatBot:
    def __init__(self):
        self.repo_id = "openchat/openchat-3.5-1210"

        self.llm = HuggingFaceHub(
            repo_id=self.repo_id, model_kwargs={"temperature": 0.6, }
        )

    def prompt_template(self, prompt):
        # Prompt templates
        title_template = PromptTemplate(
            input_variables=['topic'],
            template='{topic} '
        )

        # Memory
        title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
        script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

        # Llms
        title_chain = LLMChain(llm=self.llm, prompt=title_template, verbose=True, output_key='title',
                               memory=title_memory)

        if prompt:
            title = title_chain.run(prompt)
            return title, title_memory.buffer, script_memory.buffer
