import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from Utils.lc_model import get_lc_graph, get_respond_from_graph

load_dotenv()
# import os
# print(os.environ["OPENAI_API_KEY"])

st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Chatbot")

#region app config
# def get_response(user_query, chat_history):

#     template = """
#     You are a helpful assistant. Answer the following questions considering the history of the conversation:

#     Chat history: {chat_history}

#     User question: {user_question}
#     """

#     prompt = ChatPromptTemplate.from_template(template)

#     llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
        
#     chain = prompt | llm | StrOutputParser()
    
#     return chain.invoke({
#         "chat_history": chat_history,
#         "user_question": user_query,
#     })

graph = get_lc_graph()

def get_response(user_query, chat_history):
      """get string respond, currently ignoring chat history"""
      message = get_respond_from_graph(user_query, graph)
      return message
# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.write(response)

    st.session_state.chat_history.append(AIMessage(content=response))
#endregion