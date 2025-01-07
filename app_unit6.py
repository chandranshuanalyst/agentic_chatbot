# Import necessary modules and setup for FastAPI, LangGraph, and LangChain
from fastapi import FastAPI  # FastAPI framework for creating the web application
from pydantic import BaseModel  # BaseModel for structured data data models
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables (Make sure to set up your GROQ_API_KEY)
load_dotenv()

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    groq_api_key = os.getenv('GROQ_API_KEY')
)


DB_URI = os.getenv('DB_URI')
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}


conn=Connection.connect(DB_URI, **connection_kwargs)
checkpointer = PostgresSaver(conn)

# Create the table schema (only needs to be done once)
table_name = "Chatbot_history"

class State(TypedDict):
    question: Annotated[list,add_messages]
    answer: Annotated[list,add_messages]
    summary: str
    context: list
    
    
def web_search(state):
    
    ''' Search for a topic over the web using Tavily web search'''
    
    question= state["question"][-1].content
     # Search
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(question)
    url=[]
    content=[]
     # Format
    url = [doc["url"] for doc in search_docs]
    content=[doc["content"] for doc in search_docs]
    
    formatted_search_docs=[content,url]    

    return {"context": formatted_search_docs} 

def chatbot_withcontext(state): 
    
    """ Node to answer a question """
     
    # Get summary if it exists
    summary = state.get("summary", "")
    
    # Get answer if it exists
    answer = state.get("answer",{})

    question = state["question"][-1]
    context = state.get("context", {})[0]
    source = state.get("context", {})[1]
    
    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = system_message + question.content
    
    else:
        messages = question.content
     # Answer
    answer = llm.invoke([SystemMessage(content=f"""Answer the question,{question}, using this context,{context} given the history of the previous conversations till now is {messages}.Also list the resouces used for answer as a bullet list. The urls to be listed as a part of response are {source}.
                                       The answer should be in a format attached alongside
                                       **Response** Write your response
                                       **Refernces**
                                       [1] https://wiki..""")]+[HumanMessage(content=f"Answer the question.")])
      
    # Append it to state
    return {"answer": [answer], "summary": messages}

def chatbot_with_nocontext(state):
    
    """ Node to answer a question """

    # Get state
    question = state["question"][-1]
    
    # Get summary if it exists
    summary = state.get("summary", "")
    
    # Get answer if it exists
    answer = state.get("answer",{})

    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = system_message + question.content
    
    else:
        messages = question.content
    
    # Answer
    answer = llm.invoke([SystemMessage(content=f"Answer the question,{question} given the history of the previous conversations till now is {messages}.")]+[HumanMessage(content=f"Answer the question.")])
      
    # Append it to state
    return {"answer": [answer], "summary": messages}


def should_search(state):
    ''' Decides wheter the LLM can answer the question correctly without context or a websearch has to be done.'''
    
    question = state["question"][-1]
    
    system_instructions = SystemMessage(content= f""" 
                                        You are given a question {question} that you have to answer correctly.
                                        Acting as a fair judge you have decide if you need or donot need any external tools
                                        to correctly answer the question.
                                        Utilising your judgment ability if you feel the question can be accurately answered without
                                        any tool help then return the response.
                                        In case you have even the slighest doubt on the accuracy of response like whether response
                                        is up to date or the question is around something that has high chances of being updated 
                                        then return the text Need tool help as response.
                                        Make sure if you return a response without tool help the response has to be accurate.                    
                                        """)
    
    answer = llm.invoke([system_instructions]+[HumanMessage(content=f"Answer the question {question}.")])

    
    if "need tool help".lower() in answer.content.lower():
        return "web_search"
    
    return "chatbot_with_nocontext"

def history_conversation(state):
    ''' Node that creates a summary of the pervious conversations '''
    # First, we get any existing summary
    summary = state.get("summary", "")
    answer = state.get("answer",{})[-1]
    
    question = state["question"][-1]
    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Update the summary by taking into account the new messages making sure the updated\n\n"
            "summary is a short and crisp one like a short one."
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = summary_message + answer.content
    response = llm.invoke(messages)
    return {"summary": response.content}
    
def delete_msg(state):
    ''' Retain only the latest 2 questions and answers if the length of the list exceed 4.'''
    
    question = state.get("question",{})
    answer = state.get("answer",{})
    
    # Delete all but the 2 most recent messages
    delete_question = [RemoveMessage(id=m.id) for m in state["question"][:-2]]    
    delete_answer = [RemoveMessage(id=m.id) for m in state["answer"][:-2]]
    
    return{"question": delete_question,"answer": delete_answer}

def should_delete(state):
    ''' Decide if the questions and answers in the state have to be deleted and onlu the latest 2 are to be kept '''
    
    question = state.get("question",{})
    answer = state.get("answer",{})
    
    if(len(question)>4 and len(answer)>4):
        return "delete_msg"
    
    return END   

# Add nodes
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("chatbot_with_nocontext",chatbot_with_nocontext)
builder.add_node("web_search",web_search)
builder.add_node("chatbot_with_tools",chatbot_withcontext)
builder.add_node("history_conversation",history_conversation)
builder.add_node("delete_msg",delete_msg)

# Flow
builder.add_conditional_edges(START,should_search)
builder.add_edge("chatbot_with_nocontext", "history_conversation")
builder.add_edge("web_search","chatbot_with_tools") 
builder.add_edge("chatbot_with_tools", "history_conversation")
builder.add_conditional_edges("history_conversation",should_delete)

# Compile
memory = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Create a thread
config = {"configurable": {"thread_id": "5"}}

# FastAPI application setup with a title
app = FastAPI(title='AI Agentic Chatbot')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501"],  # Allow Streamlit to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the request schema using Pydantic's BaseModel
class RequestState(BaseModel):
    user_question: str  # List of messages in the chat

# Define an endpoint for handling chat requests
@app.post("/chat/")
def chat_endpoint(request: RequestState):
    """
    API endpoint to interact with the chatbot using LangGraph and tools.
    Dynamically selects the model specified in the request.
    """
    
    checkpointer = PostgresSaver(conn)
    result = graph.invoke({"question": [HumanMessage(content=request.user_question)]}, config)

    # Return the result as the response
    return result
