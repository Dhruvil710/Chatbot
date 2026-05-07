import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Import your tools from tools.py
from tools import get_battery_status, set_temperature, get_vehicle_health

# Load credentials
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
MODEL = os.getenv("MODEL") # gpt-4.1-mini

# 1. RETRIEVER (Search manual for car knowledge)
def get_car_context(user_query: str):
    """Simple text search to replace the embedding RAG for custom APIs."""
    with open("car_knowledge.txt") as f:
        knowledge = f.read().splitlines()
    
    # Simple word-matching search
    relevant = [line for line in knowledge if any(word.lower() in line.lower() for word in user_query.split())]
    return "\n".join(relevant) if relevant else "No specific manual entry found."

# 2. DEFINE THE STATE
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 3. DEFINE THE NODES
def chatbot(state: State):
    # Search car manual
    user_input = state["messages"][-1].content
    context = get_car_context(user_input)
    
    # Setup LLM with STREAMING ENABLED
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        model_name=MODEL,
        streaming=True # <--- Key for token-by-token feel
    ).bind_tools([get_battery_status, set_temperature, get_vehicle_health])
    
    sys_msg = SystemMessage(content=f"You are a premium Audi assistant. Context: {context}")
    
    # Return message from LLM
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}

# 4. BUILD THE GRAPH
workflow = StateGraph(State)
workflow.add_node("agent", chatbot)
workflow.add_node("tools", ToolNode([get_battery_status, set_temperature, get_vehicle_health]))

workflow.set_entry_point("agent")

def should_continue(state: State):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# 5. RUN WITH TOKEN STREAMING
if __name__ == "__main__":
    print(f"--- Audi Assistant Online ({MODEL}) ---")
    
    while True:
        user_input = input("\nDriver: ")
        if user_input.lower() in ["exit", "quit"]: break

        print("Assistant: ", end="", flush=True)

        # Iterate through message events as they happen
        for msg, metadata in app.stream(
            {"messages": [("user", user_input)]}, 
            stream_mode="messages" # <--- Tells LangGraph to stream tokens
        ):
            # Only print content from the "agent" node to avoid printing raw tool JSON
            if metadata.get("langgraph_node") == "agent":
                if msg.content:
                    print(msg.content, end="", flush=True)
        
        print() # New line when finished