import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# your modules
from rag import search_knowledge
from memory import load_memory, update_preference
from tools import get_battery_status, set_temperature, get_vehicle_health , open_sunroof

# ---------------------------
# 🔐 ENV
# ---------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
MODEL = os.getenv("MODEL")

# ---------------------------
# 🧠 LLM (ONCE ONLY)
# ---------------------------
llm = ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base=API_BASE,
    model_name=MODEL,
    streaming=True
).bind_tools([
    get_battery_status,
    set_temperature,
    get_vehicle_health,
    open_sunroof,
])

# ---------------------------
# 🧾 STATE
# ---------------------------
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------
# 🧠 NODE 1: MEMORY EXTRACTION
# (your extract_memory logic)
# ---------------------------
def memory_node(state: State):
    user_input = state["messages"][-1].content

    prompt = f"""
    Extract user preference from this sentence.

    Return JSON:
    {{"key": "category", "value": "info"}}

    If nothing important, return null.

    Sentence: "{user_input}"
    """

    try:
        res = llm.invoke([("user", prompt)])
        content = res.content.strip()

        data = json.loads(content)

        if data and "key" in data:
            update_preference(data["key"], data["value"])
    except:
        pass

    return {}


# ---------------------------
# 🔍 NODE 2: RETRIEVAL (R
# ---------------------------
def retrieve_node(state: State):
    user_input = state["messages"][-1].content

    retrieved = search_knowledge(user_input)
    context = "\n".join(retrieved)

    if context.strip():
        return {
            "messages": [
                SystemMessage(content=f"Vehicle knowledge:\n{context}")
            ]
        }

    return {}


# ---------------------------
# 💾 NODE 3: LOAD MEMORY
# ---------------------------
def load_memory_node(state: State):
    memory = load_memory()

    if memory:
        return {
            "messages": [
                SystemMessage(content=f"User preferences: {memory}")
            ]
        }

    return {}


# ---------------------------
# 🤖 NODE 4: AGENT
# ---------------------------
def agent_node(state: State):
    return {
        "messages": [llm.invoke(state["messages"])]
    }   


# ---------------------------
# 🔀 ROUTING (same as your if tool_calls)
# ---------------------------
def should_continue(state: State):
    last = state["messages"][-1]

    if last.tool_calls:
        return "tools"
    return END


# ---------------------------
# 🧱 GRAPH
# ---------------------------
workflow = StateGraph(State)

workflow.add_node("memory", memory_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("load_memory", load_memory_node)
workflow.add_node("agent", agent_node)

workflow.add_node(
    "tools",
    ToolNode([
        get_battery_status,
        set_temperature,
        get_vehicle_health,
        open_sunroof,
    ])
)

# FLOW (this replaces your entire while-loop logic)
workflow.set_entry_point("memory")

workflow.add_edge("memory", "retrieve")
workflow.add_edge("retrieve", "load_memory")
workflow.add_edge("load_memory", "agent")

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()


# ---------------------------
# ▶️ RUN (streaming stays simple)
# ---------------------------
if __name__ == "__main__":
    print(f"--- Audi Assistant (LangGraph) ---")

    while True:
        user_input = input("\nDriver: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        print("Assistant: ", end="", flush=True)

        for msg, meta in app.stream(
            {"messages": [("user", user_input)]},
            stream_mode="messages"
        ):
            if meta.get("langgraph_node") == "agent":
                if msg.content:
                    print(msg.content, end="", flush=True)

        print()