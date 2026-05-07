import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# --- 1. STATE DEFINITION ---
class Chatstate(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- 2. TOOL DEFINITIONS ---

# --- DOMAIN: VEHICLE ---
@tool
def get_battery_status():
    """Returns the current Audi battery percentage and range."""
    return {"battery": "78%", "range": "320km"}

@tool
def open_sunroof():
    """Commands the vehicle to open the sunroof."""
    return {"message": "Sunroof opened"}

@tool
def set_temperature(temp: int):
    """Sets the cabin temperature to a specific degree Celsius."""
    return {"message": f"Cabin temperature set to {temp}°C"}

@tool
def get_vehicle_health():
    """Provides a report of tire pressure, brakes, and oil levels."""
    return {"tire_pressure": "Normal", "oil_level": "Optimal"}

# --- DOMAIN: NAVIGATION ---
@tool
def discover_places(query: str):
    """Search for nearby places like 'Charging Stations' or 'Parks'."""
    return {"results": [f"Audi Charging Hub Near {query}", "City Center Parking"]}

@tool
def get_route(destination: str):
    """Calculates the best route to a destination."""
    return {"destination": destination, "eta": "25 mins", "traffic": "Light"}

# --- DOMAIN: COMMUNICATION ---
@tool
def make_phone_call(contact_name: str):
    """Initiates a hands-free phone call."""
    return {"status": f"Calling {contact_name}..."}

@tool
def send_email(recipient: str, body: str):
    """Drafts and sends an email via voice command."""
    return {"status": f"Email sent to {recipient}"}

# --- DOMAIN: MEDIA & ENTERTAINMENT ---
@tool
def radio_control(frequency: float):
    """Tunes the Audi FM/AM radio."""
    return {"status": f"Tuned to {frequency} MHz"}

@tool
def spotify_control(track_name: str):
    """Plays a specific track or playlist on Spotify."""
    return {"status": f"Playing {track_name} on Spotify"}

# --- DOMAIN: SERVICES ---
@tool
def reserve_table(restaurant: str, time: str):
    """Books a table at a restaurant."""
    return {"status": f"Reservation confirmed at {restaurant} for {time}"}

# --- DOMAIN: CALENDAR ---
@tool
def calendar_search(query: str):
    """Search for upcoming meetings or events."""
    return {"events": [f"Meeting with Design Team at {query}"]}

@tool
def calendar_add_event(title: str, time: str):
    """Adds a new event to the driver's calendar."""
    return {"status": f"Event '{title}' added for {time}"}

@tool
def calendar_delete_event(event_title: str):
    """Deletes or cancels an event from the driver's calendar by title."""
    # In a real system, this would search for the ID and delete it
    return {"status": f"Event '{event_title}' has been successfully removed from your calendar."}

# --- 3. TOOL GROUPING & NODE SETUP ---

# Tools that DON'T need driver confirmation (Information/Entertainment)
SAFE_TOOLS = [
    get_battery_status, get_vehicle_health, discover_places, 
    get_route, radio_control, spotify_control, calendar_search
]

# Tools that DO need driver confirmation (Action/Privacy/Cost)
SENSITIVE_TOOLS = [
    open_sunroof, set_temperature, make_phone_call, 
    send_email, reserve_table, calendar_add_event, calendar_delete_event
]

safe_tool_node = ToolNode(SAFE_TOOLS)
sensitive_tool_node = ToolNode(SENSITIVE_TOOLS)

# --- 4. MODEL & GRAPH SETUP ---
load_dotenv()
llm = ChatOpenAI(
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_BASE"),
    model_name=os.getenv("MODEL")
)
llm_with_tools = llm.bind_tools(SAFE_TOOLS + SENSITIVE_TOOLS)

def assistant_node(state: Chatstate):
    sys_msg = SystemMessage(content=(
        "You are an advanced Audi AI Assistant. You help with Navigation, Calls, "
        "Calendar, Media, and Vehicle settings. Trigger tools directly. "
        "Sensitive actions (calls, emails, reservations, sunroof) will be confirmed by the system."
    ))
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

def should_continue(state: Chatstate):
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        return END
    
    tool_name = last_message.tool_calls[0]["name"]
    sensitive_names = [t.name for t in SENSITIVE_TOOLS]
    
    if tool_name in sensitive_names:
        return "sensitive_tools"
    return "safe_tools"

workflow = StateGraph(Chatstate)
workflow.add_node("assistant", assistant_node)
workflow.add_node("safe_tools", safe_tool_node)
workflow.add_node("sensitive_tools", sensitive_tool_node)

workflow.add_edge(START, "assistant")
workflow.add_conditional_edges("assistant", should_continue, {
    "safe_tools": "safe_tools", "sensitive_tools": "sensitive_tools", END: END
})
workflow.add_edge("safe_tools", "assistant")
workflow.add_edge("sensitive_tools", "assistant")

checkpointer = MemorySaver()
chatbot = workflow.compile(checkpointer=checkpointer, interrupt_before=["sensitive_tools"])

# --- 5. EXECUTION LOOP ---

def run_assistant():
    config = {"configurable": {"thread_id": "audi_full_suite"}}
    print("\n--- Audi Full-Suite Assistant Active ---")

    while True:
        user_input = input("\nDriver: ")
        if user_input.lower() in ["quit", "exit"]: break

        prefix_printed = False
        for msg, metadata in chatbot.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="messages"):
            if metadata["langgraph_node"] == "assistant" and msg.content:
                if not prefix_printed:
                    print("Assistant: ", end="", flush=True)
                    prefix_printed = True
                print(msg.content, end="", flush=True)

        snapshot = chatbot.get_state(config)
        if snapshot.next and "sensitive_tools" in snapshot.next:
            last_msg = snapshot.values["messages"][-1]
            t_name = last_msg.tool_calls[0]['name']
            
            print(f"\n\n[VEHICLE SYSTEM]: Action Required: {t_name}")
            confirm = input("Confirm this request? (yes/no): ")
            
            if confirm.lower() == "yes":
                resume_prefix = False
                for msg, metadata in chatbot.stream(None, config, stream_mode="messages"):
                    if metadata["langgraph_node"] == "assistant" and msg.content:
                        if not resume_prefix:
                            print("Assistant: ", end="", flush=True)
                            resume_prefix = True
                        print(msg.content, end="", flush=True)
            else:
                print("Assistant: I have cancelled that request for you.")
                chatbot.update_state(config, {"messages": [HumanMessage(content="Action declined.")]}, as_node="assistant")
        print()

if __name__ == "__main__":
    run_assistant()