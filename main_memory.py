import json
from openai import OpenAI
from rag import search_knowledge
from memory import load_memory, update_preference
import tools as car_tools
from tool_schema import tools
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
MODEL = os.getenv("MODEL")



client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)


messages = [{
    "role": "system",
    "content": """You are a premium Audi in-car assistant.
Be concise, intelligent, and proactive.

You are a premium Audi in-car assistant.

You have access to these capabilities:
- battery status
- temperature control
- sunroof control
- vehicle health


CRITICAL RULES:
- Do NOT claim or imply any capability outside this list.
- If a user asks for something outside these capabilities (e.g., navigation, music, calls):
→ Clearly say it is not available.
→ Offer an alternative if possible.
- Never say you can "help with" or "assist with" features you do not support.

Guidelines:
- Use tools ONLY when relevant to the user request
- Do NOT force tool suggestions in every response
- For general conversation (greetings, boredom, etc.), respond naturally
- Do NOT mention unavailable features
- If user asks for unsupported features, politely say it's not available
- Only suggest additional actions if they are clearly relevant to the user's current intent.

Be helpful, natural, and context-aware.

If user shares preferences, remember them.
"""
}]


handlers = {
    "get_battery_status": car_tools.get_battery_status,
    "set_temperature": lambda args: car_tools.set_temperature(args.get("temp")),
    "open_sunroof": car_tools.open_sunroof,
    "get_vehicle_health": car_tools.get_vehicle_health
}



def extract_memory(user_input):
    prompt = f"""
    Extract user preference from this sentence.

    Return JSON:
    {{"key": "category", "value": "info"}}

    Example:
    "I love pizza" → {{"key": "food", "value": "pizza"}}

    If nothing important, return null.

    Sentence: "{user_input}"
    """

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = res.choices[0].message.content.strip()

        return json.loads(content)
    except:
        return None


print("Audi Assistant Online. Type 'exit' to quit.")

while True:
    user_input = input("\nDriver: ")

    if user_input.lower() == "exit":
        print("Shutting down assistant.")
        break


    mem = extract_memory(user_input)
    if mem and "key" in mem:
        update_preference(mem["key"], mem["value"])


    retrieved = search_knowledge(user_input)
    context = "\n".join(retrieved)

    if context.strip():
        messages.append({
            "role": "system",
            "content": f"Vehicle knowledge:\n{context}"
        })


    memory = load_memory()
    if memory:
        messages.append({
            "role": "system",
            "content": f"User preferences: {memory}"
        })

    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            stream=True
        )

        full = ""
        tool_calls = {}

        print("Assistant: ", end="", flush=True)

        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                print(delta.content, end="", flush=True)
                full += delta.content

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index

                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc.id or f"call_{idx}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        }

                    if tc.function.name:
                        tool_calls[idx]["function"]["name"] += tc.function.name

                    if tc.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc.function.arguments

        print()

    except Exception as e:
        print("[Error]:", e)
        continue

    assistant_msg = {
        "role": "assistant",
        "content": full if full else None
    }

    if tool_calls:
        assistant_msg["tool_calls"] = list(tool_calls.values())

    messages.append(assistant_msg)

    # ⚙️ EXECUTE TOOLS
    if tool_calls:
        for call in tool_calls.values():
            name = call["function"]["name"]

            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except:
                args = {}

            if name in handlers:
                print(f"[*] Running {name}...")
                result = handlers[name](args) if args else handlers[name]()
            else:
                result = {"error": "not found"}

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": json.dumps(result)
            })

        # second response
        second = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True
        )

        print("Assistant: ", end="", flush=True)

        final = ""
        for chunk in second:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                final += text

        print()

        messages.append({"role": "assistant", "content": final})
