import json
from openai import OpenAI
from tools import TOOL_MAP, TOOL_DEFINITIONS
from config import OPENROUTER_API_KEY, MODEL_NAME, MAX_AGENT_ITERATIONS


def run_agent(user_message: str, system_prompt: str, status_callback=None) -> dict:
    """
    Run the agent loop.

    Args:
        user_message: The formatted user request with all trip details
        system_prompt: The system prompt that guides agent behavior
        status_callback: Optional callback fn(step_description: str) for UI updates

    Returns:
        dict with keys: "response" (final text), "tool_calls" (list of tools called), "iterations" (int)
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    tool_calls_log = []

    for i in range(MAX_AGENT_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=4096,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Case 1: LLM wants to call one or more tools
        if finish_reason == "tool_calls" or message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)
                tool_call_id = tool_call.id

                if status_callback:
                    status_callback(tool_name, tool_input)

                result = execute_tool(tool_name, tool_input)
                tool_calls_log.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "success": "error" not in result,
                    "result": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result),
                })

        # Case 2: LLM gives final answer (no tool calls)
        elif finish_reason == "stop":
            return {
                "response": message.content or "",
                "tool_calls": tool_calls_log,
                "iterations": i + 1,
            }

        # Case 3: Unexpected finish reason
        else:
            return {
                "response": message.content or "Agent encountered an unexpected state.",
                "tool_calls": tool_calls_log,
                "iterations": i + 1,
            }

    # Hit max iterations without a final answer. Ask the LLM to produce a real
    # diagnostic instead of returning generic boilerplate, so the user knows
    # WHY we gave up (most common cause: an unrealistic budget).
    diagnostic_messages = list(messages) + [{
        "role": "user",
        "content": (
            f"You have reached the maximum of {MAX_AGENT_ITERATIONS} planning steps "
            "without producing a final itinerary. Do NOT call any more tools. "
            "Instead, in 4–6 sentences, explain to the user:\n"
            "1. What you were trying to do and what tools you already called.\n"
            "2. The SPECIFIC constraint that prevented you from finishing. The most "
            "common cause is an unrealistic budget (e.g. less than $30/person/day "
            "after transportation), but it could also be an impossible destination, "
            "no available flights, conflicting requirements, etc.\n"
            "3. The math behind your conclusion — e.g. 'Your $100 budget for 5 people × "
            "5 days = $4/person/day, which doesn't cover even one meal at a fast food "
            "restaurant, let alone lodging or transportation.'\n"
            "4. A concrete suggestion for what the user should change (a realistic minimum "
            "budget for this trip, fewer days, fewer people, a closer destination, etc.).\n"
            "Be honest, specific, and use real numbers. Format with markdown."
        ),
    }]

    diag_text = ""
    try:
        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=diagnostic_messages,
            max_tokens=600,
        )
        diag_text = (final_response.choices[0].message.content or "").strip()
    except Exception as e:
        diag_text = f"_(Could not generate detailed diagnostic: {e})_"

    fallback_body = (
        diag_text
        or "Your request may have constraints that make it hard to plan — most often this "
           "is an unrealistic budget. Try increasing the budget, reducing trip length or "
           "the number of travelers, or relaxing special requests."
    )

    return {
        "response": (
            f"### ⚠️ I couldn't finish planning this trip in {MAX_AGENT_ITERATIONS} steps\n\n"
            f"{fallback_body}"
        ),
        "tool_calls": tool_calls_log,
        "iterations": MAX_AGENT_ITERATIONS,
    }


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """Execute a tool by name. Returns result dict or error dict."""
    func = TOOL_MAP.get(tool_name)
    if not func:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return func(**tool_input)
    except Exception as e:
        return {"error": f"{tool_name} failed: {str(e)}"}
