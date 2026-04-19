import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter (LLM gateway)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Google Places
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# Model config — everything goes through OpenRouter with the OpenAI SDK.
MODEL_NAME = "openai/gpt-5.4-nano"  # Agent model — via OpenRouter
# MODEL_NAME = "openai/gpt-4.1-mini"  # alt: heavier, more expensive
JUDGE_MODEL_NAME = "openai/gpt-5.4"  # Eval judge — stronger than agent to reduce self-bias
MAX_AGENT_ITERATIONS = 15
