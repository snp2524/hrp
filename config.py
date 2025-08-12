import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

RANDOM_STATE = 2025
USE_GEMINI_EMBEDDINGS = False

load_dotenv()


def get_gemini_llm():
    """Initializes and returns the Google Gemini LLM."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. Please set it."
        )

    # Using a lower temperature for more factual, less creative answers
    return ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)


# --- Placeholder for OpenAI ---
# Uncomment the following code and set your OPENAI_API_KEY
# from langchain_openai import OpenAI
# def get_openai_llm():
#     """Initializes and returns the OpenAI GPT LLM."""
#     if "OPENAI_API_KEY" not in os.environ:
#         raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it.")
#     return OpenAI(temperature=0.1)

# --- Placeholder for Anthropic ---
# Uncomment the following code and set your ANTHROPIC_API_KEY
# from langchain_anthropic import Anthropic
# def get_claude_llm():
#     """Initializes and returns the Anthropic Claude LLM."""
#     if "ANTHROPIC_API_KEY" not in os.environ:
#         raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it.")
#     return Anthropic(model="claude-3-sonnet-20240229", temperature=0.1)
