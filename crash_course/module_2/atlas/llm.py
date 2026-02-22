import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("you must set GEMINI_API_KEY env var")
client = genai.Client(api_key=api_key)


def query_llm(query: str) -> str:
    "heres a query"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
    )
    if response.text is None:
        raise RuntimeError("model response text is none")
    return response.text
