
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API keys and LangSmith settings
load_dotenv()

# Check required API keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY in .env")

if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError("Missing LANGSMITH_API_KEY in .env")

# Enable tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault(
    "LANGSMITH_PROJECT",
    "My-LangChain-Trace"
)

# Initialize model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Answer clearly and concisely."
    ),
    ("human", "{question}")
])

# Build LangChain pipeline
chain = prompt | llm | StrOutputParser()


def run_app(question: str) -> str:
    """Run the app and trace its execution in LangSmith."""
    return chain.invoke(
        {"question": question},
        config={
            "run_name": "Question-Answer-Chain",
            "tags": ["demo", "langchain"],
            "metadata": {"app": "python-tracing-demo"}
        }
    )


if __name__ == "__main__":
    question = input("Ask a question: ")

    try:
        answer = run_app(question)
        print("\nAI Answer:")
        print(answer)
        print(
            "\nTrace project: "
            + os.getenv("LANGSMITH_PROJECT", "My-LangChain-Trace")
        )
        print("Open LangSmith to inspect the trace.")
    except Exception as error:
        print(f"Application error: {error}")
