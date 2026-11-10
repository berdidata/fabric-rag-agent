"""Search + agent logic. The model gets one tool (search_docs) and decides when to call it."""
import json
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

aoai = AzureOpenAI(
    azure_endpoint=os.environ["AOAI_ENDPOINT"],
    api_key=os.environ["AOAI_KEY"],
    api_version=os.getenv("AOAI_API_VERSION", "2024-10-21"),
)
search = SearchClient(
    os.environ["SEARCH_ENDPOINT"],
    os.getenv("SEARCH_INDEX", "docs-index"),
    AzureKeyCredential(os.environ["SEARCH_KEY"]),
)
CHAT_MODEL = os.getenv("CHAT_DEPLOYMENT", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-small")

SYSTEM_PROMPT = """You answer questions about internal documents.
Rules:
- Use the search_docs tool before answering any factual question. You may search more than once with different wording.
- Answer ONLY from the returned chunks. Cite them inline as [1], [2] using the numbers given.
- If the chunks do not contain the answer, say you could not find it in the documents.
- Be concise."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Hybrid (keyword + vector) search over the internal document index.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query in natural language"},
                "top": {"type": "integer", "description": "Number of chunks, 1-8", "default": 5},
            },
            "required": ["query"],
        },
    },
}]


def search_docs(query: str, top: int = 5) -> list[dict]:
    vec = aoai.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    results = search.search(
        search_text=query,  # keyword side of hybrid
        vector_queries=[VectorizedQuery(vector=vec, k_nearest_neighbors=top, fields="embedding")],
        select=["source", "page", "content"],
        top=top,
    )
    return [{"source": r["source"], "page": r["page"], "content": r["content"]} for r in results]


def ask(question: str, history: list[dict] | None = None, max_steps: int = 4):
    """Run the agent loop. Returns (answer, citations)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []),
                {"role": "user", "content": question}]
    citations: list[dict] = []

    for _ in range(max_steps):
        resp = aoai.chat.completions.create(model=CHAT_MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content, citations

        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            hits = search_docs(args["query"], min(int(args.get("top", 5)), 8))
            numbered = []
            for h in hits:
                citations.append(h)
                numbered.append(f"[{len(citations)}] ({h['source']} p.{h['page']})\n{h['content']}")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": "\n\n".join(numbered) or "No results."})

    return "Stopped after too many search steps.", citations
