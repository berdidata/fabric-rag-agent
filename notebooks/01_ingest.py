# Fabric notebook: ingest Lakehouse documents into Azure AI Search
# Paste each "# CELL" block into its own notebook cell. Attach the Lakehouse as default.

# CELL 1 ---------------------------------------------------------------
%pip install openai azure-search-documents pypdf tiktoken -q

# CELL 2: config ------------------------------------------------------
import os
# Prefer Key Vault:
# AOAI_KEY = notebookutils.credentials.getSecret("https://<vault>.vault.azure.net/", "aoai-key")
AOAI_ENDPOINT = "https://<your-aoai>.openai.azure.com/"
AOAI_KEY = "<key>"
AOAI_API_VERSION = "2024-10-21"          # check current GA version
EMBED_DEPLOYMENT = "text-embedding-3-small"
EMBED_DIM = 1536

SEARCH_ENDPOINT = "https://<your-search>.search.windows.net"
SEARCH_KEY = "<admin-key>"
INDEX_NAME = "docs-index"

DOCS_PATH = "/lakehouse/default/Files/docs"
CHUNK_TOKENS, OVERLAP_TOKENS = 500, 80

# CELL 3: extract text -------------------------------------------------
from pathlib import Path
from pypdf import PdfReader

def extract(path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text). Markdown is page 1."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]

files = [p for p in Path(DOCS_PATH).rglob("*") if p.suffix.lower() in (".md", ".pdf")]
print(f"{len(files)} files found")

# CELL 4: chunk --------------------------------------------------------
import hashlib, tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str):
    tokens = enc.encode(text)
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    for start in range(0, max(len(tokens), 1), step):
        piece = enc.decode(tokens[start:start + CHUNK_TOKENS]).strip()
        if piece:
            yield piece

chunks = []
for f in files:
    for page, text in extract(f):
        for n, piece in enumerate(chunk_text(text)):
            cid = hashlib.md5(f"{f.name}|{page}|{n}".encode()).hexdigest()
            chunks.append({"id": cid, "source": f.name, "page": page, "chunk_no": n, "content": piece})
print(f"{len(chunks)} chunks")

# CELL 5: embed --------------------------------------------------------
from openai import AzureOpenAI
aoai = AzureOpenAI(azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY, api_version=AOAI_API_VERSION)

BATCH = 64
for i in range(0, len(chunks), BATCH):
    batch = chunks[i:i + BATCH]
    resp = aoai.embeddings.create(model=EMBED_DEPLOYMENT, input=[c["content"] for c in batch])
    for c, d in zip(batch, resp.data):
        c["embedding"] = d.embedding

# CELL 6: save to Delta (lineage + reporting) -------------------------
import pandas as pd
df = pd.DataFrame([{k: v for k, v in c.items() if k != "embedding"} for c in chunks])
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("doc_chunks")

# CELL 7: create index (vector + keyword) -----------------------------
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
)
cred = AzureKeyCredential(SEARCH_KEY)
index = SearchIndex(
    name=INDEX_NAME,
    fields=[
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32),
        SimpleField(name="chunk_no", type=SearchFieldDataType.Int32),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=EMBED_DIM,
                    vector_search_profile_name="vprofile"),
    ],
    vector_search=VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="vprofile", algorithm_configuration_name="hnsw")],
    ),
)
SearchIndexClient(SEARCH_ENDPOINT, cred).create_or_update_index(index)

# CELL 8: upload -------------------------------------------------------
from azure.search.documents import SearchClient
sc = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, cred)
for i in range(0, len(chunks), 500):
    res = sc.merge_or_upload_documents(chunks[i:i + 500])
    print(sum(r.succeeded for r in res), "uploaded")
