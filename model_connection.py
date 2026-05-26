from functools import lru_cache

from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate


OLLAMA_MODEL = "llama3.2:latest"
EMBED_MODEL = "nomic-embed-text"

POSTGRES_CONNECTION = (
    "postgresql+psycopg://postgres:Dipak%40123@localhost:5432/postgres"
)

COLLECTION_NAME = "documents"

PROMPT_TEMPLATE = """
You are a helpful AI assistant.

Answer using the provided context if dont find then get answer from LLM.


Context:
{context}

Question:
{question}

Answer:
"""


@lru_cache(maxsize=1)
def get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    return OllamaEmbeddings(
        model=EMBED_MODEL,
    )


@lru_cache(maxsize=1)
def get_vector_store():
    return PGVector(
        connection=POSTGRES_CONNECTION,
        collection_name=COLLECTION_NAME,
        embeddings=get_embedding_model(),
        use_jsonb=True,
    )


def get_prompt():
    return PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )


@lru_cache(maxsize=1)
def get_qa_chain():
    retriever = get_vector_store().as_retriever(
        search_kwargs={"k": 4},
    )

    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": get_prompt(),
        },
    )
