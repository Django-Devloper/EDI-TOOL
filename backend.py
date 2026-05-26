from file_upload import process_document
from model_connection import get_qa_chain, get_vector_store


def preview_text(text, limit=500):
    cleaned_text = " ".join(text.split())
    if len(cleaned_text) <= limit:
        return cleaned_text

    return f"{cleaned_text[:limit]}..."


def process_uploaded_file(uploaded_file):
    vector_store = get_vector_store()
    process_document(uploaded_file, vector_store)


def answer_question(question):
    vector_store = get_vector_store()
    retrieved_docs = vector_store.similarity_search_with_score(question, k=4)

    print(f"[QUESTION] {question}", flush=True)
    print(f"[RETRIEVAL] returned_chunks={len(retrieved_docs)}", flush=True)

    for index, (doc, score) in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        print(
            f"[RETRIEVED {index}] score={score} source={source} "
            f"chars={len(doc.page_content)} text={preview_text(doc.page_content)}",
            flush=True,
        )

    qa_chain = get_qa_chain()
    response = qa_chain.invoke({"query": question})
    return response["result"]
