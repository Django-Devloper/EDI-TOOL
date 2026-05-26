import os
import tempfile

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def preview_text(text, limit=300):
    cleaned_text = " ".join(text.split())
    if len(cleaned_text) <= limit:
        return cleaned_text

    return f"{cleaned_text[:limit]}..."


def read_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        reader = PdfReader(temp_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

        return text
    finally:
        os.remove(temp_path)


def get_uploaded_file_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return read_pdf(uploaded_file)

    return uploaded_file.read().decode("utf-8")


def process_document(uploaded_file, vector_store):
    text = get_uploaded_file_text(uploaded_file)

    print(
        f"[UPLOAD] file={uploaded_file.name} extracted_chars={len(text)}",
        flush=True,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_text(text)

    print(f"[CHUNKS] total={len(chunks)}", flush=True)
    for index, chunk in enumerate(chunks, start=1):
        print(
            f"[CHUNK {index}] chars={len(chunk)} text={preview_text(chunk)}",
            flush=True,
        )

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "source": uploaded_file.name,
            },
        )
        for chunk in chunks
    ]

    vector_store.add_documents(docs)
    print(f"[EMBEDDINGS] stored_chunks={len(docs)}", flush=True)
