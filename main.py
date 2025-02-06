
import re
import numpy as np
import faiss
import pandas as pd
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM


def pdf_to_chunks(pdf_path):

    reader = PdfReader(pdf_path)
    text = reader.pages
    all_chunks = []
    for page in text:
        text_of_page = page.extract_text()
        if text_of_page:
            if not re.search(r'[.!?]', text_of_page):
                all_chunks.append(text_of_page)
            sentences = re.split(r'(?<=[.!?])\s+', text_of_page)
            chunks = []
            current_chunk = ""
            for i in sentences:
                if len(current_chunk) + len(i) > 500:
                    chunks.append(current_chunk.strip())
                    current_chunk = i
                else:
                    current_chunk += " " + i
            all_chunks.extend(chunks)
    return all_chunks


def csv_to_chunks(csv_path):
    df = pd.read_csv(csv_path)
    prefixes = ["Услуга", "Условие", "Тариф"]

    all_chunks = df.apply(lambda row: " | ".join(
        [f"{prefixes[i]}: {str(row[col])}" for i, col in enumerate(df.columns) if pd.notna(row[col])],
    ), axis=1).tolist()

    return all_chunks


def process_and_search(file_paths, query):

    all_chunks = []


    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            chunks_pdf = pdf_to_chunks(file_path)
            all_chunks.extend(chunks_pdf)
        elif file_path.endswith('.csv'):
            chunks_csv = csv_to_chunks(file_path)
            all_chunks.extend(chunks_csv)
        else:
            print(f"Формат файла {file_path} не поддерживается.")


    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
    embeddings = embedding_model.embed_documents(all_chunks)
    embeddings = np.array(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    for i in range(0, len(embeddings), 100):
        index.add(embeddings[i:i + 100])


    query_embedding = embedding_model.embed_query(query)
    query_embedding = np.array(query_embedding)
    k = 10
    indexes = index.search(query_embedding.reshape(1, -1), k)[1][0]

    context = []
    for i in indexes:
        context.append(all_chunks[i])

    context = [text.replace("\n", " ") for text in context]
    context = [text.replace("\xa0", " ") for text in context]

    return context


def get_response_from_model(query, context):
    input_text = f"Вот контекст из документа, выбери из него подходящую информацию для вопроса пользователя и отвечай по контексту. Если в контексте нет информации близкой к вопросу, просто дословно напиши 'Я не знаю'. Контекст:\n\n{context}\n\nВопрос пользователя: {query}"
    llm = OllamaLLM(model="llama3.1")
    response = llm.invoke(input_text)
    return response


def main(file_paths, query):

    context = process_and_search(file_paths, query)
    print("Контекст:", context)

    response = get_response_from_model(query, context)
    print("Ответ модели:", response)
