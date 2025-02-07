import re
import numpy as np
import faiss
import pandas as pd
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM


def split_text_into_chunks(text, max_length=500):
    paragraphs = re.split(r'(?<=\.)\s*\n', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_length = len(paragraph)
        if paragraph_length > max_length:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_length = len(sentence)

                if current_length + sentence_length > max_length:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
        else:
            if current_length + paragraph_length > max_length:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def extract_text_from_pdf(pdf_path, max_length=500):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        page_text = page.extract_text()

        if page_text.strip():
            page_text += ' '

        full_text += page_text


    return split_text_into_chunks(full_text, max_length)



def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    prefixes = ["Услуга", "Условие", "Тариф"]
    csv_chunks = df.apply(lambda row: " | ".join(
        [f"{prefixes[i]}: {str(row[col])}" for i, col in enumerate(df.columns) if pd.notna(row[col])]
    ), axis=1).tolist()
    return csv_chunks


def process_and_search(file_paths, query):
    all_chunks = []

    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            all_chunks.extend(extract_text_from_pdf(file_path))
        elif file_path.endswith('.csv'):
            all_chunks.extend(extract_text_from_csv(file_path))
        else:
            print(f"Формат файла {file_path} не поддерживается.")

    embedding_model = OllamaEmbeddings(model="bge-m3")
    embeddings = np.array(embedding_model.embed_documents(all_chunks))

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query_embedding = np.array(embedding_model.embed_query(query))
    k = 10
    indexes = index.search(query_embedding.reshape(1, -1), k)[1][0]

    return [all_chunks[i] for i in indexes]


def get_response_from_model(query, context):
    input_text = f"Вот контекст из документа, выбери из него подходящую информацию для вопроса пользователя и отвечай по контексту. Если в контексте нет информации близкой к вопросу, просто дословно напиши 'Я не знаю'. Контекст:\n\n{context}\n\nВопрос пользователя: {query}"
    llm = OllamaLLM(model="llama3.1")
    return llm.invoke(input_text)


def main(file_paths, query):
    context = process_and_search(file_paths, query)
    context = [text.replace("\n", " ") for text in context]
    context = [text.replace("\xa0", " ") for text in context]
    print("Контекст:", context)
    response = get_response_from_model(query, context)
    print("Ответ модели:", response)
