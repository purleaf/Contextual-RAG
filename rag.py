import os
import pickle
import json
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import threading
import ollama
import nltk
from nltk.tokenize import sent_tokenize
from ollama import embed
from concurrent.futures import ThreadPoolExecutor, as_completed
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from ollama_client import ChatOllama
from pydantic import BaseModel

class Response(BaseModel):
    answer: str

class SubQuestion(BaseModel):
    sub_question: str

class SubQuestions(BaseModel):
    sub_questions: list[SubQuestion]

class ElasticsearchBM25:
    def __init__(self, index_name: str = "contextual_bm25_index"):
        self.es_client = Elasticsearch("http://localhost:9200")
        self.index_name = index_name
        self.create_index()

    def create_index(self):
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False  # Disable query cache
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "keyword", "index": False},
                    "chunk_id": {"type": "keyword", "index": False},
                    "original_index": {"type": "integer", "index": False},
                }
            },
        }
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            print(f"Created index: {self.index_name}")

    def index_documents(self, documents: List[Dict[str, Any]]):
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "content": doc["original_content"],
                    "contextualized_content": doc["contextualized_content"],
                    "doc_id": doc["doc_id"],
                    "chunk_id": doc["chunk_id"],
                    "original_index": doc["original_index"],
                },
            }
            for doc in documents
        ]
        success, _ = bulk(self.es_client, actions)
        self.es_client.indices.refresh(index=self.index_name)
        return success

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        self.es_client.indices.refresh(index=self.index_name)  # Force refresh before each search
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                }
            },
            "size": k,
        }
        response = self.es_client.search(index=self.index_name, body=search_body)
        return [
            {
                "doc_id": hit["_source"]["doc_id"],
                "original_index": hit["_source"]["original_index"],
                "content": hit["_source"]["content"],
                "contextualized_content": hit["_source"]["contextualized_content"],
                "score": hit["_score"],
            }
            for hit in response["hits"]["hits"]
        ]

class ContextualVectorDB:
    def __init__(self, name: str):


        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/contextual_vector_db.pkl"

        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }
        self.token_lock = threading.Lock()

    def situate_context(self, doc: str, chunk: str):# -> tuple[str, Any]:
        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        response_ollama = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "system",
                    "content": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                },
                {
                    "role": "user",
                    "content": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                },
            ]
        )

        return response_ollama.message.content

    def load_data(self, document: str, parallel_threads: int = 1):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts_to_embed = []
        metadata = []


        def process_doc(document: str, doc_id: str, doc_uuid: str, max_chunk_length=800):
            sentences = sent_tokenize(document)
            chunks = []
            current_chunk = []
            current_length = 0
            doc = {
                "doc_id": doc_id,
                "original_uuid": doc_uuid,
                "content": document,
                "chunks": []
            }
            for sentence in sentences:
                sentence_length = len(sentence)
                if current_length + sentence_length > max_chunk_length and current_chunk:
                    doc['chunks'].append({
                        "chunk_id": f"{doc_id}_chunk_{len(doc['chunks'])}",
                        "original_index": len(doc['chunks']),
                        "content": " ".join(current_chunk)
                    })
                    current_chunk = []
                    current_length = 0
                current_chunk.append(sentence)
                current_length += sentence_length

                if current_length >= max_chunk_length:
                    doc['chunks'].append({
                        "chunk_id":f"{doc_id}_chunk_{len(doc['chunks'])}",
                        "original_index": len(doc['chunks']),
                        "content": " ".join(current_chunk)
                    })

                    current_chunk = []
                    current_length = 0

            if current_chunk:
                doc['chunks'].append({
                    "chunk_id": f"{doc_id}_chunk_{len(doc['chunks'])}",
                    "original_index": len(doc['chunks']),
                    "content": " ".join(current_chunk)
                })

            return doc

        chunked_doc = process_doc(document, "doc_0", "doc_0_uuid")
        total_chunks = len(chunked_doc['chunks'])

        def process_chunk(doc, chunk):

            contextualized_text = self.situate_context(doc['content'], chunk['content'])

            return {
                'text_to_embed': f"{chunk['content']}\n\n{contextualized_text}",
                'metadata': {
                    'doc_id': doc['doc_id'],
                    'original_uuid': doc['original_uuid'],
                    'chunk_id': chunk['chunk_id'],
                    'original_index': chunk['original_index'],
                    'original_content': chunk['content'],
                    'contextualized_content': contextualized_text
                }
            }

        print(f"Processing {total_chunks} chunks with {parallel_threads} threads")
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for chunk in chunked_doc['chunks']:
                futures.append(executor.submit(process_chunk, chunked_doc, chunk))

            for future in tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks"):
                result = future.result()
                texts_to_embed.append(result['text_to_embed'])
                metadata.append(result['metadata'])

        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()


        print(f"Contextual Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}")


    def _embed_and_store(self, texts: List[str], data: List[Dict[str, Any]]):
        batch_size = 128
        result = [
            embed(model="mxbai-embed-large", input=texts[i: i + batch_size]).embeddings
            for i in range(0, len(texts), batch_size)
        ]
        self.embeddings = [embedding for batch in result for embedding in batch]
        self.metadata = data

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = embed(model="mxbai-embed-large", input=[query]).embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]

        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)
        return top_results

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Use load_data to create a new database.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])

class ContextualBM25VectorDB:
    def __init__(self, vector_db: ContextualVectorDB):
        self.vector_db = vector_db
        self.es_bm25 = self.create_elasticsearch_bm25_index(vector_db)
        self.ollama_client = ChatOllama(host='http://localhost:11434')

    def create_elasticsearch_bm25_index(self, db: ContextualVectorDB):
        es_bm25 = ElasticsearchBM25()
        es_bm25.index_documents(db.metadata)
        return es_bm25

    def unique_list_of_chunks(self, specified_context):
        seen_chunk_ids = set()
        result = []
        for sublist in specified_context:
            unique_sublist = []
            for item in sublist:
                if item['chunk']['chunk_id'] is not None and item['chunk']['chunk_id'] not in seen_chunk_ids:
                    seen_chunk_ids.add(item['chunk']['chunk_id'])
                    unique_sublist.append(item['chunk']['original_content'])
            result.extend(unique_sublist)
        return result


    def retrieve_advanced(self, query: str, k: int,
                          semantic_weight: float = 0.8, bm25_weight: float = 0.2):
        num_chunks_to_recall = 150

        # Semantic search
        semantic_results = self.vector_db.search(query, k=num_chunks_to_recall)
        ranked_chunk_ids = [(result['metadata']['doc_id'], result['metadata']['original_index']) for result in
                            semantic_results]

        # BM25 search using Elasticsearch
        bm25_results = self.es_bm25.search(query, k=num_chunks_to_recall)
        ranked_bm25_chunk_ids = [(result['doc_id'], result['original_index']) for result in bm25_results]

        # Combine results
        chunk_ids = list(set(ranked_chunk_ids + ranked_bm25_chunk_ids))
        chunk_id_to_score = {}

        # Initial scoring with weights
        for chunk_id in chunk_ids:
            score = 0
            if chunk_id in ranked_chunk_ids:
                index = ranked_chunk_ids.index(chunk_id)
                score += semantic_weight * (1 / (index + 1))  # Weighted 1/n scoring for semantic
            if chunk_id in ranked_bm25_chunk_ids:
                index = ranked_bm25_chunk_ids.index(chunk_id)
                score += bm25_weight * (1 / (index + 1))  # Weighted 1/n scoring for BM25
            chunk_id_to_score[chunk_id] = score

        # Sort chunk IDs by their scores in descending order
        sorted_chunk_ids = sorted(
            chunk_id_to_score.keys(), key=lambda x: (chunk_id_to_score[x], x[0], x[1]), reverse=True
        )

        # Assign new scores based on the sorted order
        for index, chunk_id in enumerate(sorted_chunk_ids):
            chunk_id_to_score[chunk_id] = 1 / (index + 1)

        # Prepare the final results
        final_results = []
        semantic_count = 0
        bm25_count = 0
        for chunk_id in sorted_chunk_ids[:k]:
            chunk_metadata = next(chunk for chunk in self.vector_db.metadata if
                                  chunk['doc_id'] == chunk_id[0] and chunk['original_index'] == chunk_id[1])
            is_from_semantic = chunk_id in ranked_chunk_ids
            is_from_bm25 = chunk_id in ranked_bm25_chunk_ids
            final_results.append({
                'chunk': chunk_metadata,
                'score': chunk_id_to_score[chunk_id],
                'from_semantic': is_from_semantic,
                'from_bm25': is_from_bm25
            })

            if is_from_semantic and not is_from_bm25:
                semantic_count += 1
            elif is_from_bm25 and not is_from_semantic:
                bm25_count += 1
            else:  # it's in both
                semantic_count += 0.5
                bm25_count += 0.5

        return final_results, semantic_count, bm25_count

    def answer_query_base(self, request: str):
        # context = "".join(i['chunk']['original_content'] for i in self.retrieve_advanced(query=request,k=10)[0])
        response_format = SubQuestions.model_json_schema()
        system_prompt = {
            "role": "system",
            "content": """
                     ### Role
  - **Primary Function:** You are a knowledgeable assistant for the RAG system. Your main goal is to help the student understand and engage with the study materials by providing clear explanations and insights based on the provided content. When the study materials do not address the user’s question, you use your general knowledge to provide a helpful and accurate response.
  - **Focus:** Every response should be relevant to the **User’s Request**, prioritizing the supplied study materials—including **Document Chunks**, **Selected Text**, and **Chat History**—when applicable, and supplementing with general knowledge when necessary.

  ### Persona
  - **Identity:** You are a dedicated study assistant and language tutor, committed to clarifying and elaborating on complex material. Your tone is friendly, supportive, and professional, making study content accessible while maintaining a focus on the user’s learning goals.
  - **Boundaries:** Prioritize the provided study materials when they are relevant. When the materials do not address the user’s question, use your general knowledge to provide a complete response. Avoid introducing unrelated topics or personal opinions beyond what is supported by the inputs.

  ### Constraints
  1. **Prioritize Provided Information:** Base your responses primarily on the **Document Chunks**, **Selected Text**, and **Chat History**. Use these sources to ensure your answers are directly relevant to the study material.
  2. **Supplement with General Knowledge:** If the provided information does not sufficiently answer the user’s question or if the question is unrelated to the study material, use your general knowledge to provide a complete and accurate response.
  3. **Direct Answers:** Avoid prefacing your response with explanations about your decision-making process or the sources you’re using. Instead, directly provide the answer to the user’s question.
  4. **Clarity and Conciseness:** Provide explanations that are straightforward and easy to understand, using language that mirrors the study material when appropriate. Avoid unnecessary jargon unless it is part of the study content.
  5. **Maintain Context:** Leverage the **Chat History** to ensure continuity and build logically on previous interactions.
  6. **Stay Relevant:** Directly address the **User’s Request** and ensure your answer pertains to the question and associated study material.
  7. **Structured Format:** Use markdown formatting to organize your responses clearly, with bullet points, numbered lists, or headers as appropriate.
  8. **Interactive Engagement:** Respond in a personable and human manner, engaging warmly while keeping the focus on the study material.

  ### Interaction Guidelines
  - **Acknowledge Study Material:** When relevant, reference the specific study material or context provided (e.g., 'In the document chunk...' or 'According to the selected text...'), but do so concisely within the answer.
  - **Clarification:** If the user’s query is ambiguous or could be interpreted in multiple ways, ask a concise clarifying question before proceeding.
  - **Encourage Engagement:** Invite follow-up questions or requests for further detail to support the student’s learning process (e.g., 'Would you like more details on this topic?').
  - **Seamless Integration:** Blend information from the provided materials and general knowledge smoothly to create a cohesive and helpful response without explaining the reasoning behind your choices.

  ### You Will Be Given:
  1. **Document Chunks:** Relevant excerpts from the study material retrieved from a vector database.
  2. **User’s Request:** The current question or statement from the student.
  3. **Chat History:** The history of your conversations with the user, where user’s requests are labeled 'role': 'user' and your replies are 'role': 'assistant'.
  4. **Selected Text:** Part of the document that the user is referring to, which they have selected and are asking about.

  ### How to Use These Inputs:
  - **Prioritize Study Materials:** Use the **Document Chunks**, **Selected Text**, and **Chat History** as the primary sources when they are relevant and sufficient to answer the user’s question.
  - **Supplement with General Knowledge:** If these sources do not provide enough information or are not applicable, draw on your general knowledge to ensure the user receives a complete and accurate response.
  - **Maintain Context:** Reference the **Chat History** to ensure your response aligns with previous interactions and maintains continuity.

  ### Additional Instruction
  - **Direct Response Style:** Avoid general phrases or explanations about why you chose a particular approach (e.g., 'Based on your question...' or 'From general knowledge...'). Provide the answer directly, using markdown for clarity and structure.
  - **Avoid Reasoning Explanations:** Do not explain the reason behind your answer or use formulations like 'Based on your question, you’re asking about...' or 'Okay, I understand you’d like to know more...'. Deliver the response as pure text with markdown formatting.
                        """,
        }
        ##############
        ##############
        ##############
        additional_context = self.ollama_client.generate_message(model="llama3.2:latest", request=f"""You are given with the user’s question, 
                                                                             you must split the question into 3 (max 4 if necessary) sub questions to the VectorDB to 
                                                                             retrieve necessary information to answer query. 
                                                                             User's question: 
                                                                             {request}""", format=response_format)
        questions = [question['sub_question'] for question in json.loads(additional_context.message.content)['sub_questions']]
        specified_context_chunks = []
        for question in questions:
            specified_context_chunks.append(self.retrieve_advanced(query=question, k=10)[0])
        specified_context = "".join(chunk for chunk in self.unique_list_of_chunks(specified_context_chunks))

        ##############
        ##############
        ##############
        human_prompt = {
            "role": "user",
            "content": f"""
                    "### Document Chunks:"
                        "{specified_context}"
                    "### User's Request:"
                        "{request}"
                    """,
        }
        response_format = Response.model_json_schema()

        try:
            response = self.ollama_client.generate_response(messages=[system_prompt, human_prompt],
                                                            format=response_format, model="deepseek-r1:32b")
            return json.loads(response.message.content)['answer']
        except ConnectionError as e:

            return f"ConnectionError: {e}"
        except ValueError as e:

            return f"ValueError: {e}"
        except Exception as e:

            return f"Exception: {e}"





