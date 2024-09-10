import requests
import os, dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
import requests
import pdfplumber
from io import BytesIO
import sys, io


dotenv.load_dotenv()

# driver = GraphDatabase.driver(
#     os.environ["NEO4J_URI"],
#     auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
# )

uri = "neo4j+s://332e43eb.databases.neo4j.io"  # Your Neo4j Aura URI
username = "neo4j"
password = "ZyvWu0bndBWMNu6lYlb5Fa3PkfsrWXes-gg0DPrAZLc"
driver = GraphDatabase.driver(uri, auth=(username, password))


def get_jina_embeddings(texts):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": os.environ["JINA_AUTHORIZATION_KEY"],
    }
    data = {
        "model": "jina-clip-v1",
        "normalized": True,
        "embedding_type": "float",
        "input": [{"text": text} for text in texts],
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        response_json = response.json()
        embeddings = [item["embedding"] for item in response_json["data"]]
        return embeddings
    else:
        raise Exception(
            f"Error fetching embeddings: {response.status_code}, {response.text}"
        )


def perform_similarity_search(query_embedding, user_type, user_id=None, top_k=10):
    with driver.session() as session:
        if user_type == "common":
            result = session.run(
                """
                MATCH (c:Common)-[:HAS_DOCUMENT]->(d:Document)
                RETURN d.id AS id, d.content AS content, d.embedding AS embedding
            """
            )
        else:
            result = session.run(
                """
                MATCH (u:User)-[:HAS_USER]->(userID:UserID {id: $user_id})-[:HAS_DOCUMENT]->(d:Document)
                RETURN d.id AS id, d.content AS content, d.embedding AS embedding
            """,
                user_id=user_id,
            )

        documents = []
        embeddings = []

        for record in result:
            documents.append((record["id"], record["content"]))
            embeddings.append(record["embedding"])

        embeddings = np.array(embeddings)
        print(embeddings.shape)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        distances = cosine_similarity(query_embedding, embeddings).flatten()

        # Get top_k results
        top_k_indices = np.argsort(distances)[-top_k:]
        top_docs = [
            (documents[i][0], documents[i][1], distances[i]) for i in top_k_indices
        ]
    driver.close()
    return sorted(top_docs, key=lambda x: x[2], reverse=True)


# Function to get relevant context from the similarity search
def get_relevant_context(query_embedding, user_type, user_id=None):
    results = perform_similarity_search(query_embedding, user_type, user_id)
    context = ""
    for id, content, _ in results:
        context += content + "\n"
    return context


def store_embeddings_in_neo4j(documents, embeddings, user_id=None):
    with driver.session() as session:
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            session.run(
                """
                MERGE (u:User)
                MERGE (userID:UserID {id: $user_id})
                CREATE (d:Document {id: $id, content: $content, embedding: $embedding})
                MERGE (u)-[:HAS_USER]->(userID)
                MERGE (userID)-[:HAS_DOCUMENT]->(d)
                    """,
                user_id=user_id,
                id=i,
                content=doc.page_content,
                embedding=embedding,
            )
    driver.close()


def get_prompt(context, query_text):
    return f"""
    You are a helpful assistant with access to specific documents. Please follow these guidelines:
    
    0. *Output*: Should be descriptive with respect to the question in three (3) lines.

    1. *Contextual Relevance*: Only provide answers based on the provided context. If the query does not relate to the context or if there is no relevant information, respond with "The query is not relevant to the provided context."

    2. *Language and Behavior*: Ensure that your response is polite and respectful. Avoid using any inappropriate language or making offensive statements.

    3. *Content Limitations*: Do not use or refer to any external data beyond the context provided.

    *Context*: {context}

    *Question*: {query_text}

    *Answer*:
    """
