from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
import requests
import pdfplumber

# Initialize Neo4j Aura connection
uri = "neo4j+s://332e43eb.databases.neo4j.io"  # Your Neo4j Aura URI
username = "neo4j"
password = "ZyvWu0bndBWMNu6lYlb5Fa3PkfsrWXes-gg0DPrAZLc"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Initialize Gemini
GOOGLE_API_KEY = "AIzaSyC5gv15479xiPka5pH4iYgphdPyrFKDuz4"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Function to get embeddings from Jina AI API
def get_jina_embeddings(texts):
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer jina_6f5c954572b64f59b9981f7e3c0bf550lTiWT5sW_gyTkoFQEaHQTRwZhmlW'
    }
    data = {
        'model': 'jina-clip-v1',
        'normalized': True,
        'embedding_type': 'float',
        'input': [{'text': text} for text in texts]
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        response_json = response.json()
        embeddings = [item['embedding'] for item in response_json['data']]
        return embeddings
    else:
        raise Exception(f"Error fetching embeddings: {response.status_code}, {response.text}")

# Function to create Neo4j nodes for each document and store embeddings
def store_embeddings_in_neo4j(documents, embeddings, user_type, user_id=None):
    with driver.session() as session:
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            if user_type == 'common':
                session.run("""
                    MERGE (c:Common)
                    CREATE (d:Document {id: $id, content: $content, embedding: $embedding})
                    MERGE (c)-[:HAS_DOCUMENT]->(d)
                    """, id=i, content=doc.page_content, embedding=embedding)
            else:
                # Create User node and sub-node with user ID
                session.run("""
                    MERGE (u:User)
                    MERGE (userID:UserID {id: $user_id})
                    CREATE (d:Document {id: $id, content: $content, embedding: $embedding})
                    MERGE (u)-[:HAS_USER]->(userID)
                    MERGE (userID)-[:HAS_DOCUMENT]->(d)
                    """, user_id=user_id, id=i, content=doc.page_content, embedding=embedding)

# Function to perform similarity search using cosine similarity
def perform_similarity_search(query_embedding, user_type, user_id=None, top_k=10):
    with driver.session() as session:
        if user_type == 'common':
            result = session.run("""
                MATCH (c:Common)-[:HAS_DOCUMENT]->(d:Document)
                RETURN d.id AS id, d.content AS content, d.embedding AS embedding
            """)
        else:
            result = session.run("""
                MATCH (u:User)-[:HAS_USER]->(userID:UserID {id: $user_id})-[:HAS_DOCUMENT]->(d:Document)
                RETURN d.id AS id, d.content AS content, d.embedding AS embedding
            """, user_id=user_id)
        
        documents = []
        embeddings = []

        for record in result:
            documents.append((record['id'], record['content']))
            embeddings.append(record['embedding'])

        embeddings = np.array(embeddings)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        distances = cosine_similarity(query_embedding, embeddings).flatten()

        # Get top_k results
        top_k_indices = np.argsort(distances)[-top_k:]
        top_docs = [(documents[i][0], documents[i][1], distances[i]) for i in top_k_indices]

    return sorted(top_docs, key=lambda x: x[2], reverse=True)

# Function to get relevant context from the similarity search
def get_relevant_context(query_embedding, user_type, user_id=None):
    results = perform_similarity_search(query_embedding, user_type, user_id)
    context = ""
    for id, content, _ in results:
        context += content + "\n"
    return context

# Function to generate answer using Gemini
def generate_answer(prompt):
    response = model.generate_content(prompt)
    return response.text

# Function to process and store PDF documents
def process_and_store_pdf(file_path, user_type, user_id=None):
    documents = []

    # Extract text from each page of the PDF using pdfplumber
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            raw_text = page.extract_text()
            if raw_text:  # If there is text on the page, create a Document object
                document = Document(page_content=raw_text)
                documents.append(document)

    # Split the document into chunks using the text splitter
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=80)
    documents = text_splitter.split_documents(documents)

    # Get the text content from documents to pass to Jina AI
    texts = [doc.page_content for doc in documents]

    # Fetch embeddings using Jina AI
    embeddings = get_jina_embeddings(texts)
    print("Embeddings fetched.")

    # Store the document embeddings in Neo4j
    store_embeddings_in_neo4j(documents, embeddings, user_type, user_id)
    print("Embeddings stored in Neo4j.")

def chatbot(user_type, user_id):
    query_text = input("Enter Query: ")
    query_embedding = get_jina_embeddings([query_text])[0]  # Take the first embedding from the response

    # Get relevant context from Neo4j
    context = get_relevant_context(query_embedding, user_type, user_id)

    # Generate the response using Gemini
    prompt = f"""
    You are a helpful assistant with access to specific documents. Please follow these guidelines:

    If there is any bad language, you should first give an warning and procced with the output
    
    0. **Output**: Should be descriptive with respect to the question in three (3) lines.

    1. **Contextual Relevance**: Only provide answers based on the provided context. If the query does not relate to the context or if there is no relevant information, respond with "The query is not relevant to the provided context."

    2. **Language and Behavior**: Ensure that your response is polite and respectful. Avoid using any inappropriate language or making offensive statements.

    3. **Content Limitations**: Do not use or refer to any external data beyond the context provided.

    **Context**: {context}

    **Question**: {query_text}

    **Answer**:
    """
    answer = generate_answer(prompt)

    print("Generated Answer:")
    print(answer)
    
    
# Main workflow
def main():
    # Example of setting user type and ID
    user_type = "user"  # or set to "common" if applicable
    user_id = "user1"  # set to the specific user ID if user_type is "user"

    # Uncomment the following line to process and store PDF documents
    # process_and_store_pdf('UNIT 1.pdf', user_type, user_id)
    chatbot(user_type, user_id)
    driver.close()
    


    # Get query embedding using Jina AI
    

if __name__ == "__main__":
    main()