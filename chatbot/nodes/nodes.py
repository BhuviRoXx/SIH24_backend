from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
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

from models.route_query import obtain_question_router
from models.model_generation import obtain_rag_chain
from models.route_summ_query import obtain_summ_usernode_router


from utils.utils import (
    get_jina_embeddings,
    get_relevant_context,
    store_embeddings_in_neo4j,
    get_prompt,
)


def route(state):
    """
    Route question to to retrieve information from the user or common node in neo4j.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]

    if state["pdf"] != None:
        return "update_knowledge_graph"
    elif state["video"] != None:
        return "video_processing"

    question_router = obtain_question_router()

    source = question_router.invoke({"question": question})
    if source.datasource == "user_node":
        print("---ROUTE QUERY TO USER NODE IN NEO4J---")
        return "user_node"
    elif source.datasource == "common_node":
        print("---ROUTE QUERY TO COMMON NODE IN NEO4J---")
        return "common_node"


def route_summarization_usernode(state):
    """
    Route summarization user node based on the given state.
    Args:
        state (dict): The state containing the question.
    Returns:
        dict: A dictionary indicating the next node and any additional data.
    """
    print("---ROUTE SUMMARIZATION USER NODE---")
    question = state["question"]
    summ_usernode_router = obtain_summ_usernode_router()
    source = summ_usernode_router.invoke({"question": question})

    if source.routeoutput == "neo4j_user_node":
        print("---ROUTE QUERY TO USER NODE IN NEO4J---")
        return {"next": "neo4j_user_node", "question": question}
    elif source.routeoutput == "generate":
        print("---ROUTE QUERY TO GENERATE---")
        print(source.routeoutput)
        return {"next": "summarize", "question": question}


def neo4j_user_node(state):
    """
    Retrieves relevant documents from Neo4j database based on the user's query.
    Args:
        state (dict): The state containing the user's question and user ID.
    Returns:
        dict: A dictionary containing the retrieved documents and the original question.
    """

    query = state["question"]
    user_id = state["user_id"]
    query_embedding = get_jina_embeddings([query])[0]

    documents = get_relevant_context(query_embedding, "user", user_id)

    return {"documents": documents, "question": query}


def neo4j_common_node(state):
    """
    Executes a Neo4j query to retrieve relevant documents based on the given state.
    Args:
        state (dict): The state containing the question and user ID.
    Returns:
        dict: A dictionary containing the retrieved documents and the original question.
    """

    query = state["question"]
    # user_id = state["user_id"]
    query_embedding = get_jina_embeddings([query])[0]

    documents = get_relevant_context(query_embedding, "common")

    return {"documents": documents, "question": query}


def generate(state):
    """
    Generate answer from retrieved documentation.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    rag_chain = obtain_rag_chain()
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def summarize(state):
    """
    Summarize the retrieved documents.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with summarization key
    """
    print("---SUMMARIZE---")
    documents = []
    file_path = f"_files/{state['pdf'].filename}"
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            raw_text = page.extract_text()
            if raw_text:
                document = Document(page_content=raw_text)
                documents.append(document)

    return {"question": state["question"], "documents": documents}


def update_knowledge_graph(state):

    # pdf_file = BytesIO(state["pdf"])
    print("---UPDATE KNOWLEDGE GRAPH---")
    # Process the PDF
    documents = []
    file_path = f"_files/{state['pdf'].filename}"
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            raw_text = page.extract_text()
            if raw_text:
                document = Document(page_content=raw_text)
                documents.append(document)

    # Split document
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=80)
    documents = text_splitter.split_documents(documents)

    # documents = text_splitter.split_text(state["pdf"])

    # Get embeddings
    texts = [doc.page_content for doc in documents]
    embeddings = get_jina_embeddings(texts)

    # Store embeddings in Neo4j
    store_embeddings_in_neo4j(documents, embeddings, state["user_id"])

    return {"user_id": state["user_id"], "question": state["question"]}


def video_processing(state):
    """
    Process the video and extract text.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with video key
    """
    print("---VIDEO PROCESSING---")
    video = state["video"]

    video_loader = YoutubeLoader(video)
    video_text = video_loader.load_text()

    return {"question": state["question"], "documents": video_text}


def human_in_the_loop(state):
    """
    Check if human intervention is required.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with human_in_the_loop key

    """
    print("---HUMAN IN THE LOOP---")
    generation = state["generation"]
    documents = state["documents"]
    question = state["question"]

    check_human_interaction = human_in_the_loop.check_human_interaction()
    human_in_the_loop = human_in_the_loop.invoke(
        {"generation": generation, "documents": documents, "question": question}
    )
    return human_in_the_loop.binary_score


def review_generation(state):
    pass


def regenerate():
    pass
