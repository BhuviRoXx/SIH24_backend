from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq

import os
import dotenv

dotenv.load_dotenv()


class RouteSummQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    routeoutput: Literal["neo4j_user_node", "generate"] = Field(
        ...,
        description="Given a user query route to user node to query from knowledge graph or summarize the document.",
    )


def obtain_summ_usernode_router():
    """
    Route the query based on whether the user asks to summarize or asks a question.

    Args:
        query (str): The user's query (either a question or request for summarization).
        documents (List[str]): A list of documents that may need summarization.

    Returns:
        str: The action or result based on query routing (either summarize or answer question).
    """

    model = ChatGroq(
        temperature=0, model_name="gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]
    )

    structured_llm_router = model.with_structured_output(RouteSummQuery)

    system = """You are a specialized routing agent in a question-answering system. Your task is to determine whether a user's query should be directed to the knowledge graph (neo4j_user_node) or to the document summarization module (generate).
Follow these guidelines to make your decision:

Route to "neo4j_user_node" if:

The query is a specific question that could be answered from a structured knowledge base.
The query asks for factual information, definitions, or relationships between entities.
The query doesn't explicitly mention summarization or document analysis.


Route to "generate" if:

The query explicitly asks for a summary or overview of a document or topic.
The query requires analysis or synthesis of information from multiple sources.
The query is about the content of a specific document or set of documents.



Examples:

"What are the key features of our latest product?" -> neo4j_user_node
"Can you summarize the main points of the attached document?" -> generate
"Who is the CEO of our company?" -> neo4j_user_node
"What are the highlights from the quarterly report?" -> generate
"What is our company's policy on remote work?" -> neo4j_user_node
"Give me an overview of the project timeline from the attached PDF." -> generate

Your output should be either "neo4j_user_node" or "generate" based on your analysis of the user's query.
        """
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    return route_prompt | structured_llm_router
