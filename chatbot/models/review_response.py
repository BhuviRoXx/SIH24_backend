from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq

import os


class ReviewResponse(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Response is relevant to the question, 'yes' or 'no'"
    )


def obtain_retrivel_reviewer():

    GROQ_API_KEY = "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"
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    ...

    model = ChatGroq(temperature=0, model_name="gemma2-9b-it")

    structured_llm_router = model.with_structured_output(ReviewResponse)

    system = """"You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    return grade_prompt | structured_llm_router
