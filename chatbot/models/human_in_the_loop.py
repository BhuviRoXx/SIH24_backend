from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq

import os


class HumanInTheLoop(BaseModel):
    """Used to check if human intervention is required"""

    binary_score: str = Field(
        description="Response is relevant to the question, 'yes' or 'no'"
    )


def check_human_interaction():

    GROQ_API_KEY = "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"
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    model = ChatGroq(temperature=0, model_name="gemma2-9b-it")

    structured_llm_router = model.with_structured_output(HumanInTheLoop)

    system = """You have to decide now if human intervention is required.
        Human in the loop or Human intervention is required when you are required to use a tool to send an email or raise a ticket.
        Give a binary answer 'yes' or 'no' to answer whether human interaction is required or not.
        """
    human_loop_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    return human_loop_prompt | structured_llm_router
