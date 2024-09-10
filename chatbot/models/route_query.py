from typing import Literal
# import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq

import os
import dotenv

dotenv.load_dotenv()
# GOOGLE_API_KEY = "AIzaSyC5gv15479xiPka5pH4iYgphdPyrFKDuz4"
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-pro')

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["common_node", "user_node"] = Field(
        ...,
        description="Given a user query route to either common node or current user node in knowledge graph.",
    )


def obtain_question_router():

    model = ChatGroq(
        temperature=0, model_name="gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]
    )

    structured_llm_router = model.with_structured_output(RouteQuery)

    system = """You are an expert routing agent for a company's knowledge graph system. Your task is to determine whether a user's query should be directed to the common node (for global company information) or the user node (for user-specific information) in the knowledge graph.
Follow these guidelines to make your decision:

Route to "common_node" if:

The query is about general company policies,HR polices, procedures,company events ,or global information.
The question relates to company-wide support, products, or services.
The inquiry is about standard practices, regulations, or information that applies to all employees or departments.
The query doesn't mention or imply any user-specific context.


Route to "user_node" if:

The query is about user-specific documents, data, or information.
The question relates to individual Quality Management System (QMS) data or processes.
The inquiry references personal projects, tasks, or user-uploaded documents.
The query implies or explicitly mentions individual or role-specific context.



Examples:

"What is the company's vacation policy?" -> common_node
"Can you show me the status of my current project?" -> user_node
"What are the steps for requesting IT support?" -> common_node
"What were the key points from the document I uploaded last week?" -> user_node
"Who should I contact for HR-related questions?" -> common_node
"What is the deadline for my team's current sprint?" -> user_node
"What are the company's core values?" -> common_node
"Can you summarize my performance review from last quarter?" -> user_node

Your output should only be either "common_node" or "user_node" based on your analysis of the user's query. Remember, route to the common node for global, company-wide information, and to the user node for personalized, user-specific data.
        """
        
    print(system)
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    print(route_prompt)

    return route_prompt | structured_llm_router
