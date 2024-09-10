from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from utils.utils import get_prompt
import os


def obtain_rag_chain(document,question):

    # prompt = hub.pull("rlm/rag-prompt")

    # llm = ChatGroq(
    #     temperature=0, model_name="gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]
    # )
    prompt = f"""
    You are a helpful assistant with access to specific documents. Please follow these guidelines:

    If there is any bad language, you should first give an strong warning and procced with the output
    
    0. **Output**: Should be descriptive with respect to the question in three (3) lines.

    1. **Contextual Relevance**: Only provide answers based on the provided context. If the query does not relate to the context or if there is no relevant information, respond with "The query is not relevant to the provided context."

    2. **Language and Behavior**: Ensure that your response is polite and respectful. Avoid using any inappropriate language or making offensive statements.

    3. **Content Limitations**: Do not use or refer to any external data beyond the context provided.

    **Context**: {document}

    **Question**: {question}

    **Answer**:
    """


    # rag_chain = prompt | llm | StrOutputParser()

    return prompt