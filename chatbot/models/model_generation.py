from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from utils.utils import get_prompt
import os


def obtain_rag_chain():

    prompt = hub.pull("rlm/rag-prompt")

    llm = ChatGroq(
        temperature=0, model_name="gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]
    )

    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain
