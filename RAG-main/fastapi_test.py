from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware  # Correct CORS import
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import google.generativeai as genai
import uvicorn
import aiofiles
from pydantic import BaseModel


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)



# raw_prompt = PromptTemplate.from_template(
#     """ 
#     <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information, say so. [/INST] </s>
#     [INST] {input}
#            Context: {context}
#            Answer:
#     [/INST]
# """
# )

folder_path = "RAG-main/db"
# embedding = FastEmbedEmbeddings()
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

GOOGLE_API_KEY = "AIzaSyDRIIeaCaEu45HX2ykLe64EQcA9gH4RIzI"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

# Wrapper for Google chat model to make it compatible with LangChain
class GoogleChatWrapper:
    def __init__(self, chat_session):
        self.chat_session = chat_session

    def __call__(self, query):
        response = self.chat_session.send_message(query, stream=True)
        response_content = ""
        for chunk in response:
            if hasattr(chunk, 'text'):
                response_content += chunk.text
            else:
                response_content += str(chunk)
        return response_content

@app.post("/ai")
async def ai_post(request: Request):
    json_content = await request.json()
    query = json_content.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is missing")

    try:
        response = chat.send_message(query, stream=True)
        response_content = "".join([chunk.text if hasattr(chunk, 'text') else str(chunk) for chunk in response])
        return JSONResponse(content={"answer": response_content})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/pdf")
async def pdf_post(file: UploadFile = File(...)):
    file_name = file.filename
    save_file = f"RAG-main/pdf/{file_name}"
    
    async with aiofiles.open(save_file, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)

    try:
        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()
        chunks = text_splitter.split_documents(docs)

        vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=folder_path)
        vector_store.persist()

        return {
            "status": "Successfully Uploaded",
            "filename": file_name,
            "doc_len": len(docs),
            "chunks": len(chunks),
        }
    except Exception as e:
        return JSONResponse(content={"status": "Error", "message": str(e)}, status_code=500)

# @app.post("/ask_pdf")
# async def ask_pdf_post(request: Request):
#     json_content = await request.json()
#     query = json_content.get("query")

#     try:
#         vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding_function)
#         retriever = vector_store.as_retriever(
#             search_type="similarity_score_threshold",
#             search_kwargs={"k": 6, "score_threshold": 0.1},
#         )

#         google_chat_model = GoogleChatWrapper(chat)
#         document_chain = create_stuff_documents_chain(google_chat_model, raw_prompt)
#         chain = create_retrieval_chain(retriever, document_chain)

#         result = chain.invoke({"input": query})

#         sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]

#         return {"answer": result["answer"], "sources": sources}
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)




# Define your prompt template (you may need to adjust or replace this with your actual prompt format)
def generate_rag_prompt(query, context):
    return f"Query: {query}\nContext: {context}\nAnswer:"

class QueryModel(BaseModel):
    query: str

@app.post("/ask_pdf")
async def ask(query: QueryModel):
    try:
        # Retrieve relevant context from the vector store
        context = get_relevant_context_from_db(query.query)
        
        # Generate the RAG prompt
        prompt = generate_rag_prompt(query=query.query, context=context)
        
        # Generate the answer using the generative model
        answer = generate_answer(prompt)
        
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def get_relevant_context_from_db(query):
    context = ""
    vector_db = Chroma(persist_directory=folder_path, embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context

def generate_answer(prompt):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
