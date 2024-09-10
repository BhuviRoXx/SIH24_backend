from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io
import uuid
import mysql.connector
from typing import List
import uvicorn
from typing import Optional
from pprint import pprint
from pdfminer.high_level import extract_text
from graph.graph import graph

app = FastAPI()

# Create directories if not exist
os.makedirs('files', exist_ok=True)
os.makedirs('videos', exist_ok=True)

# CORS configuration
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MySQL configuration
db_config = {
    'user': 'root',
    'password': 'CowTheGreat',
    'host': 'localhost',
    'database': 'sih'
}

# Create a database connection
def get_db_connection():
    return mysql.connector.connect(**db_config)

# Pydantic models
# class MessageRequest(BaseModel):
#     session_id: str
#     session_title: str = "Default Title"
#     query: str

class UpdateSessionTitleRequest(BaseModel):
    session_id: str
    new_title: str

# Route to generate a new session ID
@app.get("/session")
async def generate_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

# Route to handle incoming messages and bot response
class QueryRequest(BaseModel):
    user_id: str
    question: str
    pdf: Optional[UploadFile] = None
    video: Optional[UploadFile] = None


@app.post("/query")
async def receive_message(user_id: str = Form(...),
    question: str = Form(...),
    pdf: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")

    try:
        cursor = connection.cursor()

        # Insert the user's message into the database
        user_message_query = "INSERT INTO messages (session_id, session_title, sender, text) VALUES (%s, %s, %s, %s)"
        cursor.execute(user_message_query, (request.session_id, request.session_title, 'user', request.query))
        connection.commit()

        # Dummy bot response (you can replace this with AI response logic)
        graph_app = graph()
    # Access the uploaded files
        filename = Optional[str]
        if pdf:
            file_path = os.path.join("_files", pdf.filename)
            # Save the file
            with open(file_path, "wb") as f:
                content = await pdf.read()  # Read the file content asynchronously
                f.write(content)  # Write the file content to the defined path

            print("PDF content recieved")

        if video:
            video_path = os.path.join("_videos", video.filename)
            # Save the video file
            with open(video_path, "wb") as f:
                content = await video.read()  # Read the file content asynchronously
                f.write(content)  # Write the video content to the defined path

            print(f"Video content received and saved to {video_path}.")

        for output in graph_app.stream(
            {
                "user_id": user_id,
                "question": question,
                "pdf": pdf,
                "video": video,
            }
        ):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint("\n---\n")
            print("\n")

    # Final generation
        pprint(value["generation"])
        bot_reply = value["generation"]

        # Insert the bot's response into the database
        bot_message_query = "INSERT INTO messages (session_id, session_title, sender, text) VALUES (%s, %s, %s, %s)"
        cursor.execute(bot_message_query, (request.session_id, request.session_title, 'bot', bot_reply))
        connection.commit()

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cursor.close()
        connection.close()

    return {"answer": bot_reply}

# Route to retrieve all messages for a session
@app.get("/messages/{session_id}")
async def fetch_messages(session_id: str):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # Retrieve all messages for the session
    cursor.execute("SELECT sender, text FROM messages WHERE session_id = %s", (session_id,))
    messages = cursor.fetchall()

    cursor.close()
    connection.close()

    return {"messages": messages}

# Route to fetch unique session titles
@app.get("/sessions")
async def fetch_unique_sessions():
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Failed to connect to the database")

    try:
        cursor = connection.cursor(dictionary=True)
        # Fetch unique session titles
        cursor.execute("SELECT DISTINCT session_title FROM messages")
        sessions = cursor.fetchall()

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cursor.close()
        connection.close()

    return {"sessions": sessions}

# Route to update session title
@app.put("/update-session-title")
async def update_session_title(request: UpdateSessionTitleRequest):
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Update the session title in the database
        update_query = "UPDATE messages SET session_title = %s WHERE session_id = %s"
        cursor.execute(update_query, (request.new_title, request.session_id))
        connection.commit()

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Error updating session title: {err}")
    finally:
        cursor.close()
        connection.close()

    return {"message": "Session title updated successfully"}

# Route to fetch messages by session title
@app.get("/messages/title/{session_title}")
async def fetch_messages_by_title(session_title: str):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # Retrieve all messages for the session title
    cursor.execute("SELECT session_id, sender, text FROM messages WHERE session_title = %s", (session_title,))
    messages = cursor.fetchall()

    cursor.close()
    connection.close()

    return {"messages": messages}

# Route to handle PDF, video, and YouTube URL submission
@app.post("/submit")
async def submit_data(
    pdf: UploadFile = File(None), 
    video: UploadFile = File(None), 
    url: str = Form(None)
):
    pdf_path, video_path = None, None

    # Save PDF file
    if pdf:
        pdf_path = os.path.join('files', pdf.filename)
        with open(pdf_path, 'wb') as f:
            f.write(pdf.file.read())

    # Save video file
    if video:
        video_path = os.path.join('videos', video.filename)
        with open(video_path, 'wb') as f:
            f.write(video.file.read())

    response_message = {
        'message': 'Data received successfully',
        'pdf_path': pdf_path if pdf else 'No PDF uploaded',
        'video_path': video_path if video else 'No video uploaded',
        'youtube_url': url if url else 'No YouTube URL provided'
    }

    return response_message

# Run the FastAPI app using Uvicorn or Gunicorn if deployed on a server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
