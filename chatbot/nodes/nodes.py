from email.mime.text import MIMEText
import smtplib
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from neo4j import GraphDatabase
import pdfplumber
from io import BytesIO
import dotenv, os
from datetime import datetime, timedelta
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import json
import pytz
import re
import os
import pickle
import time
from models.route_query import obtain_question_router
from models.model_generation import obtain_rag_chain
from models.route_summ_query import obtain_summ_usernode_router

import whisper
from moviepy.editor import VideoFileClip

model_audio = whisper.load_model("small")

from utils.utils import (
    get_jina_embeddings,
    get_relevant_context,
    store_embeddings_in_neo4j,
    generate_ticket_id,
)

dotenv.load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the generative model
model = genai.GenerativeModel("gemini-pro")

global response


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

    pattern = r"^Schedule meeting @(\d{1,2}:\d{2}\s(?:AM|PM))\s(\d{2}/\d{2}/\d{4})\swith\s((?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,},?\s*)+)about\s(.+)$"

    match = re.match(pattern, state["question"])

    if match:
        time, date, emails, subject = match.groups()
        return "schedule_meeting"

    question_router = obtain_question_router()

    source = question_router.invoke({"question": question})
    if source.datasource == "user_node":
        print("---ROUTE QUERY TO USER NODE IN NEO4J---")
        return "user_node"
    elif source.datasource == "common_node":
        print("---ROUTE QUERY TO COMMON NODE IN NEO4J---")
        return "common_node"
    elif source.datasource == "tech_support":
        print("---ROUTE QUERY TO TECH SUPPORT---")
        return "tech_support"
    elif source.datasource == "bad_language":
        print("---ROUTE QUERY TO BAD LANGUAGE NODE---")
        return "bad_language"


def bad_language(state):
    """
    Route question to bad language node.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---BAD LANGUAGE NODE---")
    return {
        "generation": "The question contains offensive language. Please rephrase your query."
    }


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


def route_after_breakpoint(state):
    decision = state["decision"]
    breakpoint = state["breakpoint"]

    if decision == "proceed":
        if breakpoint == "1":
            return "breakpoint_2"
        elif breakpoint == "2":
            return "breakpoint_3"
        elif breakpoint == "3":
            return "final_node"
    elif decision == "retry":
        return f"breakpoint_{breakpoint}"
    else:
        return "END"


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

    print(documents)

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

    # rag_chain = obtain_rag_chain()
    # # RAG generation
    # generation = rag_chain.invoke({"context": documents, "question": question})

    prompt = f"""
    You are a helpful assistant with access to specific documents. Please follow these guidelines:
    
    0. **Output**: Should be descriptive with respect to the question in three (3) lines.

    1. **Contextual Relevance**: Only provide answers based on the provided context. If the query does not relate to the context or if there is no relevant information, respond with "The query is not relevant to the provided context."

    2. **Language and Behavior**: Ensure that your response is polite and respectful. Avoid using any inappropriate language or making offensive statements.

    3. **Content Limitations**: Do not use or refer to any external data beyond the context provided.

    **Context**: {state['documents']}

    **Question**: {state['question']}

    **Answer**:

    Return your answer in Markdown format with bolded headings, italics and underlines etc. as necessary.
    Use as much markdown as possible to format your response.
    Use ## for headings and ``` code blocks for code.```
    """
    response = model.generate_content(prompt)

    return {
        "documents": documents,
        "question": question,
        "generation": response.parts[0].text,
    }


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
    audio_filename = video.filename[:-4]
    # video_path = f"../_videos/{video.filename}"
    video_path = f"C:/Users/rajku/OneDrive/Documents/ClePro/HACKATHON/SIH24_backend/chatbot/_videos/{video.filename}"
    output_audio_path = f"C:/Users/rajku/OneDrive/Documents/ClePro/HACKATHON/SIH24_backend/chatbot/_audio/"

    # Load the video file
    video_clip = VideoFileClip(video_path)

    # Extract the audio from the video
    audio_clip = video_clip.audio

    # Save the extracted audio to a file
    audio_clip.write_audiofile(output_audio_path, codec="mp3")

    # Close the clips
    audio_clip.close()
    video_clip.close()

    transcribed_text = model_audio.transcribe("output_audio.mp3")

    return {"question": state["question"], "documents": transcribed_text}


def tech_support(state):
    troubleshooting_prompt = (
        f"Please provide at least three detailed troubleshooting steps for addressing  issues related to '{state['question']}'. "
        f"Format your response with clear instructions, starting with 'Please try the following troubleshooting steps:'"
    )

    output = model.generate_content(troubleshooting_prompt)
    response = output.parts[0].text

    return {"generation": response + "Would you like to escalate this issue?"}


def send_email(state):
    sender_email = "cletocite@gmail.com"
    receiver_email = "cletocite.techs@gmail.com"
    sender_password = "dxkbhzyaqaqcgrrq"  # App password or your email password

    ticket_id = generate_ticket_id()

    subject = f"TECH SUPPORT - TROUBLESHOOT - {ticket_id}"
    body = (
        f"Dear Tech Support Team,\n\n"
        f"Please find the details of the tech support request below:\n\n"
        f"User ID: {state['user_id']}\n"
        f"Ticket ID: {ticket_id}\n"
        # f"Priority: {priority}\n\n"
        # f"Support Type: {support_type}\n"
        # f"Issue Description: {issue_description}\n\n"
        f"Troubleshooting Steps Taken:\n{state['generation']}\n\n"
        f"Please review the provided information and take the necessary actions.\n\n"
        f"Thank you,\n"
        f"Tech Support Bot"
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")

    return {"generation": state["generation"]}


def schedule_meeting(state):
    pass


def meeting_shu(state):
    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    TOKEN_FILE = "token.pickle"
    CREDENTIALS_FILE = "C:\\Users\\rajku\\OneDrive\\Documents\\ClePro\\HACKATHON\\SIH24_backend\\chatbot\\nodes\\cred.json"

    def generate_answer(prompt):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

    def extract_meeting_details(text):
        prompt = f"""
        Extract the following details from the text:
        - Date of the meeting (in DD/MM/YYYY format or 'today' or 'tomorrow')
        - Time of the meeting (in 12-hour format HH:MM AM/PM)
        - List of attendees (as a list of email addresses)
        - Summary of the meeting

        Provide the extracted details as a JSON object with the following keys:
        - "date": The date of the meeting.
        - "time": The time of the meeting.
        - "attendees": A list of email addresses.
        - "summary": A brief summary of the meeting.

        Text: "{text}"
        """
        answer = generate_answer(prompt)
        print(f'Gemini response: "{answer}"')  # Include quotes for better visibility

        if not answer:
            print("No response from Gemini.")
            return None

        # Clean the response
        cleaned_answer = re.sub(
            r"^```json", "", answer
        )  # Remove start code block delimiter
        cleaned_answer = re.sub(
            r"```$", "", cleaned_answer
        )  # Remove end code block delimiter
        cleaned_answer = cleaned_answer.strip()
        print(f'Cleaned response: "{cleaned_answer}"')  # Debugging line

        try:
            details = json.loads(cleaned_answer)
            if not isinstance(details, dict):
                raise ValueError("Response is not a JSON object")
            print(f"Parsed details: {details}")  # Debugging line
            return details
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return None
        except ValueError as e:
            print(f"Value error: {e}")
            return None

    def authenticate_google_calendar():
        creds = None
        # Check if token.pickle exists (stored token for re-use)
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "rb") as token:
                creds = pickle.load(token)

        # If there are no valid credentials available, prompt the user to log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(TOKEN_FILE, "wb") as token:
                pickle.dump(creds, token)

        service = build("calendar", "v3", credentials=creds)
        return service

    def convert_date_and_time(date_str, time_str):
        now = datetime.now(pytz.timezone("Asia/Kolkata"))

        if date_str.lower() == "today":
            meeting_date = now.date()
        elif date_str.lower() == "tomorrow":
            meeting_date = now.date() + timedelta(days=1)
        else:
            try:
                meeting_date = datetime.strptime(date_str, "%d/%m/%Y").date()
            except ValueError:
                print(f"Invalid date format: {date_str}")
                return None, None

        try:
            meeting_time = datetime.strptime(time_str, "%I:%M %p").strftime("%H:%M")
        except ValueError:
            print(f"Invalid time format: {time_str}")
            return None, None

        start_time = datetime.combine(
            meeting_date, datetime.strptime(meeting_time, "%H:%M").time()
        )
        end_time = start_time + timedelta(hours=1)

        print(f"Converted start time: {start_time.isoformat()}")
        print(f"Converted end time: {end_time.isoformat()}")

        return start_time.isoformat(), end_time.isoformat()

    def schedule_meeting(service, start_time, end_time, attendees, summary="Meeting"):
        event = {
            "summary": summary,
            "start": {
                "dateTime": start_time,
                "timeZone": "Asia/Kolkata",  # Change this to your timezone
            },
            "end": {
                "dateTime": end_time,
                "timeZone": "Asia/Kolkata",  # Change this to your timezone
            },
            "attendees": [{"email": email} for email in attendees],
            "reminders": {
                "useDefault": True,
            },
        }
        event = service.events().insert(calendarId="primary", body=event).execute()
        print(f'Event created: {event.get("htmlLink")}')

    def main_fun(user_input):
        details = extract_meeting_details(user_input)

        if details:
            print(f"Extracted details: {details}")  # Debugging line

            # Use correct keys from the parsed JSON
            meeting_date = details.get("date") or details.get("meeting_date")
            meeting_time = details.get("time") or details.get("meeting_time")
            attendees = details.get("attendees", [])
            summary = details.get("summary", "Meeting")

            if not meeting_date or not meeting_time:
                print("Date or time information missing.")
                return

            start_time, end_time = convert_date_and_time(meeting_date, meeting_time)

            if start_time and end_time:
                # Authenticate
                service = authenticate_google_calendar()

                # Retry loop for scheduling the meeting
                max_retries = 5
                retry_delay = 5  # seconds
                for attempt in range(max_retries):
                    try:
                        schedule_meeting(
                            service, start_time, end_time, attendees, summary
                        )
                        print("Event created successfully.")
                        break  # Exit loop if successful
                    except Exception as e:
                        print(f"Error creating event: {e}")
                        if attempt < max_retries - 1:
                            print(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            print("Failed to create event after multiple attempts.")
            else:
                print("Failed to convert date and time.")
        else:
            print("Failed to extract meeting details.")

    main_fun(
        "Schedule meeting @10:30 AM 10/09/2024 with gowsrini2004@gmail.com,idhikaprabakaran@gmail.com, forfungowtham@gmail.com, cowara987@gmail.com about SIH INTERNAL HACAKTHON"
    )


def hierachy(state):
    NEO4J_URI = "neo4j+s://2cbd2ddb.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "W_OwGl8HD0XAHkvFoDWf93ZNpyCf-efTsEGcmgLVU_k"

    # GOOGLE_API_KEY = "AIzaSyC5gv15479xiPka5pH4iYgphdPyrFKDuz4"
    class Neo4jClient:
        def __init__(self, uri, user, password):
            self.driver = GraphDatabase.driver(uri, auth=(user, password))

        def close(self):
            self.driver.close()

        def fetch_graph_data(self):
            query = """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n, r, m
            """
            with self.driver.session() as session:
                result = session.run(query)
                data = []
                for record in result:
                    nodes = (record["n"], record["m"])
                    relationships = record["r"]
                    data.append({"nodes": nodes, "relationships": relationships})
                return data

    def generate_answer_graph(prompt):
        response = model.generate_content(prompt)
        return response.text

    def main_fun(person):
        client = Neo4jClient(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        try:
            graph_data = client.fetch_graph_data()
            # Convert graph data to a textual representation
            graph_document = str(graph_data)

            # person = input("Enter the person's name: ")
            prompt = f"""
        These are the nodes and relationships of the graph:
        document = {graph_document}

        Provide the hierarchy for the person '{person}'. The hierarchy should trace their position up to the CEO, including all managers and seniors they report to. Format the output as follows:

        His desigination and folowing by Reports to - Name - Desiginamtion and try to indent it with there position.

        Use indentation to reflect the reporting structure. Please ensure the output is clear and organized, without any bold or special formatting.
        """

            answer = generate_answer_graph(prompt)
            print(answer)
        finally:
            client.close()

    main_fun("Devesh")
