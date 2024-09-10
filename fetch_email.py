import imaplib
from langchain.text_splitter import TokenTextSplitter
import email
from email.header import decode_header
import time
import requests
from neo4j import GraphDatabase

uri = "neo4j+s://332e43eb.databases.neo4j.io"  # Your Neo4j Aura URI
username = "neo4j"
password = "ZyvWu0bndBWMNu6lYlb5Fa3PkfsrWXes-gg0DPrAZLc"
driver = GraphDatabase.driver(uri, auth=(username, password))


def store_embeddings_in_neo4j(texts, embeddings, user_id=None):
    print("Starting to store embeddings in Neo4j...")
    with driver.session() as session:
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            print(f"Storing text {i} with embedding...")
            session.run("""
                MERGE (u:User)
                MERGE (userID:UserID {id: $user_id})
                CREATE (d:Document {id: $id, content: $content, embedding: $embedding})
                MERGE (u)-[:HAS_USER]->(userID)
                MERGE (userID)-[:HAS_DOCUMENT]->(d)
                """, user_id=user_id, id=i, content=text, embedding=embedding)
    print("Embeddings stored in Neo4j.")


def get_jina_embeddings(texts):
    print("Fetching embeddings from Jina API...")
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer jina_6f5c954572b64f59b9981f7e3c0bf550lTiWT5sW_gyTkoFQEaHQTRwZhmlW'
    }
    data = {
        'model': 'jina-clip-v1',
        'normalized': True,
        'embedding_type': 'float',
        'input': [{'text': text} for text in texts]
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        response_json = response.json()
        embeddings = [item['embedding'] for item in response_json['data']]
        print("Embeddings fetched successfully.")
        return embeddings
    else:
        raise Exception(f"Error fetching embeddings: {response.status_code}, {response.text}")


def process_and_store(text, user_id=None):
    print("Processing text...")
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=80)
    # Split text into chunks and keep them as a list of strings
    texts = text_splitter.split_text(text)  # Changed to split_text to work with raw strings
    print(f"Text split into {len(texts)} chunks.")
    embeddings = get_jina_embeddings(texts)
    print("Embeddings fetched.")
    store_embeddings_in_neo4j(texts, embeddings, user_id)
    print("Embeddings stored in Neo4j.")
    

def connect_to_gmail(username, password):
    print("Connecting to Gmail...")
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(username, password)
    print("Connected to Gmail.")
    return mail

def fetch_emails_from_label(mail, processed_emails, label="Cletocite"):
    print(f"Fetching emails from the '{label}' label...")
    text = ""
    mail.select(f'"{label}"')
    
    status, messages = mail.search(None, 'ALL')  # Fetch all emails with the label
    email_ids = messages[0].split()

    for num in email_ids:
        if num in processed_emails:
            continue
        
        print(f"Processing email ID: {num}")
        status, msg_data = mail.fetch(num, "(BODY.PEEK[])")  # Use PEEK to avoid marking as seen
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8")
                from_ = msg.get("From")
                print(f"From: {from_}")
                print(f"Subject: {subject}")
                text += f"From: {from_}\nSubject: {subject}\n\n"
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode("utf-8")
                            print(f"Body:\n{body}")
                            text += f"Body:\n{body}"
                else:
                    body = msg.get_payload(decode=True).decode("utf-8")
                    print(f"Body:\n{body}")
                    text += f"Body:\n{body}"
                main(text)

        print(f"Marking email ID {num} as processed.")
        processed_emails.add(num)
        mail.store(num, '+FLAGS', '\\Deleted')

    mail.expunge()  # Permanently remove the emails with the "Deleted" flag from the "Cletocite" label
    print("Finished fetching and processing emails from the label.")

def live_email_check(username, password, interval=5, label="Cletocite"):
    print(f"Starting live email check for the '{label}' label...")
    mail = connect_to_gmail(username, password)
    processed_emails = set()  # Store processed email IDs to avoid duplicates
    while True:
        fetch_emails_from_label(mail, processed_emails, label)
        print(f"Checking {label} label again in {interval} seconds...")
        time.sleep(interval)


def main(text):
    user_id = "user5"
    print(f"Main function called with user ID {user_id}.")
    process_and_store(text, user_id)

# Your Gmail credentials
username = "cletocite@gmail.com"
password = "dxkbhzyaqaqcgrrq"  # Use your app password

# Start live email check for the "Cletocite" label
live_email_check(username, password)