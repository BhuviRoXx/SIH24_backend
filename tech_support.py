import smtplib
from email.mime.text import MIMEText
import google.generativeai as genai
import random

# Configure the Gemini API with your API key
GOOGLE_API_KEY = "AIzaSyC5gv15479xiPka5pH4iYgphdPyrFKDuz4"
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the generative model
model = genai.GenerativeModel('gemini-pro')

# Function to generate an answer using the model
def generate_answer(prompt):
    response = model.generate_content(prompt)
    return response.text

# Function to generate a ticket ID
def generate_ticket_id():
    return f"TICKET-{random.randint(1000, 9999)}"

# Function to send an email
def send_email(ticket_id, support_type, issue_description, troubleshooting_steps, user_id, priority):
    sender_email = "cletocite@gmail.com"
    receiver_email = "cletocite.techs@gmail.com"
    sender_password = "dxkbhzyaqaqcgrrq"  # App password or your email password

    subject = f"TECH SUPPORT - {support_type} - {ticket_id}"
    body = (
        f"Dear Tech Support Team,\n\n"
        f"Please find the details of the tech support request below:\n\n"
        f"User ID: {user_id}\n"
        f"Ticket ID: {ticket_id}\n"
        f"Priority: {priority}\n\n"
        f"Support Type: {support_type}\n"
        f"Issue Description: {issue_description}\n\n"
        f"Troubleshooting Steps Taken:\n{troubleshooting_steps}\n\n"
        f"Please review the provided information and take the necessary actions.\n\n"
        f"Thank you,\n"
        f"Tech Support Bot"
    )
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")

# Main chatbot logic for tech support
def tech_support_chat():
    print("Tech Support Chatbot started. Type 'exit' to quit.")
    
    while True:
        # Step 1: Ask the type of tech support needed
        user_input = input("Bot: What kind of tech support do you need? (Hardware, Software, Network): ")
        
        # Step 2: Ask for the specific issue
        issue_description = input(f"Bot: What is the issue with {user_input}? Please describe it briefly: ")
        
        # Step 3: Generate troubleshooting steps dynamically from Gemini
        troubleshooting_prompt = (
        f"Please provide at least three breif troubleshooting steps for addressing {user_input} issues related to '{issue_description}'. "
        f"Format your response with clear instructions, starting with 'Please try the following troubleshooting steps:'")
        troubleshooting_steps = generate_answer(troubleshooting_prompt)
        print(f"Bot: {troubleshooting_steps}")
        
        # Step 4: Ask if the solution resolved the issue
        feedback = input("Bot: Did this solution resolve your issue? (Yes or No): ")
        
        if feedback.lower() == "no":
            # Step 5: Ask if the issue should be escalated
            escalate = input("Bot: Would you like to escalate this issue? (Yes or No): ")
            
            if escalate.lower() == "yes":
                # Ask for the priority of the issue
                priority = input("Bot: What is the priority of this issue? (Low, Medium, High): ")
                
                # Generate a ticket ID
                ticket_id = generate_ticket_id()
                print(f"Bot: Your issue has been escalated. Ticket ID: {ticket_id}")
                
                # Get User ID (for simplicity, you can hard-code it or ask the user to input it)
                user_id = "user123"  # Replace with actual user ID logic
                
                # Send the email with the ticket details
                send_email(ticket_id, user_input, issue_description, troubleshooting_steps, user_id, priority)
            else:
                print("Bot: Thank you for using tech support!")
        else:
            print("Bot: Glad the issue was resolved!")
        
        # End chat after one interaction, or loop back for more queries
        more_help = input("Bot: Do you need further assistance? (Yes or No): ")
        if more_help.lower() == 'no':
            print("Bot: Thank you for using tech support!")
            break

# Start the tech support chatbot
tech_support_chat()