from neo4j import GraphDatabase
import google.generativeai as genai








# Define the connection details
NEO4J_URI = "neo4j+s://2cbd2ddb.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "W_OwGl8HD0XAHkvFoDWf93ZNpyCf-efTsEGcmgLVU_k"
GOOGLE_API_KEY = "AIzaSyC5gv15479xiPka5pH4iYgphdPyrFKDuz4"

# Configure Google Gemini API
genai.configure(api_key="AIzaSyC5gv15479xiPka5pH4iYgphdPyrFKDuz4")
model = genai.GenerativeModel('gemini-pro')

# Create a class to handle Neo4j operations
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

def generate_answer(prompt):
    response = model.generate_content(prompt)
    return response.text

def main():
    client = Neo4jClient(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    try:
        graph_data = client.fetch_graph_data()
        # Convert graph data to a textual representation
        graph_document = str(graph_data)
        
        person = input("Enter the person's name: ")
        prompt = f"""
These are the nodes and relationships of the graph:
document = {graph_document}

Provide the hierarchy for the person '{person}'. The hierarchy should trace their position up to the CEO, including all managers and seniors they report to. Format the output as follows:

His desigination and folowing by Reports to - Name - Desiginamtion and try to indent it with there position.

Use indentation to reflect the reporting structure. Please ensure the output is clear and organized, without any bold or special formatting.
"""

        answer = generate_answer(prompt)
        print(answer)
    finally:
        client.close()

if __name__ == "__main__":
    main()