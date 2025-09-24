import os
from google.generativeai import GenerativeModel, configure
from neo4j import GraphDatabase

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyC_yR_Z1BNvkL1ixrejs2aqvGHSXK2CwvE"
configure(api_key=os.environ["GOOGLE_API_KEY"])

# Gemini model
model = GenerativeModel("gemini-2.0-flash")

# Your input medical text
input_text = """*   **Patient:** Kiran Agoural
*   **Age:** 69
*   **Sex:** Female (F)
*   **Diagnosis/Symptoms:** Dysphagia (cough)
*   **Medical History:** Cab (Possible abbreviation for a condition)
*   **Medications/Tests:** COC, TSH, T3 T4, SUPT Creat CAP
*   **Recommendations:** Liquid diet"""

# Prompt to Gemini
prompt = f"""
You are a Cypher query generator for Neo4j.

1. First, generate a Cypher query to delete all existing nodes and relationships.
2. Then, from the given medical text, extract medically relevant information (e.g., diagnosis, medications, observations, treatment plan) and generate Cypher queries to insert them into the graph.
3. Do NOT include any personal info like name, birthdate, or caste.
4. Output only raw Cypher queries without triple backticks or extra explanations.

Text:
{input_text}
"""

# Generate the response
response = model.generate_content(prompt)

# Strip code blocks if present
raw_cypher = response.text.strip()
if raw_cypher.startswith("```"):
    raw_cypher = "\n".join(line for line in raw_cypher.splitlines() if not line.startswith("```"))

# Connect to Neo4j
uri = "neo4j+s://a66a629b.databases.neo4j.io"
username = "neo4j"
password = "39Ak-WMIJZEh8oU2m7pXeN0ESFw74m1bogLfqsm3APY"

driver = GraphDatabase.driver(uri, auth=(username, password))

# Function to run Cypher queries
def run_cypher_queries(driver, queries):
    with driver.session() as session:
        for query in queries.split(";"):
            query = query.strip()
            if query:
                print("Running query:", query)
                session.run(query)

# Run the cleaned Cypher queries
run_cypher_queries(driver, raw_cypher)

# Close driver
driver.close()