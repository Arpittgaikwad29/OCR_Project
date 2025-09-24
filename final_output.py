from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize Neo4j Graph
graph = Neo4jGraph(
    url="neo4j+s://a66a629b.databases.neo4j.io",
    username="neo4j",
    password="39Ak-WMIJZEh8oU2m7pXeN0ESFw74m1bogLfqsm3APY",
    database="neo4j"
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyC_yR_Z1BNvkL1ixrejs2aqvGHSXK2CwvE")

# Get schema information from the database - use as property not method
schema = graph.schema
print("Database Schema:")
print(schema)

# Query to get Patient1's health records
patient_query = """
MATCH (p:Patient {name: 'Patient1'})-[r]->(n)
RETURN p, r, n
"""
patient_data = graph.query(patient_query)
print("\nPatient Data:")
print(patient_data)

# Create a prompt template for the LLM
prompt_template = """
You are a medical AI assistant. Based on the following patient data from a Neo4j database, 
please analyze the health records for Patient1 and provide:
1. A brief summary of the patient's condition
2. A short prescription recommendation
3. Recommendations for further actions

Patient Data:
{patient_data}

Database Schema:
{schema}

Your response should be formatted with clear sections for Summary, Prescription, and Recommendations.
"""

prompt = PromptTemplate(
    input_variables=["patient_data", "schema"],
    template=prompt_template
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.invoke({
    "patient_data": str(patient_data), 
    "schema": schema
})

# Display results
print("\n📋 Summary + 💊 Prescription + ✅ Recommendations:")
print(result['text'])