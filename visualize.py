import streamlit as st
import os
import io
import tempfile
from PIL import Image
from google.generativeai import GenerativeModel, configure
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import graphviz

# Set page configuration
st.set_page_config(page_title="Medical Document Processing System", layout="wide")

# Google API key setup
GOOGLE_API_KEY = "AIzaSyC_yR_Z1BNvkL1ixrejs2aqvGHSXK2CwvE"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
configure(api_key=GOOGLE_API_KEY)

# Neo4j credentials
NEO4J_URI = "neo4j+s://a66a629b.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "39Ak-WMIJZEh8oU2m7pXeN0ESFw74m1bogLfqsm3APY"

# Initialize Gemini models
flash_model = GenerativeModel("gemini-2.0-flash")

# Create flow chart of the process
def create_flow_chart():
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB', size='8,5', dpi='300')
    
    # Node styling
    graph.attr('node', shape='box', style='filled', color='#4285F4', 
               fontname='Arial', fontsize='12', fontcolor='white')
    
    # Add nodes
    graph.node('input', 'User Uploads Medical Document Image', fillcolor='#34A853')
    graph.node('agent1', 'Agent 1: OCR Text Extraction\nExtract text from image', fillcolor='#4285F4')
    graph.node('agent2', 'Agent 2: Text Correction\nFix spelling mistakes', fillcolor='#4285F4')
    graph.node('agent3', 'Agent 3: Text Understanding\nExtract medical information', fillcolor='#4285F4')
    graph.node('agent4', 'Agent 4: FHIR Format Conversion\nStandardize to healthcare JSON', fillcolor='#4285F4')
    graph.node('agent5', 'Agent 5: Graph Database Creation\nStore in Neo4j database', fillcolor='#4285F4')
    graph.node('agent6', 'Agent 6: Final Response Generation\nGenerate medical analysis', fillcolor='#EA4335')
    
    # Add edges
    graph.edge('input', 'agent1')
    graph.edge('agent1', 'agent2', label='Extracted Text')
    graph.edge('agent2', 'agent3', label='Corrected Text')
    graph.edge('agent3', 'agent4', label='Understood Info')
    graph.edge('agent3', 'agent5', label='Understood Info')
    graph.edge('agent5', 'agent6', label='Graph Data')
    
    return graph

# Define the agent functions
def agent_ocr(image):
    """Agent 1: Extract text from image using OCR"""
    st.write("### Agent 1: OCR Text Extraction")
    with st.spinner("Extracting text from image..."):
        response = flash_model.generate_content(
            [image, "Only extract the exact text content from the image. Don't include any explanation or additional information."],
            stream=False
        )
        extracted_text = response.text.strip()
        st.write("Extracted Text:")
        st.text_area("", extracted_text, height=200)
        return extracted_text

def agent_correction(extracted_text):
    """Agent 2: Correct the extracted text"""
    st.write("### Agent 2: Text Correction")
    with st.spinner("Correcting text..."):
        prompt = f"Correct only the spelling mistakes in the following text. Do not add, remove, or change anything else. Correct obvious mistakes to make the statements clear and meaningful if needed:\n\n{extracted_text}"
        response = flash_model.generate_content(prompt)
        corrected_text = response.text.strip()
        st.write("Corrected Text:")
        st.text_area("", corrected_text, height=200)
        return corrected_text

def agent_understanding(corrected_text):
    """Agent 3: Extract meaningful information from the text"""
    st.write("### Agent 3: Text Understanding")
    with st.spinner("Extracting meaningful information..."):
        prompt = f"Extract only the meaningful relationships or medically relevant information from the following text. Format as a structured list without asterisks. Do not add anything extra:\n\n{corrected_text}"
        response = flash_model.generate_content(prompt)
        understood_text = response.text.strip()
        st.write("Understood Medical Information:")
        st.text_area("", understood_text, height=200)
        return understood_text

def agent_fhir_format(understood_text):
    """Agent 4: Convert to FHIR format"""
    st.write("### Agent 4: FHIR Format Conversion")
    with st.spinner("Converting to FHIR format..."):
        prompt = f"""
        Convert the following medical report into FHIR (Fast Healthcare Interoperability Resources) JSON format.
        - Do NOT include any personal identifying information (e.g. name, date of birth, caste).
        - Only include medically relevant data such as diagnosis, medications, allergies, observations, and treatment plan.
        - Do not add any extra information. Keep it minimal and structured.

        Text:
        {understood_text}
        """
        response = flash_model.generate_content(prompt)
        fhir_json = response.text.strip()
        
        # Try to parse and display as formatted JSON
        try:
            parsed_json = json.loads(fhir_json)
            st.write("FHIR JSON:")
            st.json(parsed_json)
            return fhir_json
        except json.JSONDecodeError:
            st.write("FHIR Format (not valid JSON):")
            st.text_area("", fhir_json, height=200)
            return fhir_json

def agent_graph_db(understood_text):
    """Agent 5: Create Neo4j Graph Database entries"""
    st.write("### Agent 5: Graph Database Creation")
    with st.spinner("Creating graph database entries..."):
        # Prompt to Gemini
        prompt = f"""
        You are a Cypher query generator for Neo4j.

        1. First, generate a Cypher query to delete all existing nodes and relationships.
        2. Then, from the given medical text, extract medically relevant information (e.g., diagnosis, medications, observations, treatment plan) and generate Cypher queries to insert them into the graph.
        3. Do NOT include any personal info like name, birthdate, or caste. Use 'Patient1' as generic identifier.
        4. Output only raw Cypher queries without triple backticks or extra explanations.

        Text:
        {understood_text}
        """

        # Generate the response
        response = flash_model.generate_content(prompt)
        raw_cypher = response.text.strip()
        
        # Strip code blocks if present
        if raw_cypher.startswith("```"):
            raw_cypher = "\n".join(line for line in raw_cypher.splitlines() if not line.startswith("```"))

        st.write("Generated Cypher Queries:")
        st.text_area("", raw_cypher, height=200)
        
        # Execute the queries
        with st.spinner("Executing queries in Neo4j..."):
            try:
                driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
                with driver.session() as session:
                    for query in raw_cypher.split(";"):
                        query = query.strip()
                        if query:
                            session.run(query)
                driver.close()
                st.success("Graph database updated successfully!")
            except Exception as e:
                st.error(f"Error connecting to Neo4j database: {str(e)}")
        
        return raw_cypher

def agent_final_output():
    """Agent 6: Generate final output using LangChain"""
    st.write("### Agent 6: Final Response Generation")
    with st.spinner("Generating final medical response..."):
        try:
            # Initialize Neo4j Graph
            graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database="neo4j"
            )
            
            # Initialize Gemini LLM
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
            
            # Get schema information from the database
            schema = graph.schema
            
            # Query to get Patient1's health records
            patient_query = """
            MATCH (p:Patient {name: 'Patient1'})-[r]->(n)
            RETURN p, r, n
            """
            patient_data = graph.query(patient_query)
            
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

            Your response should be formatted with clear sections for Summary, Prescription, and Recommendations. Do not use asterisks in your response.
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
            st.write("Medical Analysis and Recommendations:")
            st.write(result['text'])
            return result['text']
        except Exception as e:
            st.error(f"Error generating final output: {str(e)}")
            return None

# Main app
def main():
    st.title("Medical Document Processing System")
    
    # Create tabs for Flow Chart and Processing
    tab1, tab2 = st.tabs(["System Flow Chart", "Document Processing"])
    
    with tab1:
        st.header("Medical Document Processing Flow")
        st.graphviz_chart(create_flow_chart())
        
        st.subheader("Process Description")
        st.write("""
        This system processes medical documents through six specialized AI agents:
        
        1. **OCR Agent**: Takes a medical document image and extracts the raw text content
        2. **Correction Agent**: Fixes spelling mistakes and typos in the extracted text
        3. **Understanding Agent**: Analyzes the corrected text to extract meaningful medical information
        4. **FHIR Format Agent**: Converts the medical information to standardized FHIR JSON format
        5. **Graph DB Agent**: Creates a knowledge graph in Neo4j database representing medical relationships
        6. **Response Agent**: Analyzes the graph data to generate a comprehensive medical report
        
        Each agent builds upon the output of previous agents to transform raw document images into actionable medical insights.
        """)
    
    with tab2:
        st.write("Upload a medical document image to process through our AI agent pipeline.")
        
        # Image upload
        uploaded_file = st.file_uploader("Choose a medical document image", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            
            # Process button
            if st.button("Process Document"):
                with st.container():
                    # Run the agent pipeline
                    extracted_text = agent_ocr(image)
                    corrected_text = agent_correction(extracted_text)
                    understood_text = agent_understanding(corrected_text)
                    fhir_json = agent_fhir_format(understood_text)
                    graph_queries = agent_graph_db(understood_text)
                    final_output = agent_final_output()
                    
                    # Display the final summary at the top for easy access
                    if final_output:
                        st.write("## Final Medical Analysis")
                        st.write(final_output)

if __name__ == "__main__":
    main()