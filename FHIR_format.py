import os
from google.generativeai import GenerativeModel, configure

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyC_yR_Z1BNvkL1ixrejs2aqvGHSXK2CwvE"
configure(api_key=os.environ["GOOGLE_API_KEY"])

# Use Gemini model
model = GenerativeModel("gemini-2.0-flash")

# Input medical text
input_text = """**Patient:** Kiran Agoural
*   **Sex:** Female
*   **Age:** 69
*   **Diagnosis/Symptoms:** Dysphagia (cough)
*   **Medical History/Relevant Information:** Cab
*   **Medications/Labs:** COC, TSH, T3, T4
*   **SUPT Creat CAP**"""

# Prompt for FHIR conversion with PHI hidden
prompt = f"""
Convert the following medical report into FHIR (Fast Healthcare Interoperability Resources) JSON format.
- Do NOT include any personal identifying information (e.g. name, date of birth, caste).
- Only include medically relevant data such as diagnosis, medications, allergies, observations, and treatment plan.
- Do not add any extra information. Keep it minimal and structured.

Text:
{input_text}
"""

# Generate response
response = model.generate_content(prompt)

# Print FHIR-compliant output
print(response.text.strip())
