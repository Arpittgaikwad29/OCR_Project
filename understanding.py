import os
from google.generativeai import GenerativeModel, configure

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyC_yR_Z1BNvkL1ixrejs2aqvGHSXK2CwvE"
configure(api_key=os.environ["GOOGLE_API_KEY"])

# Use Gemini model
model = GenerativeModel("gemini-2.0-flash")

# Input text
input_text = """Dr. L. H. Hiranandani Hospital
Kiran Agoural
23/olite 201
F. 69
Dysphagia (Cough for than gody)
Cab
COC TSH T3TY
Nival Meker 1130/
SUPT Creat CAP
Liquid Toll dich
On fedh
11 Am The fron
Ap Tricani mos
2- Hygody
C. mylara 19-cod"""

# Prompt for extracting only valuable relational information
prompt = f"Extract only the meaningful relationships or medically relevant information from the following text. Do not add anything extra:\n\n{input_text}"

# Generate response
response = model.generate_content(prompt)

# Print extracted info
print(response.text.strip())
