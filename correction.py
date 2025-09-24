import os
from google.generativeai import GenerativeModel, configure

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyC_yR_Z1BNvkL1ixrejs2aqvGHSXK2CwvE"
configure(api_key=os.environ["GOOGLE_API_KEY"])

# Use Gemini model
model = GenerativeModel("gemini-2.0-flash")

# Input text
input_text = """Dr L H Hiranandani Hospital
Kiran Agoural.
23/olite 201
F. 69
Dysphgrat hy (Coy for than gody)
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

# Prompt for correction
prompt = f"Correct only the spelling mistakes in the following text. Do not add, remove,Correct obvious mistakes to make the statements clear and meaningful if needed. or change anything else:\n\n{input_text}"

# Generate response
response = model.generate_content(prompt)

# Print corrected text
print(response.text.strip())
