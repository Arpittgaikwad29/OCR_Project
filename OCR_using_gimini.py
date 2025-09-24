import os
from google.generativeai import GenerativeModel, configure
from PIL import Image

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyC_yR_Z1BNvkL1ixrejs2aqvGHSXK2CwvE"
configure(api_key=os.environ["GOOGLE_API_KEY"])

# Load image
image = Image.open("p6.jpg")

# Use Gemini Flash model
model = GenerativeModel("gemini-2.0-flash")

# Stream response
response = model.generate_content(
    [image, "Only extract the exact text content from the image. Don't include any explanation or additional information."],
    stream=True
)

# Print only extracted text
print("".join(part.text for part in response).strip())

