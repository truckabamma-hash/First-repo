import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Read your document
with open("data/my_document.txt", "r") as f:
    document = f.read()

# Ask question with the document as context
query = "What is the main topic of this document?"
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": f"Answer questions based on this document: {document}"},
        {"role": "user", "content": query}
    ]
)

print(response.choices[0].message.content)
