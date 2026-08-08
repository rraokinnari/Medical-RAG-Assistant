import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load environment variables
load_dotenv()


# Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# Test text
text = "Hypertension is a major risk factor for stroke."


# Generate embedding
vector = embeddings.embed_query(text)


print("Embedding generated successfully!")
print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])