from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

import os
from dotenv import load_dotenv
from collections import OrderedDict

# Load environment variables from .env file
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the text file
loader = TextLoader("phone_specs.txt")  # Using the provided phone specs file
documents = loader.load()

# Split the text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Reduced chunk size for better handling of individual phone entries
    chunk_overlap=200,
    separators=["\n\n"]  # Split on double newlines between phone entries
)
texts = text_splitter.split_documents(documents)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma vector store
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="phone_data"
)

# Add documents to the vector store
vector_store.add_documents(documents=texts)

# Validate the setup
try:
    # Test query to validate data retrieval
    test_query = "Phones with 8GB RAM under 30000"
    results = vector_store.similarity_search(test_query, k=5)

    # Deduplicate results
    unique_results = OrderedDict()
    for doc in results:
        content = doc.page_content
        if content not in unique_results:
            unique_results[content] = doc

    # Convert unique results to a list and limit to top 3
    final_results = list(unique_results.values())[:3]
    
    print("Top matching results:")
    for i, result in enumerate(final_results, 1):
        print(f"\nResult {i}:")
        print(result.page_content[:500] + "...")  # Show preview of the content

except Exception as e:
    print(f"Error during test query: {e}")
