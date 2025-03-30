import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings, os
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
import sqlite3

# Load environment variables from .env file
load_dotenv()

# Set the directory where your mobile phone specs vector database is stored
data_directory = os.path.join(os.path.dirname(__file__), "data")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Alternatively, you can use st.secrets["huggingface_api_token"]

# Load the vector store from disk (built using phone_specs.txt data)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

# Initialize the Hugging Face Hub LLM (you can change the repo_id if desired)
hf_hub_llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    # repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 1, "max_new_tokens": 1024},
)

# Update the prompt template for a mobile recommender system
prompt_template = """
You are a highly knowledgeable mobile assistant specialized in recommending smartphones based on their technical specifications.
Your domain is mobile phones only. Do NOT mention clothing or disclaim about clothing under any circumstances.

1. Answer only using details from our mobile specifications database.
2. Provide concise, accurate, and relevant recommendations.
3. If the question is ambiguous, ask for clarification or provide the most balanced recommendation.
4. Do not include extraneous commentary or disclaimers about other domains.

Mobile Specs:
{context}

Question: {question}

Answer:
"""


custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(top_k=3),  # Retrieve the top 3 matching documents
    chain_type_kwargs={"prompt": custom_prompt}
)

def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Streamlit app UI styling
st.markdown(
    """
    <style>
        .appview-container .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
    <h1 style='text-align: center; color: white; 
               background-color: #007BFF; 
               padding: 20px; 
               border-radius: 8px; 
               margin-top: 20px; 
               margin-bottom: 20px;'>
        Discover Your Perfect Mobile Phone 📱
    </h1>
""", unsafe_allow_html=True)


side_bar_message = """
Hi there! 👋 I'm here to help you find the right smartphone based on your preferences and technical needs. You can ask me about:
1. **Battery Life** 🔋
2. **Display Quality** 📱
3. **Camera Performance** 📷
4. **Price Range** 💰
5. **Overall Specifications** ⚙️

Just type your question and I'll recommend a mobile phone from our database!
"""

with st.sidebar:
    st.title('🤖 MobileBot: Your Smartphone Recommender')
    st.markdown(side_bar_message)

initial_message = """
Hi! I'm your MobileBot 🤖 
Here are some sample questions you can ask:
- What is the best smartphone for battery life?
- Which phone has the best display quality under 30,000 INR?
- Can you recommend a mobile with a great camera?
- Which phone offers the best value for its price?
"""

# Store chat messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

# Get user input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate response if last message is from the user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Finding the best mobile recommendation for you..."):
            response = get_response(prompt)
            placeholder = st.empty()
            full_response = response  # Use the response directly
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


