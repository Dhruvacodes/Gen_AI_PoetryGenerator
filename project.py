import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Use local embedding model instead of OpenAI
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Ensure OpenAI API Key is set (optional, if using OpenAI for text generation)
os.environ["OPENAI_API_KEY"] = "AIzaSyBquVnh-NM0g7oJAhdXK6YZQ7JspH6wOV8"
if "OPENAI_API_KEY" not in os.environ:
    st.warning("Warning: OpenAI API key is missing. Local embeddings will be used.")

# Initialize ChromaDB client
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="poetry_references")

# Load poetry dataset (Ensure poetry_data exists in the directory)
data_dir = "poetry_data"
if not os.path.exists(data_dir):
    st.error(f"Error: The directory '{data_dir}' does not exist. Please add it and restart the app.")
else:
    reader = SimpleDirectoryReader(data_dir)
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    retriever = index.as_retriever()

# Setup OpenAI model for text generation
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define Gemini LLM function
def generate_poem(prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# Define LangChain prompt template
prompt_template = PromptTemplate(
    input_variables=["theme", "context"],
    template="""
    Write a beautiful poem about {theme}. Here are some poetic inspirations:
    {context}
    """
)
def run_chain(theme, context):
    prompt = prompt_template.format(theme=theme, context=context)
    return generate_poem(prompt)

# Streamlit UI
st.title("AI Poetry Generator")
theme = st.text_input("Enter a theme for your poem:")
if st.button("Generate Poem"):
    if theme:
        # Retrieve relevant poetic inspirations
        docs = retriever.retrieve(theme)
        context = "\n".join([doc.text for doc in docs])
        
        # Generate poem using RAG
        poem = run_chain(theme, context)
        
        st.subheader("Generated Poem:")
        st.write(poem)
    else:
        st.warning("Please enter a theme!")
