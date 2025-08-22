# UI for asking questions on the knowledge base
import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pages.app_admin import get_vector_store, get_text_chunks
from langchain.chains.combine_documents import create_stuff_documents_chain
import boto3
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Apply Tebra-inspired CSS
def apply_tebra_css():
    st.markdown("""
    <style>
    /* Import Google Fonts similar to Tebra */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background-color: #FAFBFC;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #FF8D6E 0%, #FF7A5C 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(255, 141, 110, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.95;
    }
    
    /* Streamlit element overrides */
    .stTextInput > div > div > input {
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
        background-color: white;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FF8D6E;
        box-shadow: 0 0 0 3px rgba(255, 141, 110, 0.1);
        outline: none;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF8D6E 0%, #FF7A5C 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(255, 141, 110, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 141, 110, 0.4);
        background: linear-gradient(135deg, #FF7A5C 0%, #FF6B4A 100%);
    }
    
    /* Response Box Styling */
    .response-box {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        position: relative;
    }
    
    .response-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #FF8D6E, #FF7A5C);
        border-radius: 16px 16px 0 0;
    }
    
    /* Tech Stack Footer */
    .tech-footer {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 3rem;
        color: #6B7280;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .tech-footer h3 {
        color: #374151;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #FF8D6E;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    /* Card-like sections */
    .stExpander {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        margin-bottom: 1rem;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F9FAFB;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 12px;
        color: #065F46;
    }
    
    /* Info messages */
    .stInfo {
        background-color: #EFF6FF;
        border: 1px solid #93C5FD;
        border-radius: 12px;
        color: #1E40AF;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        border-radius: 12px;
        color: #92400E;
    }
    
    /* Spinner customization */
    .stSpinner {
        color: #FF8D6E;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)


genai.configure(api_key=os.getenv("GENAI_API_KEY"))
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

nvidia_api_key = st.secrets["NVIDIA_API_KEY"]

def get_prompt_template():
    return PromptTemplate()

def get_chat_chain():
    prompt_template="""
    Answer the questions based on local konwledge base honestly

    Context:\n {context} \n
    Questions: \n {questions} \n

    Answers:
"""
    model=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3)
    # This is too slow
    #model = ChatNVIDIA(
    #    model="deepseek-ai/deepseek-r1",
    #    api_key=nvidia_api_key,
    #    temperature=0.7,
    #    top_p=0.8,
    #    max_tokens=4096
    #)
    #
    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "questions"], output_variables=["answers"])
    chain = create_stuff_documents_chain(llm=model, prompt=prompt, document_variable_name="context")
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_chat_chain()

    response = chain.invoke({"context": docs, "questions": user_question})

    print(response)
    
    # Create a styled response box
    st.markdown(f"""
    <div class="response-box">
        <h4 style="color: #374151; margin-bottom: 1rem; font-weight: 600;">🤖 AI Assistant Response</h4>
        <p style="color: #4B5563; line-height: 1.6; margin: 0;">{response}</p>
    </div>
    """, unsafe_allow_html=True)


def download_s3_bucket(bucket_name, download_dir):
    s3 = boto3.client('s3')
    
    # Ensure the download directory exists
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    # Pagination in case the bucket has many objects
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get('Contents', []):
            key = obj['Key']
            local_file_path = os.path.join(download_dir, key)
            
            # Create local directories if they don't exist
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))
                
            print(f"Downloading {key} to {local_file_path}")
            s3.download_file(bucket_name, key, local_file_path)

def download_faiss_from_s3():
    s3 = boto3.client(
        "s3",
        region_name="us-west-2",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )
    bucket_name = st.secrets["BUCKET_NAME"]
    print(bucket_name)
    # Download the FAISS index file from S3
    s3.download_file(bucket_name, "index.faiss", "index.faiss")
    s3.download_file(bucket_name, "index.pkl", "index.pkl")
    print(f"Downloaded FAISS index from s3://${bucket_name} to local directory")

def main():
    # Apply Tebra-inspired CSS
    apply_tebra_css()
    
    # Create custom header
    st.markdown("""
    <div class="main-header">
        <h1>AI Knowledge Assistant</h1>
        <p>Chat with your personal knowledge base powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

    # fix the empty vector store issue
    get_vector_store(get_text_chunks("Loading some documents to build your knowledge base"))

    # Input section with better styling
    st.markdown("### 💬 Ask me anything about Tebra")
    user_question = st.text_input(
        "Your question:",
        placeholder="How do I get started with Tebra?",
        label_visibility="collapsed"
    )
    
    if user_question:
        with st.spinner("🔍 Searching knowledge base..."):
            user_input(user_question)
    
    

if __name__ == "__main__":
    main()