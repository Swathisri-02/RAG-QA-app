import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

# API Keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Directory for uploaded PDFs
UPLOAD_FOLDER = "./pdf"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Sidebar for navigation
st.sidebar.title("Options")
option = st.sidebar.radio("Choose an option", ("Upload PDFs", "Chatbot"))

if option == "Upload PDFs":
    st.title("Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(UPLOAD_FOLDER, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files successfully uploaded.")

elif option == "Chatbot":
    st.title("Chatbot")

    # Initialize the models and vector store
    def initialize_models_and_vectors():
        if "vectors" not in st.session_state:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.docs = []
            
            # Load each PDF file in the directory
            for filename in os.listdir(UPLOAD_FOLDER):
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(UPLOAD_FOLDER, filename))
                    st.session_state.docs.extend(loader.load())
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    
    initialize_models_and_vectors()

    # Set up the language model
    llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    # User input for the question
    question = st.text_input("What do you want to ask from the documents?")

    if question:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({'input': question})
        st.write(response['answer'])

        # with st.expander("Document Search"):
        #     for i, doc in enumerate(response['context']):
        #         st.write(doc.page_content)
        #         st.write("-----------------------------------------------------------")
