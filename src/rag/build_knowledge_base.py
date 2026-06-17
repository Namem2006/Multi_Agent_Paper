import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


GOOGLE_API_KEY = "AIzaSyBrwzeGRALqrf2Hdl0s7cXnwr6QqqpOk-Q"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def build_vector_db():
  
    file_path = r"D:\Project\multi agent\work\guideline.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        guideline_content = f.read()
    #Chunking
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Main_Section"), ("##", "Entity_Detail")]
    )
    md_docs = markdown_splitter.split_text(guideline_content)
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, 
    chunk_overlap=50  
    )
    final_docs = text_splitter.split_documents(md_docs)


    #embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    persist_dir = "./chroma_db_acsa" 
    #vector hoa va luu
    vectorstore = Chroma.from_documents(
        documents=final_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
if __name__ == "__main__":
    build_vector_db()