from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_pinecone import PineconeVectorStore

def setup():
    load_dotenv()
    loader = PyPDFLoader("knowledge/153510596.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = loader.load()
    documents = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(
    documents, embeddings, index_name="pdf-tesis-rag"
    )
    print("****Loading to vectorstore done ***")




if __name__ == "__main__":
    print("Run From Main")
    setup()