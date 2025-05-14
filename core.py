from langchain import hub
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def run_llm(prompt):
    retrieval_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Anda adalah asisten AI yang menjawab pertanyaan berdasarkan isi skripsi akademik. "
     "Tugas Anda adalah:\n"
     "1. Menggunakan informasi yang relevan dari dokumen.\n"
     "2. Menentukan informasi tersebut berasal dari BAB mana.\n"
     "3. Memberikan jawaban ringkas dan jelas.\n\n"
     "Jika tidak ditemukan informasi relevan, katakan bahwa jawabannya tidak tersedia."),

    ("human", 
     "Pertanyaan: {input}\n\n"
     "Referensi dokumen:\n{context}\n\n"
     "Jawab pertanyaan di atas dan sebutkan informasi berasal dari BAB apa dan sub bab apa. "
     "Contoh format jawaban:\n"
     "\"Jawaban Anda di sini... (Sumber: BAB III - Sub bab 3.1)\"")
])


    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name="pdf-tesis-rag",embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0.7)

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_prompt)
    qa = create_retrieval_chain(retriever=vectorstore.as_retriever(),combine_docs_chain=stuff_documents_chain)
    result = qa.invoke(input={"input": prompt})
    return result

if __name__ == "__main__":
    tanya = input("Tanyakan Sesuatu : ")
    res = run_llm(tanya)
    print(res["answer"])