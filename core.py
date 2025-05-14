from langchain import hub
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser 
from pydantic import BaseModel, Field
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

def run_llm(prompt):
    class formatResponse(BaseModel):
        answer: str = Field(description="Jawaban atas pertanyaan user berdasarkan dokumen")
        bab: str = Field(description="BAB dari mana informasi diambil")
        subbab: str = Field(description="SUB BAB dari mana informasi diambil")

        def to_dict(self) -> Dict[str,any]:
            return{"answer":self.answer,"bab":self.bab,"subbab":self.subbab}
        
    parser = PydanticOutputParser(pydantic_object=formatResponse)
    
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
     "Contoh format jawaban:\n"
     "\n{format_instructions}")
]).partial(format_instructions=parser.get_format_instructions())


    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name="pdf-tesis-rag",embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0.7)

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_prompt)
    qa = create_retrieval_chain(retriever=vectorstore.as_retriever(),combine_docs_chain=stuff_documents_chain)
    result = qa.invoke(input={"input": prompt})
    parsed = parser.parse(result["answer"])
    return parsed

if __name__ == "__main__":
    tanya = input("Tanyakan Sesuatu : ")
    res = run_llm(tanya)
    print(res)