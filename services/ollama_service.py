from langchain_ollama import ChatOllama
from langchain_community.document_loaders import CSVLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

llm = ChatOllama(model="llama3.2", temperature=0.7)


# Load CSV file
loader = CSVLoader(file_path="test_data/Used_Bikes_With_Header.csv", encoding="utf-8")
data = loader.load()

embeddings = OllamaEmbeddings(model="nomic-embed-text")


persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(
    documents=data,
    embedding=embeddings,
    persist_directory="docs/chroma"
)

retriever = vectorstore.as_retriever(search_type="similarity")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

query = "What is the best bike you can recommned based on age, kms driven and high mileage?"
response = qa_chain.invoke({"query": query})
print(response["result"])
# print(response["source_documents"])