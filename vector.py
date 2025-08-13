from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import chardet

#df = pd.read_csv("realistic_restaurant_reviews.csv")
#df = pd.read_csv("atividades_complementares_integrado2.csv")

with open("TCC.csv", "rb") as f:
    result = chardet.detect(f.read())
    print(result)  # {'encoding': 'Windows-1252', 'confidence': 0.73}

df = pd.read_csv("TCC.csv", encoding=result['encoding'], sep=';')

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            #page_content=row["Title"] + " " + row["Review"],
            #metadata={"rating": row["Rating"], "date": row["Date"]},

            #page_content=row["Texto Introdutório"] + " " + row["Documento Comprobatório"] + " " + row["Carga Horária Máxima"]+" " + row["Texto"],
            page_content=row["Tema"]+" " + row["info"] ,
            metadata={},

            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)