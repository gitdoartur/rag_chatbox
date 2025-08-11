from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """

Você é um expert em informações sobre o curso de Ciência de Computação do IFNMG. 
Você tem acesso a um banco de dados de informações sobre o TCC e pode responder perguntas sobre este tópico.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


#result=chain.invoke({"reviews": [], "question": "Oq eu você sabe sobre o curso de ciencia da computação no ifnmg"})
#print(result)


while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)