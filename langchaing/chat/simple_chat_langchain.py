from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Inicializa o modelo
llm = OpenAI()

print("Chat simples com LangChain. Digite 'sair' para encerrar.")

while True:
    user_input = input("Você: ")
    if user_input.lower() == 'sair':
        print("Encerrando o chat. Até logo!")
        break
    
    response = llm.invoke(user_input)
    print("OpenAI:", response)