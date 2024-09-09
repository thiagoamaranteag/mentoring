from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (como a chave da API do OpenAI)
load_dotenv()

# Inicializar o modelo de linguagem
llm = ChatOpenAI(temperature=0.7)

# Criar o prompt do sistema
system_prompt = "Você é um assistente amigável e prestativo. Mantenha suas respostas concisas e relevantes."

# Criar o template do prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Inicializar a memória
memory = ConversationBufferMemory(return_messages=True)

# Criar a cadeia de conversação
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# Função para executar o chat
def chat():
    print("Assistente: Olá! Como posso ajudar você hoje? (Digite 'sair' para encerrar)")
    
    while True:
        user_input = input("Você: ")
        
        if user_input.lower() == 'sair':
            print("Assistente: Foi um prazer conversar com você. Até logo!")
            break
        
        response = conversation.predict(input=user_input)
        print(f"Assistente: {response}")

# Iniciar o chat
if __name__ == "__main__":
    chat()