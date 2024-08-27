from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (como a chave da API do OpenAI)
load_dotenv()

# Função auxiliar para ler o arquivo
def read_file(filepath: str) -> str:
    try:
        with open(filepath, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Erro ao ler o arquivo: {e}"

# Inicializar o modelo de linguagem
llm = ChatOpenAI(temperature=0, model="gpt-4")

# Função principal para executar o agente
def run_agent(filepath):
    content = read_file(filepath)  # Lê o conteúdo do arquivo uma vez
    if "Erro ao ler o arquivo" in content:
        print(content)
        return

    while True:
        query = input("\nDigite sua pergunta sobre o arquivo (ou 'sair' para encerrar):\n")
        if query.lower() == 'sair':
            break

        # Combinar o conteúdo do arquivo com a pergunta do usuário
        messages = [
            SystemMessage(content=f"O arquivo contém o seguinte texto:\n{content}"),
            HumanMessage(content=query)
        ]

        # Enviar as mensagens ao modelo e obter a resposta
        response = llm.invoke(messages)
        print(f"\nResposta: {response.content}")

# Exemplo de uso
run_agent("news.txt")
