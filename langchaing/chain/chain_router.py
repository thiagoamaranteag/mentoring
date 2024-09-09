from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()
import certifi

# Carregar o certificado SSL
cert_path = '../zscaler.pem'
with open(cert_path, 'rb') as cert_file:
    zscaler_cert = cert_file.read()
certifi_path = certifi.where()
with open(certifi_path, 'ab') as certifi_file:
    certifi_file.write(zscaler_cert)

# MultiPromptChain foi depreciado e evoluiu para utilizar LCEL (LangChain Expression Language)

class StringFormatterResponse(Runnable):
    def invoke(self, data, config=None):
        print(f"Resposta:\n{data}")

# Inicializar o modelo de linguagem
llm = ChatOpenAI(model="gpt-4o-mini")

# Definindo os prompts especializados
prompt_fisica = ChatPromptTemplate.from_template("Você é um professor de física muito inteligente. Aqui está a pergunta: {input}")
prompt_matematica = ChatPromptTemplate.from_template("Você é um professor de matemática muito inteligente. Aqui está a pergunta: {input}")
prompt_quimica = ChatPromptTemplate.from_template("Você é um professor de química muito inteligente. Aqui está a pergunta: {input}")

# Prompt para o roteador
router_template = """Dado a entrada do usuário, selecione a área mais apropriada:
Input: {input}
Responda apenas com o nome da área (fisica, matematica, ou quimica). Se nenhuma destas áreas for apropriada, responda com 'geral'.
Área:"""

router_prompt = ChatPromptTemplate.from_template(router_template)

# Definindo o roteador
router_chain = router_prompt | llm | StrOutputParser() | StringFormatterResponse()

# Definindo as cadeias para cada área
chain_map = {
    "fisica": prompt_fisica | llm | StrOutputParser(),
    "matematica": prompt_matematica | llm | StrOutputParser(),
    "quimica": prompt_quimica | llm | StrOutputParser(),
}

# Prompt padrão para casos gerais
default_prompt = ChatPromptTemplate.from_template("Aqui está uma pergunta geral: {input}")
default_chain = default_prompt | llm | StrOutputParser()

# Função para selecionar a cadeia apropriada
def route(input_dict):
    route_result = router_chain.invoke(input_dict)
    return chain_map.get(route_result, default_chain)

# sistema de composição do LCEL
# Criando a cadeia final, o primiero chain recebe a mensagem do usuario, o segundo seleciona o chain especifico, e o ultimo processo chain selecionado (LCEL)
chain = RunnablePassthrough() | RunnableLambda(route) | RunnablePassthrough()

# Usando a cadeia
resposta = chain.invoke("Qual estilo muiscal mais ouvido no centro oeste do Brasil?")
print(resposta)