from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class WebsiteChat:
    def __init__(self, url):
        # Carrega o conteúdo do site
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Divide o texto em chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        print(f"####\n{texts}\n####")
        
        # Cria embeddings
        embeddings = OpenAIEmbeddings()
        
        # Cria o banco de dados de vetores Chroma
        self.db = Chroma.from_documents(texts, embeddings)
        
        # Inicializa o modelo de chat
        self.chat = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
        
        # Cria a cadeia de conversação
        self.qa = ConversationalRetrievalChain.from_llm(
            self.chat, 
            self.db.as_retriever(search_kwargs={"k": 4}), 
            return_source_documents=True
        )
        
        self.chat_history = []

    def ask(self, question):
        result = self.qa.invoke({"question": question, "chat_history": self.chat_history})
        self.chat_history.append((question, result['answer']))
        return result['answer']

# Solicita o URL do site ao usuário
url = input("Digite o URL do site que você deseja carregar: ")
web_chat = WebsiteChat(url)

print(f"# Chat inicializado com o conteúdo de {url}. Você pode começar a fazer perguntas sobre o conteúdo do site.")

while True:
    question = input("\n# Faça uma pergunta (ou digite 'exit' para encerrar): ")
    if question.lower() == 'exit':
        break
    answer = web_chat.ask(question)
    print(f"# Resposta: {answer} \n\n###########")