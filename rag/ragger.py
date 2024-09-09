import warnings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Carregar o modelo de embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo leve para embeddings

# 1. Carregar o documento de texto
with open("newsletter.txt", "r") as file:
    document = file.read()

# 2. Dividir o texto em sentenças ou partes menores
sentences = document.split('.')

# 3. Gerar embeddings para cada sentença
sentence_embeddings = embedding_model.encode(sentences)

# 4. Criar um índice FAISS para armazenar os embeddings
d = sentence_embeddings.shape[1]  # Dimensão dos embeddings
index = faiss.IndexFlatL2(d)  # L2 é a métrica de distância Euclidiana
index.add(np.array(sentence_embeddings))  # Adicionar os embeddings ao índice

# Função para fazer perguntas ao sistema
def ask_question(question):
    # 5. Gerar embedding da pergunta
    question_embedding = embedding_model.encode([question])
    
    # 6. Buscar as sentenças mais similares no FAISS
    D, I = index.search(np.array(question_embedding), k=3)  # Retorna as top-3 sentenças mais similares
    relevant_sentences = [sentences[i] for i in I[0]]  # Extrair as sentenças relevantes
    
    # 7. Retornar as sentenças mais relevantes como resposta
    response = " ".join(relevant_sentences)
    
    return response

# Loop para perguntas
while True:
    question = input("# Digite sua pergunta (ou 'exit' para sair): ")
    
    if question.lower() == 'exit':
        print("Encerrando o sistema...")
        break
    
    answer = ask_question(question)
    print(f"####\nResposta: {answer}\n")
    print("\n####\n\n")
