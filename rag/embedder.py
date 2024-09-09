import warnings
import torch
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Carrega o tokenizador pré-treinado BERT (Bidirectional Encoder Representations from Transformers)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Carrega o modelo pré-treinado BERT
model = AutoModel.from_pretrained("bert-base-uncased")

# Texto de exemplo que será tokenizado e processado
text = """
StackSpot AI é uma plataforma incrível para desenvolvedores. Ela oferece uma variedade de ferramentas e serviços que facilitam o desenvolvimento de software. 
Com StackSpot, você pode criar, testar e implantar suas aplicações de maneira eficiente e rápida. A plataforma é projetada para ser intuitiva e fácil de usar, 
permitindo que desenvolvedores de todos os níveis de habilidade possam aproveitar ao máximo seus recursos. Além disso, StackSpot AI está constantemente evoluindo 
e adicionando novas funcionalidades para atender às necessidades dos desenvolvedores modernos.
"""

# Tokeniza o texto, convertendo-o em IDs de tokens e retornando como tensores do PyTorch
tokens = tokenizer(text, return_tensors="pt")

# Gera embeddings a partir dos tokens usando o modelo BERT
# torch.no_grad() desativa o cálculo de gradiente (Ajuste de Pesos) para economizar memória e melhorar a performance
with torch.no_grad():
    outputs = model(**tokens)  # Passa os tokens pelo modelo para obter as saídas
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Calcula a média dos embeddings ao longo da dimensão 1

# Imprime os resultados
print("# Texto original:", text)  # Imprime o texto original
print("# Tokens:", tokens['input_ids'][0].tolist())  # Imprime os IDs dos tokens gerados
print("# Tokens decodificados:", tokenizer.decode(tokens['input_ids'][0]))  # Decodifica os tokens de volta para texto
print("# Forma dos embeddings:", embeddings.shape)  # Imprime a forma dos embeddings gerados
print("# Valores do embedding:", embeddings)  # Imprime os valores dos embeddings


# Mesmo para uma palavra pequena, o embedding gerado pelo modelo BERT é grande porque o modelo utiliza uma representação vetorial de alta dimensão para capturar o máximo de informações semânticas e contextuais possíveis. No caso do BERT-base, cada token é representado por um vetor de 768 dimensões. Isso permite que o modelo capture nuances e relações complexas entre palavras, mesmo em contextos variados.