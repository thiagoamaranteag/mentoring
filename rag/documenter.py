import tiktoken

# Inicializa o codificador de tokens para o modelo treinado
tokenizer = tiktoken.get_encoding("cl100k_base")

# Lendo o texto do arquivo .txt
with open('newsletter.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Dividindo o texto em "documents"
document_size = 10000  # Tamanho do "document" em caracteres
documents = [text[i:i + document_size] for i in range(0, len(text), document_size)]

# Processando cada "document"
for i, document in enumerate(documents):
    # Tokenizando o "document"
    tokens = tokenizer.encode(document)
    
    # Mostra os tokens gerados
    print(f"Document {i + 1}:")
    # print(f"Texto: {document}")
    # print(f"Tokens gerados: {tokens}")
    print(f"Número de tokens: {len(tokens)}")
    
    # Decodifica os tokens de volta para o texto original
    # decoded_text = tokenizer.decode(tokens)
    # print(f"Texto decodificado: {decoded_text}")
    
    # Converte cada token individualmente de volta para texto
    # tokens_as_text = [tokenizer.decode_single_token_bytes(token) for token in tokens]
    # print(f"Tokens em formato de texto: {tokens_as_text}")
    
    print("\n" + "-"*50 + "\n")