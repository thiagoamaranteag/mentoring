import tiktoken

# Inicializa o codificador de tokens para o modelo treinado
tokenizer = tiktoken.get_encoding("cl100k_base")

# Texto de exemplo maior
text = """
StackSpot AI é uma plataforma incrível para desenvolvedores. Ela oferece uma variedade de ferramentas e serviços que facilitam o desenvolvimento de software. 
Com StackSpot, você pode criar, testar e implantar suas aplicações de maneira eficiente e rápida. A plataforma é projetada para ser intuitiva e fácil de usar, 
permitindo que desenvolvedores de todos os níveis de habilidade possam aproveitar ao máximo seus recursos. Além disso, StackSpot AI está constantemente evoluindo 
e adicionando novas funcionalidades para atender às necessidades dos desenvolvedores modernos.
"""

# Dividindo o texto em chunks
chunk_size = 100  # Tamanho do chunk em caracteres
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Processando cada chunk
for i, chunk in enumerate(chunks):
    # Tokenizando o chunk
    tokens = tokenizer.encode(chunk)
    
    # Mostra os tokens gerados
    print(f"# Chunk {i + 1}:")
    print(f"# Texto: {chunk}")
    print(f"# Tokens gerados: {tokens}")
    print(f"# Número de tokens: {len(tokens)}")
    
    # Decodifica os tokens de volta para o texto original
    decoded_text = tokenizer.decode(tokens)
    print(f"# Texto decodificado: {decoded_text}")
    
    # Converte cada token individualmente de volta para texto
    tokens_as_text = [tokenizer.decode_single_token_bytes(token) for token in tokens]
    
    print(f"# Tokens em formato de texto: {tokens_as_text}")
    print("\n" + "-"*50 + "\n")