import tiktoken

# Inicializa o codificador de tokens para o modelo treinado
tokenizer = tiktoken.get_encoding("cl100k_base")

# Texto de exemplo que será tokenizado
text = "StackSpot AI é uma plataforma incrível para desenvolvedores."

# Tokenizando o texto
tokens = tokenizer.encode(text)

# Mostra os tokens gerados
print(f"# Tokens gerados: {tokens}")
print(f"# Número de tokens: {len(tokens)}")

# Converte cada token individualmente de volta para texto
tokens_as_text = [tokenizer.decode_single_token_bytes(token) for token in tokens]
print(f"# Tokens em formato de texto: {tokens_as_text}")


# Decodifica os tokens de volta para o texto original
decoded_text = tokenizer.decode(tokens)
print(f"# Texto decodificado: {decoded_text}")