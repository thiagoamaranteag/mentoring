import tiktoken

def texto_para_tokens(texto):
    # Converte texto em tokens e faz prin
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(texto)
    
    print(f"Texto: {texto}")
    print(f"Tokens: {tokens}")
    print(f"Total: {len(tokens)} tokens\n")
    
    return tokens


def comparar_frases(frase1, frase2):
    # Compara tokens de duas frases e mostra sobreposição
    encoding = tiktoken.encoding_for_model("gpt-4")
    
    tokens1 = set(encoding.encode(frase1))
    tokens2 = set(encoding.encode(frase2))
    
    sobreposicao = tokens1.intersection(tokens2)
    
    print(f"Frase 1: {frase1}")
    print(f"Tokens: {len(tokens1)}")
    print()
    print(f"Frase 2: {frase2}")
    print(f"Tokens: {len(tokens2)}")
    print()
    print(f"Tokens em comum: {len(sobreposicao)}")
    print(f"Tokens sobrepostos: {sobreposicao}")


if __name__ == "__main__":
    texto_para_tokens("StackSpot AI é uma plataforma incrível para desenvolvedores.")
    
    comparar_frases(
        "ZUP Innovation é uma empresa brasileira.",
        "Brasil é um país com grande potencial tecnologico."
    )
