import os, json
import urllib.request
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

def chat_with_openai(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    
    request = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(request) as response:
            response_data = json.loads(response.read().decode())
            return response_data['choices'][0]['message']['content']
    except urllib.error.HTTPError as e:
        return f"Erro na requisição: {e.code} {e.reason}"

def main():
    print("Bem-vindo ao chat simples com OpenAI. Digite 'sair' para encerrar.")
    
    while True:
        user_input = input("Você: ")
        if user_input.lower() == 'sair':
            print("Encerrando o chat. Até logo!")
            break
        
        response = chat_with_openai(user_input)
        print("OpenAI:", response)

if __name__ == "__main__":
    main()