import requests
import json

def responder_pergunta(pergunta):
    """Envia uma pergunta ao modelo rodando no Ollama e retorna a resposta."""
    
    # 1. Defina a URL da API do Ollama
    # Certifique-se de que o Ollama está rodando na sua máquina!
    url_ollama = "http://localhost:11434/api/generate"
    
    # 2. Defina o payload (dados a serem enviados)
    # 'mistral' é o nome do modelo que você baixou
    payload = {
        "model": "mistral",
        "prompt": pergunta,
        "stream": False # Não queremos a resposta em pedaços (streaming), mas completa de uma vez
    }
    
    # 3. Envie a requisição POST para o Ollama
    try:
        response = requests.post(
            url_ollama,
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'}
        )
        
        # 4. Verifique se a requisição foi bem-sucedida
        if response.status_code == 200:
            # 5. Extraia e retorne o texto da resposta
            data = response.json()
            return data.get("response", "Erro: O modelo não retornou uma resposta.")
        else:
            return f"Erro na requisição: Código de status {response.status_code}. Verifique se o Ollama está ativo."
            
    except requests.exceptions.ConnectionError:
        return "Erro de Conexão: Certifique-se de que o **Ollama está rodando** e o modelo 'mistral' foi baixado."
    except Exception as e:
        return f"Ocorreu um erro: {e}"

# --- EXECUÇÃO ---

# Exemplo de pergunta simples:
minha_pergunta = "Qual a capital do Brasil?"

print(f"**Pergunta:** {minha_pergunta}")

resposta_ia = responder_pergunta(minha_pergunta)

print("---")
print(f"**Resposta da IA:** {resposta_ia.strip()}")