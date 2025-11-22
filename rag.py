import os
import sys

# Garante que o Python possa encontrar os novos módulos
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    
    # Importações do pacote Community para Document Loaders
    from langchain_community.document_loaders import DirectoryLoader
    
    # Importações dos pacotes específicos (ollama e chroma)
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_chroma import Chroma
    

except ImportError as e:
    print(f"ERRO DE IMPORTAÇÃO: {e}")
    print("Certifique-se de instalar: pip install langchain-text-splitters langchain-ollama langchain-chroma")
    sys.exit(1)


# --- CONFIGURAÇÕES ---
MODELO = "mistral"
REGRAS_DIR = "regras/"
PERSIST_DIR = "./chroma_db"

# -------------------------------------------------------
# 1) CRIAR ÍNDICE
# -------------------------------------------------------
def criar_indice():
    """Carrega documentos, cria chunks e salva o banco Chroma."""
    print("📄 Carregando documentos...")

    # A LangChain é robusta para encontrar todos os .txt
    loader = DirectoryLoader(REGRAS_DIR, glob="**/*.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} chunks criados.")

    embeddings = OllamaEmbeddings(model=MODELO)

    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    print("📦 Chroma criado e salvo!")

# -------------------------------------------------------
# 2) CRIAR RAG COM LCEL (PIPELINE)
# -------------------------------------------------------
def criar_rag_chain():
    """Cria a cadeia RAG usando LCEL (Expressões LangChain)."""
    
    embeddings = OllamaEmbeddings(model=MODELO)
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings  # Use embedding_function para consistência
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=MODELO, temperature=0.0) # Temp=0.0 é melhor para análise de código/regras

    # Define o Prompt Template
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=(
            "Você é um Analisador de Código rigoroso. Use APENAS o contexto de REGRAS fornecido.\n\n"
            "REGRAS (Contexto):\n{context}\n\n"
            "CÓDIGO A SER ANALISADO (Pergunta):\n{input}\n\n"
            "DECISÃO: Permita ou Proíba o uso do operador '|' no código e justifique com base nas regras."
        )
    )

    # -----------------------------
    # 🔥 LCEL PIPELINE (ESTRUTURA CORRETA)
    # -----------------------------
    # 1. RunnableParallel: Define as duas chaves de entrada ("context" e "input")
    #    - "context": Pega a entrada original, passa pelo retriever, e formata os docs em uma string.
    #    - "input": Pega a entrada original e a passa adiante (RunnablePassthrough).
    # 2. prompt: Formata a string de prompt com as chaves context e input.
    # 3. llm: Envia o prompt formatado para o modelo (ChatOllama).
    rag_chain = (
        {
            # A entrada do pipeline é o código (string)
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "input": RunnablePassthrough() 
        }
        | prompt
        | llm
    )

    return rag_chain


# -------------------------------------------------------
# 3) CONSULTA (CORREÇÃO AQUI)
# -------------------------------------------------------
def consultar(rag_chain, codigo_analise):
    """
    Executa a cadeia RAG.
    
    A correção: O rag_chain (que usa RunnablePassthrough) espera a entrada 
    como uma string simples, não como um dicionário.
    """
    # A entrada aqui deve ser o valor esperado pelo RunnablePassthrough, 
    # que neste caso é a string 'codigo_analise'.
    resposta = rag_chain.invoke(codigo_analise) 
    
    # O objeto de resposta de um ChatOllama é um objeto Message, 
    # o conteúdo está em .content
    return resposta.content 


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(PERSIST_DIR):
        criar_indice()

    rag = criar_rag_chain()

    # Teste 1: Lógico (Deve Proibir)
    codigo_proibido = "if (variavel | 0) { console.log('Erro'); }"
    # Teste 2: Bitwise (Deve Permitir)
    codigo_permitido = "flags = MASK_A | MASK_B;"

    print("\n=== Consulta Código Lógico ===")
    print(consultar(rag, codigo_proibido))
    
    print("\n=== Consulta Código Bitwise ===")
    print(consultar(rag, codigo_permitido))