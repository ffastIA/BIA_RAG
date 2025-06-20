import os  # Permite acessar arquivos, pastas e configurações do computador
import streamlit as st  # Framework para criar aplicações web interativas
from langchain_community.document_loaders import PyPDFLoader  # Carrega arquivos PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Divide textos grandes em pedaços menores
from langchain_openai import OpenAIEmbeddings  # Converte texto em números (vetores)
from langchain_community.vectorstores import FAISS  # Banco de dados para armazenar e buscar vetores
from langchain_openai import ChatOpenAI  # Modelo de linguagem da OpenAI (GPT)
from langchain.chains import RetrievalQA  # Sistema de perguntas e respostas
from langchain.prompts import ChatPromptTemplate  # Template para formatar perguntas ao modelo
from dotenv import load_dotenv  # Carrega senhas do arquivo .env (arquivo secreto)
import tiktoken  # Conta quantas "palavras" (tokens) tem um texto
from langchain_core.runnables import RunnablePassthrough, RunnableParallel # processamento paralelo na chain
from langchain_core.output_parsers import StrOutputParser  #organizador de output
from langchain.schema.runnable import RunnableLambda  #encapsulador de função
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader


# ============================ INICIA O STREAMLIT =====================================
# Este código precisa acontecer antes de qualquer chamada de outra função do streamlit
# Define título da aba, ícone e layout geral da página
st.set_page_config(
    page_title="BIA",  # Título que aparece na aba do navegador
    page_icon="🐟",  # Ícone que aparece na aba
    layout="wide"  # Usa toda a largura da tela
)
# =====================================================================================


# ============================ CHAVES DO SISTEMA ===============================
# chave para deploy
deploy = True #False == versão local / True == versão deploy
#chve para criação de vector store
cria_vector = False #False == só carrega a vector store / True == cria a vector store
# ===============================================================================

# Tenta pegar a chave da API da OpenAI de duas formas diferentes
load_dotenv()
if not deploy:
    api_key = os.getenv("OPENAI_API_KEY")  # Do arquivo .env
else:
    api_key = st.secrets["OPENAI_API_KEY"]  # Do Streamlit secrets (para deploy)

# Verifica se conseguiu pegar a chave da API
if not api_key:
    # Se não conseguiu, para tudo e mostra erro
    raise ValueError("A variável de ambiente 'OPENAI_API_KEY' não foi encontrada no seu arquivo .env.")


# -----------------------------------------------------------------------------
# 1. DEFINIÇÕES GERAIS
# -----------------------------------------------------------------------------
#"""  Definições gerais (variáveis de sistema, modelo, vectorstore, arquivos, ...) """

# Define a chave API como variável de ambiente
os.environ["OPENAI_API_KEY"] = api_key

#Define o modelo de GPT
modelo = 'gpt-3.5-turbo-0125'  # Qual modelo do ChatGPT usar

# Cria uma instância do modelo de chat da OpenAI
chat_instance = ChatOpenAI(model=modelo)

#Define o modelo de embedding
embeddings_model = OpenAIEmbeddings()  # Modelo para converter texto em números

#Define o local onde o vector store persistirá
diretorio_vectorstore_faiss = 'vectorstore_faiss'  # Onde salvar o banco vetorial

# #Define o local do arquivo a ser processado
# caminho_arquivo = r'BIA_RAG.pdf'  # Caminho do PDF para analisar

#Define o diretório onde os arquivos para gerar o vector store estão localizados
caminho_arquivo = 'docs'  # Caminho dos arquivos para analisar

#Define a quantidade máxima de documentos retornados pela função retriever()
qtd_retriever = 4


# -----------------------------------------------------------------------------
# 2. PROMPTS
# -----------------------------------------------------------------------------
#"""  Prompts """

# Template para pergunta original
prompt_inicial = ChatPromptTemplate.from_template("""
    Você é um copywiter e deve ajudar a esclarecer dúvidas sobre o projeto Biometria Inteligente para Aquicultura (BIA)
    Use o contexto fornecido para responder às perguntas. Você pode buscar dados complementares na internet, desde que
    forneça as fontes consultadas. Se você não souber a resposta, diga que não sabe, não tente inventar.
    
    Contexto: {context}
    
    Pergunta: {question}
    
    Resposta detalhada:
    """)

# Template para traduzir para o ingLês a reposta
translation_prompt = ChatPromptTemplate.from_template("""
    Você é um tradutor especializado em textos técnicos.
    Traduza o seguinte texto para inglês, mantendo termos técnicos e formatação:
    
    Texto: {text}
    
    Tradução:
    """)


# -----------------------------------------------------------------------------
# 3. FUNÇÕES
# -----------------------------------------------------------------------------
#"""  Funções """

def cria_vector_store_faiss(chunk: list[Document], diretorio_vectorestore_faiss:str) -> FAISS:
    """
    Esta função cria um banco de dados especial para busca rápida de informações

    O que ela faz:
    1. Pega os pedaços de texto do documento
    2. Converte cada pedaço em números (vetores) que representam o significado
    3. Cria um banco de dados FAISS para busca super rápida
    4. Salva tudo no computador para usar depois

    Parâmetros:
        chunks: Lista de pedaços de documento já divididos

    Retorna:
        vectorstore: O banco de dados criado
    """
    # Cria o banco vetorial usando os chunks e o modelo de embeddings
    # FAISS é uma biblioteca que faz busca muito rápida em vetores
    vector_store = FAISS.from_documents(chunk, embeddings_model)

    # Salva o banco no disco rígido para não precisar recriar toda vez
    vector_store.save_local(diretorio_vectorestore_faiss)

    return vector_store


def carrega_vector_store_faiss(diretorio_vectorestore_faiss, embedding_model):
    """
    Esta função carrega um banco de dados vetorial já criado anteriormente

    O que ela faz:
    1. Procura o banco de dados salvo no disco
    2. Carrega ele na memória
    3. Retorna pronto para uso

    Parâmetros:
        diretorio_vectorestore_faiss: Pasta onde está salvo o banco
        embeddings_model: Modelo usado para criar os embeddings

    Retorna:
        vectorstore: O banco de dados carregado
    """
    # Carrega o banco vetorial do disco rígido
    vector_store = FAISS.load_local(
        diretorio_vectorestore_faiss,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Esta função conta quantos "tokens" (pedacinhos) tem em um texto

    Por que isso é importante?
    - A IA tem limite de quantos tokens pode processar de uma vez
    - Precisamos saber se o texto cabe no limite
    - 1 token ≈ 0.75 palavras em português

    Parâmetros:
        string: O texto para contar
        encoding_name: Tipo de codificação (padrão da OpenAI)

    Retorna:
        int: Número de tokens no texto
    """
    # Pega o codificador específico da OpenAI
    encoding = tiktoken.get_encoding(encoding_name)

    # Codifica o texto em tokens e conta quantos são
    num_tokens = len(encoding.encode(string))

    return num_tokens


def cria_chunks(caminho: str, chunk_size: int, chunk_overlap: int) -> list:
    print(f"Carregando documentos do PDF: {caminho_arquivo}")  # Informa o usuário sobre o carregamento do PDF.
    # try:  # Inicia um bloco try-except para lidar com possíveis erros durante o carregamento do arquivo.
    #     loader = PyPDFLoader(arquivo)  # Instancia um PyPDFLoader, passando o caminho do arquivo PDF.
    #     documentos = loader.load()  # Carrega o conteúdo do PDF. Cada página do PDF se torna um 'Document' separado na lista 'documentos'.
    #     print(
    #         f"PDF carregado. Total de documentos (páginas): {len(documentos)}")  # Confirma o carregamento e exibe o número de páginas/documentos.
    # except FileNotFoundError:  # Captura o erro específico se o arquivo PDF não for encontrado.
    #     print(
    #         f"Erro: O arquivo PDF não foi encontrado em '{arquivo}'. Por favor, verifique o caminho.")  # Informa o erro ao usuário.
    #     exit()  # Interrompe a execução do script se o arquivo não for encontrado, pois é uma dependência crítica.

    # Instanciação do DirectoryLoader
    loader = DirectoryLoader(
        path= caminho,  # OBRIGATÓRIO: Caminho do diretório
        glob="*.pdf",  # OPCIONAL: Padrão para filtrar arquivos (padrão: *)
        loader_cls = PyPDFLoader,  # OPCIONAL: Loader específico para os arquivos encontrados
        #           Se não especificado, ele tenta inferir,
        #           mas é mais seguro especificar.
        recursive=False  # OPCIONAL: Buscar em subdiretórios (padrão: False)
    )

    # Executa o carregamento
    documentos = loader.load()


    # --- Depuração e Inspeção de Documentos (Opcional) ---
    # print("\n--- Conteúdo da primeira página para verificação ---")
    # print(
    #     documentos[0].page_content[:500] + "...")  # Imprime os primeiros 500 caracteres da primeira página para depuração.
    # print("\n--- Metadados da primeira página ---")
    # print(documentos[0].metadata)  # Imprime os metadados associados à primeira página.
    # print("-" * 50 + "\n")

    # Configuração do Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Tamanho máximo do chunk em tokens (usando a função de contagem do módulo 'tokens').
        chunk_overlap=chunk_overlap,  # Sobreposição em tokens entre chunks.
        length_function= num_tokens_from_string,  # Nossa função personalizada de contagem de tokens.
        separators= ['&', '\n\n','.', ' '],
        add_start_index=True  # Adiciona o índice de início de cada chunk no texto original como metadado.
    )

    # Divide o documento completo em chunks
    chunk = text_splitter.split_documents(documentos)
    print(f"Texto original dividido em {len(chunk)} chunks.\n")

    return chunk


def format_docs(docs: list[Document]):
    """
    Pega uma lista de documentos e junta o conteúdo em um texto só.
    Cada documento é separado por duas quebras de linha.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def retriever(pergunta: str, n: int):
    """
    O que ela faz:
    1. Recebe sua pergunta
    2. Busca "n" informações relevantes no documento
    3. Retorna lista da classe "Document"

    Parâmetros:
        pergunta: procura a ser feita na vector store
        n: quantidade máxima de documentos a serem retornados na lista

    Retorna:
        documentos_retornados : lista com documentos encontrados da classe Document
    """

    resultado = vectorstore.as_retriever(search_type='mmr',search_kwargs={"k": n})  # Como buscar no documento

    documentos_retornados = resultado.get_relevant_documents(pergunta) #passa a pergunta para o retriever

    return documentos_retornados



# -----------------------------------------------------------------------------
# 4. VECTOR STORE
# -----------------------------------------------------------------------------

if cria_vector: #Chave do sistema
    chunks = cria_chunks(caminho_arquivo,500,50)
    vectorstore = cria_vector_store_faiss(chunks,diretorio_vectorstore_faiss)
else:
    vectorstore = carrega_vector_store_faiss(diretorio_vectorstore_faiss,embeddings_model)


# -----------------------------------------------------------------------------
# 5. CONFIGURAÇÃO DA PÁGINA PELO STREAMLIT
# -----------------------------------------------------------------------------
# # Define título da aba, ícone e layout geral da página
# st.set_page_config(
#     page_title="BIA",  # Título que aparece na aba do navegador
#     page_icon="🐟",  # Ícone que aparece na aba
#     layout="wide"  # Usa toda a largura da tela
# )


# -----------------------------------------------------------------------------
# 6. TÍTULO E DESCRIÇÃO DA PÁGINA PELO STREAMLIT
# -----------------------------------------------------------------------------
# Mostra o título principal e explica o que a aplicação faz
st.title("🐟 Biometria Inteligente para Aquicultura")
st.markdown("""
Esta aplicação permite que você analise o projeto BIA usando Inteligência Artificial.
Faça perguntas sobre sua aplicação e indicadores de investimento e retorno financeiro!
""")

# =============================== PIPELINE =================================
# Cria a sequência de chains
# ==========================================================================

# PASSO 1: Chain que faz o RAG (busca + resposta)

# Lambda com a função personalizada encapsulada em busca_vectorstore
busca_vectorstore = RunnableLambda(
    lambda pergunta: retriever(pergunta, qtd_retriever)
)

rag_chain = (
        {
            "context": busca_vectorstore | format_docs,  # Busca documentos e formata
            "question": RunnablePassthrough()  # Passa a pergunta sem modificar
        }
        | prompt_inicial  # Aplica o template de prompt
        | chat_instance  # Envia para o modelo GPT
        | StrOutputParser()  # Extrai apenas o texto da resposta
)

# PASSO 2: Chain de tradução
# Cria uma chain específica para tradução
translation_chain = translation_prompt | chat_instance | StrOutputParser()


# PASSO 4: Função que executa o pipeline
def qa_chain_complete(query):
    """
    Função que executa o pipeline completo:
    1. Busca documentos relevantes
    2. Gera resposta em português
    3. Traduz a resposta para inglês
    4. Retorna ambas as versões

    CORREÇÃO: Agora passamos apenas a string da query, não um dicionário
    """
    # Primeiro: gera a resposta usando apenas a string da pergunta
    resposta_original = rag_chain.invoke(query)

    # Segundo: traduz a resposta
    resposta_traduzida = translation_chain.invoke({
        "text": resposta_original
    })

    # Terceiro: busca os documentos fonte para referência
    source_docs = retriever(query, qtd_retriever)

    # Retorna tudo em um dicionário
    return {
        "result": resposta_original,
        "translated": resposta_traduzida,
        "source_documents": source_docs
    }


# ================================================
# 10. INTERFACE DE CHAT
# ================================================

st.markdown("---")
st.header("💬 Converse com o projeto BIA!")

# Inicializa o histórico de mensagens se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra o histórico de mensagens
for msg in st.session_state.messages:
    if msg["role"] == "user":
        # Mensagem do usuário
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        # ========== MODIFICAÇÃO NA EXIBIÇÃO ==========
        # Agora mostramos a resposta em duas abas: português e inglês
        with st.chat_message("assistant", avatar="🤖"):
            # Cria duas abas para as versões da resposta
            tab1, tab2 = st.tabs(["🇧🇷 Português", "🇺🇸 English"])

            # Aba 1: Resposta original em português
            with tab1:
                st.markdown(msg["content"])

            # Aba 2: Resposta traduzida em inglês
            with tab2:
                # Verifica se existe tradução (para mensagens antigas)
                if "translated" in msg:
                    st.markdown(msg["translated"])
                else:
                    st.info("Tradução não disponível para mensagens anteriores")

            # Mostra as fontes se existirem
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Fontes"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.markdown(f"**Fonte {i}:**")
                        st.markdown(source.page_content[:200] + "...")
                        st.markdown(f"*Página: {source.metadata.get('page', 'N/A')}*")
                        st.markdown("---")
        # =============================================

# Campo de entrada para nova pergunta
user_input = st.chat_input(
    "Digite sua pergunta sobre o documento...",
    key="user_input_field"
)

# Processa nova pergunta
if user_input:
    # Adiciona pergunta do usuário ao histórico
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Mostra a pergunta do usuário
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # ========== PROCESSAMENTO MODIFICADO ==========
    # Agora processa tanto a resposta quanto a tradução
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Buscando informações e traduzindo..."):
            # CORREÇÃO: Passa apenas a string user_input
            result = qa_chain_complete(user_input)

            # Extrai as respostas
            answer = result["result"]
            translated = result["translated"]
            sources = result.get("source_documents", [])

        # Mostra as respostas em abas
        tab1, tab2 = st.tabs(["🇧🇷 Português", "🇺🇸 English"])

        with tab1:
            st.markdown(answer)

        with tab2:
            st.markdown(translated)

        # Mostra as fontes
        if sources:
            with st.expander("📚 Fontes"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Fonte {i}:**")
                    st.markdown(source.page_content[:200] + "...")
                    st.markdown(f"*Página: {source.metadata.get('page', 'N/A')}*")
                    st.markdown("---")

        # Adiciona resposta ao histórico com ambas as versões
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "translated": translated,  # NOVO: guarda a tradução
            "sources": sources
        })
    # =============================================


