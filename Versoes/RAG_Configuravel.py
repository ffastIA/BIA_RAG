import os  # Permite acessar arquivos, pastas e configurações do computador
import streamlit as st  # Framework para criar aplicações web interativas
from langchain_community.document_loaders import PyPDFLoader  # Carrega arquivos PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Divide textos grandes em pedaços menores
from langchain_openai import OpenAIEmbeddings  # Converte texto em números (vetores)
from langchain_community.vectorstores import FAISS  # Banco de dados para armazenar e buscar vetores
from langchain_openai import ChatOpenAI  # Modelo de linguagem da OpenAI (GPT)
from langchain.chains import RetrievalQA  # Sistema de perguntas e respostas
from langchain.prompts import ChatPromptTemplate  # Template para formatar perguntas ao modelo
import tempfile  # Cria arquivos temporários
from dotenv import load_dotenv  # Carrega senhas do arquivo .env (arquivo secreto)
import time  # Controla tempo e delays

# ========== NOVOS IMPORTS PARA O PIPELINE COM TRADUÇÃO ==========
# Estes imports são necessários para criar o pipeline customizado
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# =================================================================


# Tenta pegar a chave da API da OpenAI de duas formas diferentes
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')  # Do arquivo .env
# api_key = st.secrets["OPENAI_API_KEY"]  # Do Streamlit secrets (para deploy)

# Verifica se conseguiu pegar a chave da API
if not api_key:
    # Se não conseguiu, para tudo e mostra erro
    raise ValueError("A variável de ambiente 'OPENAI_API_KEY' não foi encontrada no seu arquivo .env.")

# Define a chave API como variável de ambiente
os.environ["OPENAI_API_KEY"] = api_key

#Define o modelo de GPT
model_name = 'gpt-3.5-turbo-0125'  # Qual modelo do ChatGPT usar

# 1. CONFIGURAÇÃO DA PÁGINA
# -------------------------
# Define título, ícone e layout da página
st.set_page_config(
    page_title="Assistente de IA: Biometria Inteligente ",  # Título que aparece na aba do navegador
    page_icon="🐟",  # Ícone que aparece na aba
    layout="wide"  # Usa toda a largura da tela
)

# 2. TÍTULO E DESCRIÇÃO
# ---------------------
# Mostra o título principal e explica o que a aplicação faz
st.title("🐟 Biometria Inteligente para Aquicultura")
st.markdown("""
Esta aplicação permite que você analise o projeto BIA usando Inteligência Artificial.
Faça perguntas sobre sua aplicação, benefícios e indicadores financeiros!
""")

# 3. CONFIGURAÇÃO DA BARRA LATERAL
# --------------------------------
# Cria uma barra lateral para configurações
with st.sidebar:
    st.header("⚙️ Configurações")

    # Controle de temperatura (criatividade do modelo)
    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="0 = Respostas mais precisas, 1 = Respostas mais criativas"
    )

    # Upload do arquivo PDF
    uploaded_file = st.file_uploader(
        "📄 Faça upload do PDF",
        type=['pdf'],
        help="Selecione um arquivo PDF para análise"
    )

# 4. PROCESSAMENTO DO PDF
# -----------------------
# Se um arquivo foi carregado, processa ele
if uploaded_file is not None:
    # Mostra informações sobre o arquivo
    st.success(f"✅ Arquivo '{uploaded_file.name}' carregado com sucesso!")

    # Cria um arquivo temporário para salvar o PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        # Escreve o conteúdo do upload no arquivo temporário
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Barra de progresso para mostrar o processamento
    with st.spinner("📖 Lendo e processando o PDF..."):
        # Carrega o PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Mostra quantas páginas foram carregadas
        st.info(f"📄 {len(documents)} páginas carregadas do PDF")

    # 6. DIVISÃO DO TEXTO EM CHUNKS
    # -----------------------------
    # Divide o texto em pedaços menores para melhor processamento
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Tamanho máximo de cada pedaço
        chunk_overlap=200,  # Sobreposição entre pedaços
        length_function=len,  # Função para medir o tamanho
        separators=["&","\n\n", "\n", " ", ""]  # Onde dividir o texto
    )

    # Aplica a divisão
    chunks = text_splitter.split_documents(documents)
    st.info(f"✂️ Documento dividido em {len(chunks)} pedaços")

    # 7. CRIAÇÃO DOS EMBEDDINGS E ARMAZENAMENTO
    # -----------------------------------------
    with st.spinner("🧮 Criando embeddings e armazenando no banco vetorial..."):
        # Cria o modelo de embeddings
        embeddings = OpenAIEmbeddings()

        # Cria o banco de dados vetorial com os chunks
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Cria um buscador que retorna os 4 pedaços mais relevantes
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    st.success("✅ Processamento concluído! Pronto para responder perguntas.")

    # 8. CONFIGURAÇÃO DO MODELO DE LINGUAGEM
    # --------------------------------------
    # Configura o ChatGPT com as configurações escolhidas
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=1000  # Limite de tokens na resposta
    )

    # Template para formatar as perguntas
    prompt = ChatPromptTemplate.from_template("""
    Use o contexto fornecido para responder à pergunta. 
    Se você não souber a resposta, diga que não sabe, não tente inventar.

    Contexto: {context}

    Pergunta: {question}

    Resposta detalhada:""")


    # ========== NOVA FUNÇÃO AUXILIAR ==========
    # Esta função formata os documentos encontrados em um texto único
    def format_docs(docs):
        """
        Pega uma lista de documentos e junta todo o conteúdo em um texto só.
        Cada documento é separado por duas quebras de linha.
        """
        return "\n\n".join(doc.page_content for doc in docs)


    # ==========================================

    # ========== NOVO PIPELINE COM TRADUÇÃO ==========
    # Substituímos o qa_chain original por um pipeline customizado
    # que faz tanto a busca/resposta quanto a tradução

    # PASSO 1: Chain que faz o RAG (busca + resposta)
    rag_chain = (
            {
                "context": retriever | format_docs,  # Busca documentos e formata
                "question": RunnablePassthrough()  # Passa a pergunta sem modificar
            }
            | prompt  # Aplica o template de prompt
            | llm  # Envia para o modelo GPT
            | StrOutputParser()  # Extrai apenas o texto da resposta
    )

    # PASSO 2: Template para tradução
    # Este prompt instrui o modelo a traduzir mantendo termos técnicos
    translation_prompt = ChatPromptTemplate.from_template("""
    Você é um tradutor especializado em textos técnicos.
    Traduza o seguinte texto para inglês, mantendo termos técnicos e formatação:

    Texto: {text}

    Tradução:""")

    # PASSO 3: Chain de tradução
    # Cria uma chain específica para tradução
    translation_chain = translation_prompt | llm | StrOutputParser()


    # PASSO 4: Função que executa todo o pipeline
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
        resposta_original = rag_chain.invoke(query)  # CORREÇÃO: passa só a string

        # Segundo: traduz a resposta
        resposta_traduzida = translation_chain.invoke({
            "text": resposta_original
        })

        # Terceiro: busca os documentos fonte para referência
        # CORREÇÃO: usa a string query diretamente
        source_docs = retriever.get_relevant_documents(query)

        # Retorna tudo em um dicionário
        return {
            "result": resposta_original,
            "translated": resposta_traduzida,
            "source_documents": source_docs
        }


    # ================================================

    # 10. INTERFACE DE CHAT
    # --------------------
    st.markdown("---")
    st.header("💬 Chat com o Documento")

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

    # Remove arquivo temporário
    os.unlink(tmp_file_path)

else:
    # Se nenhum arquivo foi carregado
    st.info("👈 Por favor, faça upload de um arquivo PDF na barra lateral para começar.")