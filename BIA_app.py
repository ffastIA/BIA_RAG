# =============================================================================
# SISTEMA DE IA PARA ANÁLISE DE DOCUMENTOS PDF
# =============================================================================
# Este programa cria uma interface web onde você pode fazer perguntas sobre
# um documento PDF e receber respostas inteligentes de uma IA.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. IMPORTAÇÕES - As "Caixas de Ferramentas" que Vamos Usar
# -----------------------------------------------------------------------------

# BIBLIOTECAS BÁSICAS DO PYTHON
import os  # Permite acessar arquivos, pastas e configurações do computador
import textwrap  # Ajuda a formatar texto, quebrando linhas longas em pedaços menores

# BIBLIOTECA PARA CARREGAR SENHAS E CONFIGURAÇÕES SECRETAS
from dotenv import load_dotenv  # Carrega senhas do arquivo .env (arquivo secreto)

# BIBLIOTECAS DA LANGCHAIN - Especializada em IA e Processamento de Documentos
from langchain_core.prompts import ChatPromptTemplate  # Cria "receitas" de como a IA deve responder
from langchain.chains.combine_documents import create_stuff_documents_chain  # Junta documentos com perguntas
from langchain_community.document_loaders.pdf import PyPDFLoader  # Lê arquivos PDF
from langchain_openai import ChatOpenAI  # Conversa com a IA da OpenAI (ChatGPT)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Divide textos grandes em pedaços
from langchain.chains.retrieval_qa.base import RetrievalQA  # Sistema de perguntas e respostas
from langchain_openai import OpenAIEmbeddings  # Converte texto em números para busca
from langchain_community.vectorstores import FAISS  # Banco de dados rápido para busca

# BIBLIOTECAS AUXILIARES
from typing import List  # Para definir que tipo de dados esperamos (lista)
from langchain_core.documents import Document  # Tipo especial de documento do LangChain
import tiktoken  # Conta quantas "palavras" (tokens) tem um texto

# A ESTRELA DO SHOW - STREAMLIT
import streamlit as st  # Cria páginas web bonitas usando apenas Python!

# -----------------------------------------------------------------------------
# 2. CONFIGURAÇÃO DA PÁGINA - Definindo Como a Página Vai Aparecer
# -----------------------------------------------------------------------------

# Esta função configura aspectos básicos da página web
st.set_page_config(
    page_title="Agente de IA: Biometria Inteligente para Aquicultura (BIA)",  # Nome na aba do navegador
    page_icon="🤖",  # Emoji que aparece na aba do navegador
    layout="wide",  # Usa toda a largura da tela (ao invés de ficar estreito)
    initial_sidebar_state="collapsed"  # Esconde a barra lateral para ter mais espaço
)

# -----------------------------------------------------------------------------
# 3. CSS PERSONALIZADO - O "Estilista" que Deixa Tudo Bonito
# -----------------------------------------------------------------------------

# O CSS é como um "maquiador" que define cores, tamanhos, posições, etc.
# Usamos st.markdown com HTML para aplicar estilos personalizados
st.markdown("""
<style>
    /* ===================================================================== */
    /* CONFIGURAÇÕES GERAIS DA PÁGINA */
    /* ===================================================================== */

    /* Container principal - onde fica todo o conteúdo da página */
    .main .block-container {
        padding-top: 0.5rem !important;        /* Espaço pequeno no topo (0.5rem = ~8px) */
        padding-bottom: 0.5rem !important;     /* Espaço pequeno embaixo */
        padding-left: 1rem !important;         /* Espaço nas laterais esquerda */
        padding-right: 1rem !important;        /* Espaço nas laterais direita */
        max-width: none !important;            /* Remove limite de largura (usa tela toda) */
        /* Removemos min-height para permitir layout mais natural */
    }

    /* Remove o cabeçalho padrão do Streamlit que ocupava espaço desnecessário */
    .stApp > header {
        height: 0rem !important;               /* Altura zero (invisível) */
        visibility: hidden !important;         /* Torna completamente invisível */
    }

    /* Compacta todos os elementos da página para economizar espaço */
    .element-container {
        margin-bottom: 0.3rem !important;      /* Espaço pequeno entre elementos */
    }

    /* ===================================================================== */
    /* ESTILIZAÇÃO DO TÍTULO PRINCIPAL */
    /* ===================================================================== */

    /* Container do título principal */
    .titulo-principal {
        margin-top: 0 !important;              /* Sem espaço no topo */
        padding-top: 0.5rem !important;        /* Padding interno pequeno no topo */
        padding-bottom: 0.8rem !important;     /* Padding interno pequeno embaixo */
        text-align: center;                    /* Centraliza o texto */
        border-bottom: 2px solid #e0e0e0;      /* Linha decorativa cinza embaixo */
        margin-bottom: 1rem !important;        /* Espaço após o título */
    }

    /* Título H1 (o título grande) dentro do container */
    .titulo-principal h1 {
        margin: 0 !important;                  /* Remove margens padrão do navegador */
        padding: 0 !important;                 /* Remove padding padrão do navegador */
        font-size: 2.2rem !important;          /* Tamanho da fonte (2.2rem = ~35px) */
        line-height: 1.2 !important;           /* Altura da linha (espaçamento entre linhas) */
    }

    /* Parágrafo de descrição abaixo do título */
    .titulo-principal p {
        margin: 0.3rem 0 0 0 !important;       /* Margem pequena apenas no topo */
        padding: 0 !important;                 /* Sem padding */
        font-size: 1rem !important;            /* Tamanho da fonte normal */
        color: #666;                           /* Cor cinza para texto secundário */
    }

    /* ===================================================================== */
    /* ESTILIZAÇÃO DOS BOTÕES */
    /* ===================================================================== */

    /* Botões principais (tipo "primary") */
    .stButton > button[kind="primary"] {
        /* Gradiente colorido de fundo (transição do vermelho para azul-verde) */
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4) !important;
        border: none !important;                                          /* Remove borda */
        font-weight: bold !important;                                     /* Texto em negrito */
        transition: all 0.3s ease !important;                            /* Animação suave (0.3s) */
        height: 2.5rem !important;                                       /* Altura fixa */
        font-size: 1rem !important;                                      /* Tamanho da fonte */
        margin: 0.2rem 0 !important;                                     /* Margem pequena */
    }

    /* Efeito quando o mouse passa sobre o botão (hover) */
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;                          /* Move 1px para cima */
        box-shadow: 0 3px 8px rgba(0,0,0,0.15) !important;              /* Adiciona sombra */
    }

    /* ===================================================================== */
    /* CAMPO DE PERGUNTA PRÓXIMO AO BOTÃO - NOVA CONFIGURAÇÃO */
    /* ===================================================================== */

    /* Container para o campo de pergunta - agora mais próximo do botão */
    .campo-pergunta-container {
        margin-top: 1.5rem !important;         /* Espaço pequeno após o botão */
        margin-bottom: 1rem !important;        /* Espaço pequeno antes do próximo elemento */
        padding: 1rem 0 !important;            /* Padding vertical reduzido */
    }

    /* Wrapper interno para controlar a largura máxima */
    .campo-pergunta-wrapper {
        width: 100% !important;                /* Largura total disponível */
        max-width: 800px !important;           /* Largura máxima de 800px */
        margin: 0 auto !important;             /* Centraliza horizontalmente */
    }

    /* Título do campo de pergunta */
    .titulo-pergunta {
        text-align: center !important;         /* Centraliza o título */
        margin-bottom: 1rem !important;        /* Espaço pequeno abaixo do título */
    }

    /* ===================================================================== */
    /* CAMPO DE ENTRADA DE CHAT */
    /* ===================================================================== */

    /* Container do campo de chat input */
    .stChatInput {
        margin: 0.5rem 0 !important;           /* Espaçamento vertical reduzido */
    }

    /* Div interna do campo de chat */
    .stChatInput > div {
        max-width: none !important;            /* Remove limite de largura */
        margin: 0 !important;                  /* Remove margem */
    }

    /* Caixa de dica estilizada */
    .dica-container {
        text-align: center !important;         /* Centraliza o conteúdo */
        margin-top: 0.8rem !important;         /* Espaço pequeno acima */
    }

    .dica-box {
        background-color: #e8f4fd !important;  /* Fundo azul claro */
        padding: 0.8rem !important;            /* Padding interno reduzido */
        border-radius: 8px !important;         /* Cantos arredondados */
        border-left: 4px solid #1f77b4 !important; /* Borda azul à esquerda */
        display: inline-block !important;      /* Para centralizar melhor */
        max-width: 400px !important;           /* Largura máxima da dica */
    }

    /* ===================================================================== */
    /* CONTAINERS DE RESPOSTA */
    /* ===================================================================== */

    /* Container estilizado para exibir as respostas da IA */
    .resposta-container {
        background-color: #f8f9fa;             /* Fundo cinza muito claro */
        padding: 1rem !important;              /* Espaçamento interno */
        border-radius: 8px;                    /* Cantos arredondados */
        border-left: 4px solid #4ecdc4;        /* Borda colorida à esquerda */
        margin: 0.5rem 0 !important;           /* Margem pequena */
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); /* Sombra sutil */
    }

    /* ===================================================================== */
    /* ELEMENTOS DIVERSOS */
    /* ===================================================================== */

    /* Cabeçalhos dos expandir/contrair (Expander) */
    .streamlit-expanderHeader {
        padding: 0.3rem 0 !important;          /* Padding reduzido */
    }

    /* Caixas de informação e alertas */
    .stAlert {
        padding: 0.5rem !important;            /* Padding reduzido */
        margin: 0.3rem 0 !important;           /* Margem reduzida */
    }

    /* Linhas separadoras horizontais */
    hr {
        margin: 0.5rem 0 !important;           /* Margem pequena */
        border-color: #e0e0e0 !important;      /* Cor cinza clara */
    }

    /* Animação de carregamento (spinner) */
    .stSpinner {
        margin: 0.3rem 0 !important;           /* Margem pequena */
    }

    /* Elementos de texto Markdown */
    .stMarkdown {
        margin-bottom: 0.3rem !important;      /* Margem pequena embaixo */
    }

    /* Linhas de colunas */
    .row-widget {
        margin: 0 !important;                  /* Remove margem */
    }

    /* Área de texto */
    .stTextArea textarea {
        min-height: 60px !important;           /* Altura mínima */
    }

    /* ===================================================================== */
    /* BARRA DE ROLAGEM PERSONALIZADA */
    /* ===================================================================== */

    /* Largura da barra de rolagem */
    ::-webkit-scrollbar {
        width: 6px;                            /* Barra fina (6px) */
    }

    /* Fundo da trilha da barra de rolagem */
    ::-webkit-scrollbar-track {
        background: #f1f1f1;                   /* Cor cinza clara */
    }

    /* A barra de rolagem em si */
    ::-webkit-scrollbar-thumb {
        background: #888;                      /* Cor cinza */
        border-radius: 3px;                    /* Cantos arredondados */
    }

    /* Cor da barra quando o mouse passa sobre ela */
    ::-webkit-scrollbar-thumb:hover {
        background: #555;                      /* Cinza mais escuro */
    }
</style>
""", unsafe_allow_html=True)


# unsafe_allow_html=True permite usar HTML e CSS personalizados

# -----------------------------------------------------------------------------
# 4. DEFINIÇÃO DAS FUNÇÕES - "Receitas" que o Programa Vai Usar
# -----------------------------------------------------------------------------

def cria_vector_store_faiss(chunks: List[Document]):
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
    vectorstore = FAISS.from_documents(chunks, embeddings_model)

    # Salva o banco no disco rígido para não precisar recriar toda vez
    vectorstore.save_local(diretorio_vectorestore_faiss)

    return vectorstore


def carrega_vector_store_faiss(diretorio_vectorestore_faiss, embeddings_model):
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
    vectorstore = FAISS.load_local(diretorio_vectorestore_faiss, embeddings_model)
    return vectorstore


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


def limpar_area_resposta():
    """
    Esta função limpa a "memória" da conversa anterior

    O que ela faz:
    - Remove todas as informações da conversa anterior
    - É como apagar a lousa antes de escrever algo novo
    - Prepara o sistema para uma nova pergunta
    """
    # Lista de todas as coisas que queremos apagar da memória
    keys_to_clear = [
        "ultima_pergunta",  # A pergunta que foi feita
        "ultima_resposta",  # A resposta que foi dada
        "ultima_resposta_completa",  # A resposta completa com metadados
        "resposta_exibida"  # Se já mostrou uma resposta
    ]

    # Para cada item na lista, remove da memória se existir
    for key in keys_to_clear:
        if key in st.session_state:  # Se existe na memória
            del st.session_state[key]  # Apaga da memória


def processar_pergunta(pergunta, vectorstore, modelo):
    """
    Esta é a função principal que conversa com a IA!

    O que ela faz:
    1. Recebe sua pergunta
    2. Busca informações relevantes no documento
    3. Combina pergunta + informações encontradas
    4. Manda tudo para a IA (ChatGPT)
    5. Recebe e formata a resposta
    6. Retorna a resposta pronta

    Parâmetros:
        pergunta: A pergunta feita pelo usuário
        vectorstore: Banco de dados com o documento
        modelo: Qual modelo de IA usar (ex: gpt-3.5-turbo)

    Retorna:
        tuple: (resposta_formatada, resposta_completa)
    """
    # Cria uma instância do modelo de chat da OpenAI
    chat_instance = ChatOpenAI(model=modelo)

    # Cria uma "corrente" (chain) que vai:
    # 1. Receber a pergunta
    # 2. Buscar informações relevantes no documento (retrieval)
    # 3. Combinar tudo e mandar para a IA
    # 4. Retornar a resposta
    chat_chain = RetrievalQA.from_chain_type(
        llm=chat_instance,  # O modelo de IA a usar
        chain_type='stuff',  # Tipo: "encher" o prompt com informações
        retriever=vectorstore.as_retriever(search_type='mmr'),  # Como buscar no documento
        return_source_documents=True  # Retorna também as fontes usadas
    )

    # Executa toda a cadeia com a pergunta do usuário
    # Isso faz toda a mágica acontecer!
    resposta_do_chat = chat_chain.invoke({'query': pergunta})

    # Extrai apenas o texto da resposta (sem metadados)
    resposta_llm = resposta_do_chat.get('result', 'Nenhuma resposta disponível.')

    # Formata o texto para ficar mais legível
    # textwrap.fill quebra linhas longas em linhas de 150 caracteres
    resposta_formatada = textwrap.fill(resposta_llm, width=150)

    # Retorna tanto a resposta formatada quanto a completa (com metadados)
    return resposta_formatada, resposta_do_chat


# -----------------------------------------------------------------------------
# 5. CONFIGURAÇÕES E VARIÁVEIS - Definindo os Parâmetros do Sistema
# -----------------------------------------------------------------------------

# Carrega variáveis de ambiente do arquivo .env
# O arquivo .env contém senhas e configurações secretas
load_dotenv()

# Tenta pegar a chave da API da OpenAI de duas formas diferentes
openai_key = os.getenv('OPENAI_API_KEY')  # Do arquivo .env
openai_api_key = st.secrets["OPENAI_API_KEY"]  # Do Streamlit secrets (para deploy)

# Verifica se conseguiu pegar a chave da API
if not openai_key:
    # Se não conseguiu, para tudo e mostra erro
    raise ValueError("A variável de ambiente 'OPENAI_API_KEY' não foi encontrada no seu arquivo .env.")

# Configurações dos modelos de IA
modelo = 'gpt-3.5-turbo-0125'  # Qual modelo do ChatGPT usar
embeddings_model = OpenAIEmbeddings()  # Modelo para converter texto em números

# Configurações de arquivos e diretórios
diretorio_vectorestore_faiss = 'vectorestore_faiss'  # Onde salvar o banco vetorial
caminho_arquivo = r'BIA_RAG.pdf'  # Caminho do PDF para analisar

# -----------------------------------------------------------------------------
# 6. CARREGAMENTO E PROCESSAMENTO DO DOCUMENTO
# -----------------------------------------------------------------------------

# Tenta carregar o arquivo PDF
try:
    # Cria um carregador de PDF usando LangChain
    loader = PyPDFLoader(caminho_arquivo)

    # Carrega todo o conteúdo do PDF
    # Cada página vira um "Document" separado na lista
    documentos = loader.load()

except FileNotFoundError:
    # Se não conseguir achar o arquivo, mostra erro e para
    st.error(f"Erro: O arquivo PDF não foi encontrado em '{caminho_arquivo}'. Por favor, verifique o caminho.")
    st.stop()  # Para a execução do programa

# Configura como dividir o texto em pedaços menores
# Isso é necessário porque a IA tem limite de texto que pode processar
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Cada pedaço tem no máximo 500 tokens
    chunk_overlap=50,  # Pedaços se sobrepõem em 50 tokens (para não perder contexto)
    length_function=num_tokens_from_string,  # Usa nossa função para contar tokens
    separators=['&', '\n\n', '.', ' '],  # Onde pode cortar o texto (em ordem de preferência)
    add_start_index=True  # Adiciona índice de onde começou cada pedaço
)

# Divide todos os documentos em pedaços menores
chunks = text_splitter.split_documents(documentos)

# -----------------------------------------------------------------------------
# 7. CONFIGURAÇÃO DA IA
# -----------------------------------------------------------------------------

# Cria uma instância do modelo de chat
chat = ChatOpenAI(
    model=modelo,  # Qual modelo usar
    temperature=0  # 0 = respostas mais precisas, 1 = mais criativas
)

# Cria um template de prompt (como a IA deve se comportar)
qa_prompt = ChatPromptTemplate.from_messages([
    # Mensagem do sistema (instruções para a IA)
    ("system", (
        "Você é um assistente especialista em análise financeira e de investimento para aquicultura."
        "Use o seguinte contexto para responder à pergunta, podendo complementar com informações da internet quando necessário."
        "Os tópicos principais estão destacado entre aspas duplas. Se a resposta não"
        " estiver no contexto, diga que não sabe e peça mais detalhes para o questionamento:\n\n{context}"
    )),
    # Mensagem do usuário (onde vai a pergunta)
    ("user", "{question}")
])

# Cria uma cadeia que combina documentos com o prompt
chain = create_stuff_documents_chain(llm=chat, prompt=qa_prompt)

# Carrega o banco de dados vetorial (onde estão os documentos processados)
vectorstore = FAISS.load_local(
    diretorio_vectorestore_faiss,
    embeddings_model,
    allow_dangerous_deserialization=True  # Permite carregar o arquivo (necessário para FAISS)
)

# -----------------------------------------------------------------------------
# 8. INTERFACE DO USUÁRIO - A Parte Visual que o Usuário Vê
# -----------------------------------------------------------------------------

# TÍTULO PRINCIPAL - Bem no topo da página
# Usa HTML personalizado com a classe CSS que definimos
st.markdown("""
<div class='titulo-principal'>
    <h1>🤖 Assistente de IA: Biometria Inteligente para Aquicultura</h1>
    <p>Faça perguntas sobre o documento carregado e obtenha respostas precisas</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. INICIALIZAÇÃO DA MEMÓRIA - Preparando as Variáveis de Estado
# -----------------------------------------------------------------------------

# O Streamlit "esquece" tudo quando você interage com a página
# st.session_state é como a "memória" que guarda informações entre interações

# Verifica se as variáveis de controle existem na memória, se não, cria elas
if "campo_habilitado" not in st.session_state:
    st.session_state["campo_habilitado"] = False  # Campo de pergunta desabilitado inicialmente

if "resposta_exibida" not in st.session_state:
    st.session_state["resposta_exibida"] = False  # Nenhuma resposta sendo exibida inicialmente

# -----------------------------------------------------------------------------
# 10. LAYOUT PRINCIPAL - Organizando os Elementos na Página
# -----------------------------------------------------------------------------

# Divide a página em 3 colunas com proporções [1, 2, 1]
# Coluna 1: 25% da largura (espaço lateral esquerdo)
# Coluna 2: 50% da largura (conteúdo principal - botões)
# Coluna 3: 25% da largura (informações do sistema)
header_col1, header_col2, header_col3 = st.columns([1, 2, 1])

# COLUNA CENTRAL - Onde ficam os botões principais
with header_col2:
    # Verifica o estado atual para mostrar o botão correto
    if st.session_state.get("resposta_exibida", False):
        # Se já mostrou uma resposta, mostra botão para fazer nova pergunta
        btn_nova_pergunta = st.button(
            "🔄 Fazer Nova Pergunta",
            type="primary",  # Estilo de botão principal (colorido)
            use_container_width=True  # Ocupa toda a largura disponível
        )
    else:
        # Se ainda não fez nenhuma pergunta, mostra botão para iniciar
        btn_iniciar = st.button(
            "📝 Iniciar Conversa",
            type="primary",
            use_container_width=True
        )

# COLUNA DIREITA - Informações do sistema
with header_col3:
    # Cria uma seção expansível com informações técnicas
    with st.expander("ℹ️ Sistema", expanded=False):  # Começa fechada
        # Mostra informações sobre o documento carregado
        st.caption(f"📄 **Doc:** {caminho_arquivo.split('/')[-1]}")  # Nome do arquivo
        st.caption(f"📊 **Páginas:** {len(documentos)}")  # Quantas páginas tem
        st.caption(f"🔤 **Chunks:** {len(chunks)}")  # Quantos pedaços foram criados
        st.caption(f"🤖 **Modelo:** {modelo}")  # Qual IA está sendo usada

# -----------------------------------------------------------------------------
# 11. LÓGICA DOS BOTÕES - O que Acontece Quando Clica nos Botões
# -----------------------------------------------------------------------------

# Se já tem resposta exibida e clicou no botão de nova pergunta
if st.session_state.get("resposta_exibida", False):
    # Verifica se o botão foi clicado (precisa existir na memória local)
    if 'btn_nova_pergunta' in locals() and btn_nova_pergunta:
        limpar_area_resposta()  # Limpa tudo da conversa anterior
        st.session_state["campo_habilitado"] = True  # Habilita o campo de pergunta
        st.session_state["resposta_exibida"] = False  # Marca que não tem resposta exibida
        st.rerun()  # Atualiza a página (recarrega)

# Se não tem resposta e clicou no botão de iniciar
else:
    # Verifica se o botão foi clicado
    if 'btn_iniciar' in locals() and btn_iniciar:
        st.session_state["campo_habilitado"] = True  # Habilita o campo de pergunta

# -----------------------------------------------------------------------------
# 12. INTERFACE DE PERGUNTA - Onde o Usuário Digita suas Perguntas
# -----------------------------------------------------------------------------

# Só mostra o campo de pergunta se:
# 1. O campo está habilitado E
# 2. Não tem resposta sendo exibida
if st.session_state.get("campo_habilitado", False) and not st.session_state.get("resposta_exibida", False):

    # Container principal para o campo de pergunta - agora próximo ao botão
    st.markdown("""
    <div class='campo-pergunta-container'>
    """, unsafe_allow_html=True)

    # Wrapper interno para controlar largura máxima
    st.markdown("""
    <div class='campo-pergunta-wrapper'>
    """, unsafe_allow_html=True)

    # Título da seção - agora usando classe CSS específica
    st.markdown("""
    <div class='titulo-pergunta'>
        <h3>💬 Digite sua pergunta:</h3>
    </div>
    """, unsafe_allow_html=True)

    # Campo de entrada estilo chat
    # Este é o campo onde o usuário digita a pergunta
    pergunta = st.chat_input("Digite sua pergunta aqui e pressione Enter...")

    # Se o usuário digitou algo e pressionou Enter
    if pergunta:
        try:
            # Mostra animação de carregamento enquanto processa
            with st.spinner('🔄 Processando...'):
                # Chama nossa função principal que conversa com a IA
                resposta_formatada, resposta_do_chat = processar_pergunta(pergunta, vectorstore, modelo)

            # Salva tudo na memória da sessão para usar depois
            st.session_state["ultima_pergunta"] = pergunta  # A pergunta feita
            st.session_state["ultima_resposta"] = resposta_formatada  # A resposta formatada
            st.session_state["ultima_resposta_completa"] = resposta_do_chat  # Resposta completa
            st.session_state["resposta_exibida"] = True  # Marca que tem resposta
            st.session_state["campo_habilitado"] = False  # Desabilita o campo

            # Atualiza a página para mostrar a resposta
            st.rerun()

        except Exception as e:
            # Se deu erro, mostra mensagens de ajuda
            st.error(f"❌ Erro: {str(e)}")
            st.info("💡 Tente reformular sua pergunta.")

    # Instrução para o usuário - agora usando classes CSS específicas
    st.markdown("""
    <div class='dica-container'>
        <div class='dica-box'>
            💡 <strong>Dica:</strong> Digite sua pergunta e pressione <strong>Enter</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fecha os containers HTML
    st.markdown("</div>", unsafe_allow_html=True)  # Fecha campo-pergunta-wrapper
    st.markdown("</div>", unsafe_allow_html=True)  # Fecha campo-pergunta-container

# -----------------------------------------------------------------------------
# 13. EXIBIÇÃO DA RESPOSTA - Mostrando o Resultado para o Usuário
# -----------------------------------------------------------------------------

# Se tem resposta para mostrar
elif st.session_state.get("resposta_exibida", False):

    # Divide em 2 colunas: 80% para resposta, 20% para dicas
    resp_main_col, resp_side_col = st.columns([4, 1])

    # COLUNA PRINCIPAL - A resposta da IA
    with resp_main_col:
        # Título da seção
        st.markdown("**🎯 Resposta:**")

        # Mostra qual foi a pergunta original
        st.markdown(f"**Pergunta:** {st.session_state['ultima_pergunta']}")

        # Container estilizado para a resposta
        # Usa a classe CSS que definimos
        st.markdown("""
        <div class='resposta-container'>
        """, unsafe_allow_html=True)

        # A resposta da IA
        st.write(st.session_state["ultima_resposta"])

        # Fecha o container
        st.markdown("</div>", unsafe_allow_html=True)

    # COLUNA LATERAL - Dicas para o usuário
    with resp_side_col:
        st.info("💡 Use o botão acima para nova pergunta")

    # SEÇÃO DE DOCUMENTOS FONTE - Mostra de onde veio a informação
    if "ultima_resposta_completa" in st.session_state:
        resposta_completa = st.session_state["ultima_resposta_completa"]

        # Se tem documentos fonte na resposta
        if 'source_documents' in resposta_completa and resposta_completa['source_documents']:
            # Seção expansível para mostrar as fontes
            with st.expander("📚 Documentos fonte", expanded=False):
                # Divide em 2 colunas para mostrar os documentos
                doc_col1, doc_col2 = st.columns(2)

                # Para cada documento fonte encontrado
                for i, doc in enumerate(resposta_completa['source_documents']):
                    # Alterna entre as colunas (par na primeira, ímpar na segunda)
                    with doc_col1 if i % 2 == 0 else doc_col2:
                        # Título do documento
                        st.caption(f"**📄 Doc {i + 1}:**")

                        # Conteúdo do documento (limitado a 300 caracteres)
                        conteudo = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        st.text(conteudo)

                        # Metadados (informações extras) se existirem
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.caption(f"Meta: {doc.metadata}")

# -----------------------------------------------------------------------------
# 14. ESTADO INICIAL - O que Mostrar Quando a Página Carrega pela Primeira Vez
# -----------------------------------------------------------------------------

# Se não tem campo habilitado nem resposta exibida (estado inicial)
else:
    # Cria uma coluna centralizada para a mensagem de boas-vindas
    welcome_col = st.columns([1, 2, 1])[1]  # Pega só a coluna do meio

    with welcome_col:
        # Mensagem explicativa para o usuário
        st.info("👆 **Clique em 'Iniciar Conversa' para começar**")

# =============================================================================
# FIM DO PROGRAMA
# =============================================================================
# Este programa cria uma interface completa para conversar com documentos PDF
# usando inteligência artificial. O usuário pode fazer perguntas e receber
# respostas baseadas no conteúdo do documento.
# =============================================================================