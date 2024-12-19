import streamlit as st

from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory


st.set_page_config(
    page_title="jimmny-assign",
    page_icon="./.streamlit/ham.ico",
)


if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)

memory = st.session_state["memory"]


# callback : listen events
class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token  # f"{self.mseeage}{token}"
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_data(show_spinner="Ebedding files...")
def embed_file(file):
    file_content = file.read()

    # cache files
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # for cache embeddings
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    loader = UnstructuredFileLoader(file_path)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context.
            If you don't know the answer, just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


st.title("Assignment")

st.markdown(
    """
    This is 'DocumentGPT'

    First, upload your file on the sidebar.
    """
)


with st.sidebar:
    # File uploader
    file = st.file_uploader(
        "Upload a .txt or .pdf file",
        type=["txt", "pdf"],
    )


if file:
    retriever = embed_file(file)

    send_message("Go.", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything")

    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": load_memory,
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            result = chain.invoke(message)

        memory.save_context({"input": message}, {"output": result.content})

        # docs = retriever.invoke(message)
        # docs = "\n\n".join(document.page_content for document in docs)
        # p = prompt.from_messages(context=docs, question=message)
        # llm.predict_message(p)

        memory.load_memory_variables({})["history"]

else:
    st.session_state["messages"] = []  # Init
