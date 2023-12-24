"""Builds a CLI, Webhook, and Gradio app for Q&A on the Full Stack corpus.

For details on corpus construction, see the accompanying notebook."""
import modal
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

import vecstore
from utils import pretty_log


# definition of our container image for jobs on Modal
# Modal gets really powerful when you start using multiple images!
image = modal.Image.debian_slim(  # we start from a lightweight linux distro
    python_version="3.10"  # we add a recent Python version
).pip_install(  # and we install the following packages:
    "langchain==0.0.184",
    # 🦜🔗: a framework for building apps with LLMs
    "openai~=0.27.7",
    # high-quality language models and cheap embeddings
    "tiktoken",
    # tokenizer for OpenAI models
    "faiss-cpu",
    # vector storage and similarity search
    "pymongo[srv]==3.11",
    # python client for MongoDB, our data persistence solution
    "gradio~=3.34",
    # simple web UIs in Python, from 🤗
    "gantry==0.5.6",
    # 🏗️: monitoring, observability, and continual improvement for ML systems
)

# we define a Stub to hold all the pieces of our app
# most of the rest of this file just adds features onto this Stub
stub = modal.Stub(
    name="askfsdl-backend",
    image=image,
    secrets=[
        # this is where we add API keys, passwords, and URLs, which are stored on Modal
        modal.Secret.from_name("mongodb-fsdl"),
        modal.Secret.from_name("openai-api-key-fsdl"),
        modal.Secret.from_name("gantry-api-key-fsdl"),
    ],
    mounts=[
        # we make our local modules available to the container
        modal.Mount.from_local_python_packages(
            "vecstore", "docstore", "utils", "prompts"
        )
    ],
)

VECTOR_DIR = vecstore.VECTOR_DIR
vector_storage = modal.NetworkFileSystem.persisted("vector-vol")


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
)
@modal.web_endpoint(method="GET")
def web(query: str, request_id=None):
    """Exposes our Q&A chain for queries via a web endpoint."""
    import os

    pretty_log(
        f"handling request with client-provided id: {request_id}"
    ) if request_id else None

    answer = qanda(
        query,
        request_id=request_id,
        with_logging=bool(os.environ.get("GANTRY_API_KEY")),
    )
    return {"answer": answer}


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
    keep_warm=1,
)
def qanda(query: str, request_id=None, with_logging: bool = False) -> str:
    """Runs sourced Q&A for a query using LangChain.

    Arguments:
        query: The query to run Q&A on.
        request_id: A unique identifier for the request.
        with_logging: If True, logs the interaction to Gantry.
    """
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.chat_models import ChatOpenAI

    import prompts
    import vecstore

    embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

    pretty_log("connecting to vector storage")
    vector_index = vecstore.connect_to_vector_index(
        vecstore.INDEX_NAME, embedding_engine
    )
    pretty_log("connected to vector storage")
    pretty_log(f"found {vector_index.index.ntotal} vectors to search over")

    pretty_log(f"running on query: {query}")
    pretty_log("selecting sources by similarity to query")
    sources_and_scores = vector_index.similarity_search_with_score(query, k=3)

    sources, scores = zip(*sources_and_scores)

    pretty_log("running query against Q&A chain")

    llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=256)
    chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        verbose=with_logging,
        prompt=prompts.main,
        document_variable_name="sources",
    )

    result = chain(
        {"input_documents": sources, "question": query}, return_only_outputs=True
    )
    answer = result["output_text"]

    if with_logging:
        print(answer)
        pretty_log("logging results to gantry")
        record_key = log_event(query, sources, answer, request_id=request_id)
        if record_key:
            pretty_log(f"logged to gantry with key {record_key}")

    return answer


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
    cpu=8.0,  # use more cpu for vector storage creation
)
def create_vector_index(collection: str = None, db: str = None):
    """Creates a vector index for a collection in the document database."""
    import docstore

    pretty_log("connecting to document store")
    db = docstore.get_database(db)
    pretty_log(f"connected to database {db.name}")

    collection = docstore.get_collection(collection, db)
    pretty_log(f"collecting documents from {collection.name}")
    docs = docstore.get_documents(collection, db)

    pretty_log("splitting into bite-size chunks")
    ids, texts, metadatas = prep_documents_for_vector_storage(docs)

    pretty_log(f"sending to vector index {vecstore.INDEX_NAME}")
    embedding_engine = vecstore.get_embedding_engine(disallowed_special=())
    vector_index = vecstore.create_vector_index(
        vecstore.INDEX_NAME, embedding_engine, texts, metadatas
    )
    vector_index.save_local(folder_path=VECTOR_DIR, index_name=vecstore.INDEX_NAME)
    pretty_log(f"vector index {vecstore.INDEX_NAME} created")


@stub.function(image=image)
def drop_docs(collection: str = None, db: str = None):
    """Drops a collection from the document storage."""
    import docstore

    docstore.drop(collection, db)


def log_event(query: str, sources, answer: str, request_id=None):
    """Logs the event to Gantry."""
    import os

    import gantry

    if not os.environ.get("GANTRY_API_KEY"):
        pretty_log("No Gantry API key found, skipping logging")
        return None

    gantry.init(api_key=os.environ["GANTRY_API_KEY"], environment="modal")

    application = "ask-fsdl"
    join_key = str(request_id) if request_id else None

    inputs = {"question": query}
    inputs["docs"] = "\n\n---\n\n".join(source.page_content for source in sources)
    inputs["sources"] = "\n\n---\n\n".join(
        source.metadata["source"] for source in sources
    )
    outputs = {"answer_text": answer}

    record_key = gantry.log_record(
        application=application, inputs=inputs, outputs=outputs, join_key=join_key
    )

    return record_key


def prep_documents_for_vector_storage(documents):
    """Prepare documents from document store for embedding and vector storage.

    Documents are split into chunks so that they can be used with sourced Q&A.

    Arguments:
        documents: A list of LangChain.Documents with text, metadata, and a hash ID.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100, allowed_special="all"
    )
    ids, texts, metadatas = [], [], []
    for document in documents:
        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        ids += [metadata.get("sha256")] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas

    return ids, texts, metadatas


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
)
def cli(query: str):
    answer = qanda.local(query, with_logging=False)
    pretty_log("🦜 ANSWER 🦜")
    print(answer)


web_app = FastAPI(docs_url=None)


@web_app.get("/")
async def root():
    return {"message": "See /gradio for the dev UI."}


@web_app.get("/docs", response_class=RedirectResponse, status_code=308)
async def redirect_docs():
    """Redirects to the Gradio subapi docs."""
    return "/gradio/docs"


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
    keep_warm=1,
)
@modal.asgi_app(label="askfsdl-backend")
def fastapi_app():
    """A simple Gradio interface for debugging."""
    import gradio as gr
    from gradio.routes import App

    def chain_with_logging(*args, **kwargs):
        return qanda(*args, with_logging=True, **kwargs)

    inputs = gr.TextArea(
        label="Question",
        value="What is zero-shot chain-of-thought prompting?",
        show_label=True,
    )
    outputs = gr.TextArea(
        label="Answer", value="The answer will appear here.", show_label=True
    )

    interface = gr.Interface(
        fn=chain_with_logging,
        inputs=inputs,
        outputs=outputs,
        title="Ask Questions About The Full Stack.",
        description="Get answers with sources from an LLM.",
        examples=[
            "What is zero-shot chain-of-thought prompting?",
            "Would you rather fight 100 LLaMA-sized GPT-4s or 1 GPT-4-sized LLaMA?",
            "What are the differences in capabilities between GPT-3 davinci and GPT-3.5 code-davinci-002?",  # noqa: E501
            "What is PyTorch? How can I decide whether to choose it over TensorFlow?",
            "Is it cheaper to run experiments on cheap GPUs or expensive GPUs?",
            "How do I recruit an ML team?",
            "What is the best way to learn about ML?",
        ],
        allow_flagging="never",
        theme=gr.themes.Default(radius_size="none", text_size="lg"),
        article="# GitHub Repo: https://github.com/the-full-stack/ask-fsdl",
    )

    interface.dev_mode = False
    interface.config = interface.get_config_file()
    interface.validate_queue_settings()
    gradio_app = App.create_app(
        interface, app_kwargs={"docs_url": "/docs", "title": "ask-FSDL"}
    )

    @web_app.on_event("startup")
    async def start_queue():
        if gradio_app.get_blocks().enable_queue:
            gradio_app.get_blocks().startup_events()

    web_app.mount("/gradio", gradio_app)

    return web_app
