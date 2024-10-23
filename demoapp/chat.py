import gradio as gr
import os
import logging
from dotenv import load_dotenv, find_dotenv

from prompts_en import (
    default_system_prompt,
    default_task_prompt,
    default_formatting_instruction,
)



from langchain_core.language_models import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_chroma import Chroma 
# from langchain_community.embeddings import VertexAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings

import gradio as gr

_ = load_dotenv(find_dotenv())  # read local .env file

logging.basicConfig(level=logging.DEBUG)

CHROMA_PERSISTENT_DIRECTORY = "data/chromadb"
VERTEX_EMBEDDING_MODEL_NAME = "text-multilingual-embedding-002"

CUSTOM_DOCUMENTS_DIRECTORY = "data/custom_documents"


MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = 0.2
TOP_K = 20


### Open AI GPT-4
def get_llm_openai_gpt_4(
    temperature, max_tokens=MAX_OUTPUT_TOKENS, top_p=TOP_P
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    
    MODEL = "gpt-4o"

    return ChatOpenAI(
        temperature=temperature, max_tokens=max_tokens, top_p=top_p, model=MODEL
    )


### Gemini 1.5
def get_llm_vertexai_gemini_1_5(
    temperature, max_tokens=MAX_OUTPUT_TOKENS, top_p=TOP_P, top_k=TOP_K
) -> BaseChatModel:
    """Gemini 1.5"""
    from langchain_google_vertexai import VertexAI
    MODEL = "gemini-1.5-flash"

    return VertexAI(
        model_name=MODEL,
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        verbose=True,
    )


### Llama 3.1 via API on Vertex AI
def get_llm_llama_3_1_api(
    temperature, max_tokens=MAX_OUTPUT_TOKENS, top_p=TOP_P
) -> BaseChatModel:
    """Llama 3.1 via API on Vertex AI"""

    REGION = "us-central1" # Only us-central1 is supported
    ENDPOINT = f"{REGION}-aiplatform.googleapis.com"
    PROJECT_ID = "speeltuin-327308"

    # MODEL="meta/llama-3.1-405b-instruct-maas"
    # MODEL="meta/llama-3.1-70b-instruct-maas"
    MODEL = "meta/llama-3.1-8b-instruct-maas"

    # Refresh access token
    from google.auth import default
    from google.auth.transport.requests import Request

    credentials, project_id = default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_request = Request()
    credentials.refresh(auth_request)



    from langchain_openai import ChatOpenAI

    base_url = f"https://{ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi"
    return ChatOpenAI(
        base_url=base_url,
        model=MODEL,
        api_key=credentials.token,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )


# Choose LLM
llm = get_llm_vertexai_gemini_1_5(temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS, top_p=TOP_P, top_k=TOP_K)
# llm = get_llm_openai_gpt_4(temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS, top_p=TOP_P)
# llm = get_llm_llama_3_1_api(temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS, top_p=TOP_P)


## Vectorstore
embeddings = VertexAIEmbeddings(model_name=VERTEX_EMBEDDING_MODEL_NAME)

vectorstore = Chroma(
    persist_directory=CHROMA_PERSISTENT_DIRECTORY,
    embedding_function=embeddings,
    collection_name="popcorn",
)


# Custom documents

import json
def read_custom_documents() -> list[str]:
    custom_documents = []

    for filename in os.listdir(CUSTOM_DOCUMENTS_DIRECTORY):
        if filename.endswith(".txt"):
            with open(os.path.join(CUSTOM_DOCUMENTS_DIRECTORY, filename), "r") as f:
                custom_documents.append({"page_content":f.read()})
            if filename.endswith(".ndjson"):
                for r in f:
                    if str(r).strip() == "":
                        continue
                    try:
                        doc = json.loads(r)
                        if "page_content" in doc:
                            custom_documents.append(doc)
                        else:
                            logging.warning("Unexpected document format: %s" % r)
                    except json.decoder.JSONDecodeError as e:
                        logging.error(str(e))
                        logging.error("Raw line:\n %s" % r)
    logging.info("Loaded %d custom documents" % len(custom_documents))
    return custom_documents

custom_documents = read_custom_documents()


### LLM generation
## Simple predict w/o grounding

def generate(
    message,
    history,
    system_prompt,
    # temperature, max_tokens
):
    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content=system_prompt))
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))

    logging.debug(history_langchain_format)
    
    response = llm.invoke(history_langchain_format)
    if hasattr(response, "content"):
        return response.content
    else:
        return str(response)

## RAG: search in vectorstore and answer based on found documents 
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

rag_prompt_template = PromptTemplate.from_template(
template = """{task_prompt}
------------
SNIPPETS
{context}
------------
CHAT:
{question}
------------
""",
partial_variables={
    "task_prompt": default_task_prompt,
    # "formatting_instruction": default_formatting_instruction,
}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_generate(
        message,
    history,
    system_prompt,
):
    # find similar documents and add to the user question
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    formatted_docs = format_docs(retriever.invoke(message))
    rag_prompt = rag_prompt_template.format(context=formatted_docs, question=message)

    # logging.debug(rag_prompt)

    # generate using augmented prompt
    return generate(rag_prompt, history, system_prompt)


def main():

    demo = gr.ChatInterface(
        # generate,
        rag_generate,
        # type='messages',
        additional_inputs=[
            gr.Textbox(default_system_prompt, label="System Prompt"),
            # gr.Slider(0.0, 1.0, value=0, step=0.1, label="Temperature"),
            # gr.Slider(256, 4096, value=1024, step=10, label="Max.Tokens"),
        ],
        examples=[
            ["Can you tell me which horror movies should I watch?"],
            ["I like modern horror movies, can you recommend me a few to watch?"],
            [
                "I am in a mood for a claustrophobic horror movie. Which ones can you advice me to watch?"
            ],
            ["Can you advice me horror films to watch in the genre found footage?"],
            ["I liked the A Quiet Place from 2018, can you advice me similar movies?"],
            ["Can you recommend me a movie in the genre ..."],
        ],
    )
    demo.launch()


if __name__ == "__main__":
    main()
