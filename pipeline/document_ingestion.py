# %% [markdown]
# # Ingestion of metadata

# %% [markdown]
# ## Loading dataset

# %%
# Install dependencies
#! pip install faiss-cpu

# %%
import json
import os
from langchain.schema import Document
from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from tqdm import tqdm
from argparse import ArgumentParser
import time
import subprocess
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Load all documents in one go or in chunks
LOAD_IN_CHUNKS = True
CHUNK_SIZE = 100

VERTEX_EMBEDDING_MODEL_NAME = "text-multilingual-embedding-002"

# %%
# import os
# os.getcwd()
# file_path = "../../api_app/data/raw/binge_apporved_animation.json"

# if set english description texts will be first translated to English


# USE_API = "openai"
USE_API = "vertexai"


parser = ArgumentParser()
parser.add_argument("filepath")
parser.add_argument("--language", required=False, default="nl")
parser.add_argument("--output-path", required=False, default="output")

args = parser.parse_args()
print("Passed arguments:", args)
# %%
data = pd.read_json(args.filepath, lines=True)
data.head(10)

_language = args.language
faiss_index_save_path = os.path.join(args.output_path, "index")
os.makedirs(faiss_index_save_path, exist_ok=True)
# faiss_index_save_path = os.path.join(os.path.split(args.filepath)[0], "index")

# %% [markdown]
# ## Converting structured data into natural language
#
# The following functions makes a compilation of some of the metadata above, turning the information into a fluid human-readable text. This is what the Large Language Model will use for reasoning, which is compatible to the inputs (questions) provided by the users.

# %%


class TEXTS:
    AVAILABLE_FOR_RENT = {
        "en": "{} is available for rental on {}. ",
        "nl": "{} is nu te huur op {}.",
    }
    AVAILABLE_FOR_PURCH = {
        "en": "{} is available for purchase on {}. ",
        "nl": "{} is nu te koop op {}. "
    }
    AVAILABLE_FOR_SUBS = {
        "en": "{} is available for subscription on {}. ",
        "nl": "{} is nu te kijken met abonnement op {}. "
    }
    AVAILABLE = {
        "en": "{} is available on {}. ",
        "nl": "{} is nu te kijken op {}. "
    }
    IS_THE = {
        "en": " is the {}",
        "nl": " is de {}"
    }
    THAT_PLAYS = {
        "en": " that plays {}",
        "nl": " wie speelt {}"
    }
    ORIGINAL_TITLE = {
        "en": "original title: {}",
        "nl": "originele titel: {}"
    }
    ALTERNATIVE_TITLE = {
        "en": "alt title: {}",
        "nl": "alternatieve titel: {}"
    }
    IS_A = {
        "en": "{} is a {} ({})",
        "nl": "{} is een {} ({})"
    }


def desc_person(details, language):
    """receives a dict"""

    if pd.isna(details):
        return ""

    text = details['credit_name']
    # text = f"{text} is the {details['credit_role']}"
    text += TEXTS.IS_THE[language].format(details['credit_role'])
    if "character" in details:
        # text = f"{text} that plays {details['character']}"
        text += TEXTS.THAT_PLAYS[language].format(details['character'])
    return f"{text}".replace("\n", " ")


def desc_cast_crew(details, language):
    """receives an array of dicts"""
    if type(details) != list:
        return ""

    return "; ".join([desc_person(d, language) for d in details if d["credit_role"] in ["actor", "director", "composer"]])


def desc_avail(details, title, language):
    """receives an array of dicts"""

    if type(details) != list:
        return ""

    rent = [av for av in details if "availability_type" in av and av["availability_type"] == "rental"]
    purch = [
        av for av in details if "availability_type" in av and av["availability_type"] == "purchase"]
    subs = [av for av in details if "availability_type" in av and av["availability_type"] == "subscription"]
    unk = [av for av in details if "availability_type" not in av]

    text = ""
    if len(rent) > 0:
        channels = ', '.join(set([av['channel'] for av in rent]))
        text += TEXTS.AVAILABLE_FOR_RENT[language].format(title, channels)
        # text = f"{text}{title} is available for rental on {', '.join(set([av['channel'] for av in rent]))}."

    if len(purch) > 0:
        channels = ', '.join(set([av['channel'] for av in purch]))
        text += TEXTS.AVAILABLE_FOR_PURCH[language].format(title, channels)
        # text = f"{text}{title} is available for purchase on {', '.join(set([av['channel'] for av in purch]))}."

    if len(subs) > 0:
        channels = ', '.join(set([av['channel'] for av in subs]))
        text += TEXTS.AVAILABLE_FOR_SUBS[language].format(title, channels)
        # text = f"{text}{title} is available for subscription on {', '.join(set([av['channel'] for av in subs]))}."

    if len(unk) > 0:
        channels = ', '.join(set([av['channel'] for av in unk]))
        text += TEXTS.AVAILABLE[language].format(title, channels)
        # text = f"{text}{title} is available on {', '.join(set([av['channel'] for av in unk]))}."
    return text


def desc_title(details, language):
    """receives a dict"""

    if pd.isna(details):
        return ""

    if "title" not in details:
        return list(details.keys())[0]

    text = details["title"]

    others = []
    if "original_title" in details:
        # others.append(f"original title: {details['original_title']}")
        others.append(TEXTS.ORIGINAL_TITLE[language].format(
            details['original_title']))
    if "alternative_title" in details:
        # others.append(f"alt title: {details['alternative_title']}")
        others.append(TEXTS.ALTERNATIVE_TITLE[language].format(
            details['alternative_title']))

    if len(others) > 0:
        text = f"{text} ({'; '.join(others)})"

    return f"{text}"


def desc_category(details, title, language):
    """receives a dict"""

    if pd.isna(details):
        return ""

    # text = f"{title} is a {details['category']} ({', '.join(details['genres'])})"
    text = TEXTS.IS_A[language].format(title,
                                       details['category'], ', '.join(details['genres']))
    return f"{text}. "


def translate_to_english(txt):
    # random delay not to upset the server
    time.sleep(np.random.random()*5)

    txt = txt.replace("'", "''").replace("\n", "")
    command = f"curl --header 'Accept: text/json' --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0' --data-urlencode 'client=gtx' --data-urlencode 'sl=auto' --data-urlencode 'tl=en' --data-urlencode 'dt=at' --data-urlencode 'q={txt}' -sL http://translate.googleapis.com/translate_a/single | jq -r '.[5][][2][0][0]'"
#     print(command)
    out = subprocess.run(command, shell=True, stdout=subprocess.PIPE)

    return out.stdout.decode("utf-8").replace("\n", "")


def desc_desc(details, language):
    """receives an array of dicts"""

    if type(details) != list:
        return ""

    txt = "".join([desc["text"] for desc in details])
    if language == "en":
        txt = translate_to_english(txt)

    return f"Synopsis: {txt}"

# %% [markdown]
# So a current metadata


# %%
data.loc[42]["available"]

# %% [markdown]
# becomes fluid text:

# %%
desc_avail(data.loc[42]["available"], data.loc[42]
           ["title"], language=_language)

# %% [markdown]
# ## Batch processing
#
# Applying this conversion over the whole dataset so everything is ready to be ingested. Each title is going to be considered as a chunk of data.

# %%
tqdm.pandas()


def row2chunk(x, language):
    return (
        desc_category(x["typology"], desc_title(
            x["title"], language), language)
        + "\n" +
        desc_desc(x["content_descriptions"], language)
        + "\n" +
        desc_cast_crew(x["cast_and_crew"], language)
        + "\n" +
        desc_avail(x["available"], desc_title(x["title"], language), language)
    )


print("Converting")
chunks = data.progress_apply(
    lambda x: {
        "page_content": row2chunk(x, language=_language),
        "metadata": {
            "id": x.name,
            "bindinc_uri": x["bindinc_api"][0]["uri"]
        }
    }, axis=1
)

# %% [markdown]
# Each chunk of data is going to be ingested as described below. In addition to the text chunk itself, some references are appended as metadata in order to retrieve specific information from the dataset to feed some application interface features.

# %%
chunks[42]

# %%
# print(chunks[42]["page_content"])

# %% [markdown]
# ## Ingestion
#
# From `chunks` array (of snippets/text chunks describing titles), we can now enclose them in `Document` objects that will be indexed in a vector database.

# %%

# %%
docs = [Document.parse_obj(c) for c in chunks]

# %%


def get_embeddings(type):
    if type == "vertexai":
        embeddings = VertexAIEmbeddings(
            #     model_name="textembedding-gecko@001" # stable,
            #     model_name="textembedding-geckolatest",
            model_name=VERTEX_EMBEDDING_MODEL_NAME,
            chunk_size=1
        )
    elif type == "openai":
        embeddings = OpenAIEmbeddings(
            chunk_size=1
        )
    else:
        raise ValueError("Unknown embedding API:", type)
    return embeddings


embeddings = get_embeddings(USE_API)

# %%
print("Initializing vector store from embeddings", USE_API)
if LOAD_IN_CHUNKS:
    faiss = None
    for i in tqdm(range(0, len(docs), CHUNK_SIZE), total=len(docs) // CHUNK_SIZE + 1):
        doc_chunk = docs[slice(i, i+CHUNK_SIZE)]
        if i == 0:
            faiss = FAISS.from_documents(doc_chunk, embeddings)
        faiss.add_documents(doc_chunk)
else:
    faiss = FAISS.from_documents(docs, embeddings)


# %%
print("Saving vector store to file system", faiss_index_save_path)
# faiss_index_save_path = "../../api_app/data/index"
faiss.save_local(faiss_index_save_path, "index")

# %%


def test_embedding():

    # %% [markdown]
    # ## Retrieving contents
    #
    # Reset kernel, load vector store and search by similarity.

    # %%

    # %%
    embeddings = get_embeddings(USE_API)

    # %%
    store = FAISS.load_local(faiss_index_save_path, embeddings, "index")

    # %%
    test_instruction = {
        "nl": "Wat is het verhaal van Nemo?",
        "en": "What is a storyof Nemo?"
    }
    result = store.similarity_search(test_instruction[_language])

    # %%
    print([{"chunk": r.page_content, "metadata": r.metadata} for r in result])

    # %%

    # %%
    print(json.dumps([{"chunk": r.page_content, "metadata": r.metadata}
                      for r in result]))


test_embedding()

print("Done.")
