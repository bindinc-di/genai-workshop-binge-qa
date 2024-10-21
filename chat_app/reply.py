# from langchain.llms import VertexAI
from langchain_google_vertexai import VertexAI
import requests
import json
import os
import logging
from dotenv import find_dotenv, load_dotenv

import numpy as np
import subprocess
import time

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)

SEARCH_BASE_URL = os.getenv("SEARCH_BASE_URL")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")


def translate_to_english(txt):
    # random delay not to upset the server
    # time.sleep(np.random.random()*5)

    txt = txt.replace("'", "''").replace("\n", "")
    command = f"curl --header 'Accept: text/json' --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0' --data-urlencode 'client=gtx' --data-urlencode 'sl=auto' --data-urlencode 'tl=en' --data-urlencode 'dt=at' --data-urlencode 'q={txt}' -sL http://translate.googleapis.com/translate_a/single | jq -r '.[5][][2][0][0]'"
#     logging.info(command)
    out = subprocess.run(command, shell=True, stdout=subprocess.PIPE)

    return out.stdout.decode("utf-8")  # .replace("\n","")

def get_llm(max_output_tokens, temperature, top_p, top_k):

    return VertexAI(
        model_name='gemini-1.5-flash',
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        verbose=True
    )


def build_prompt(system_prompt, task_prompt, formatting_instruction, chat_history, chunks):
    hist = "\n".join(
        [f"{m['name'].upper()}: {m['text']}" for m in chat_history])

    chunks = "\n\n".join([c["chunk"] for c in chunks])
    chunks = chunks + "\n\n".join(["\n\n".join([c["chunk"] for c in msg["chunks"][:2]])
                                  for msg in chat_history if msg["chunks"] != None])

    # chunks = translate_to_english(chunks)

    prompt = f"""
{system_prompt}
------------
{task_prompt}
------------
SNIPPETS
{chunks}
------------
CHAT:
{hist}
------------
{formatting_instruction}
"""
    logging.debug("PROMPT IS %s" % prompt)
    return prompt


def search_documents(question):
    url = f'{SEARCH_BASE_URL}/similarity'
    data = {
        "query": question
    }
    headers = {
        "Authorization": f"Bearer {SEARCH_API_KEY}"
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()


def reply(history, system_prompt, task_prompt, formatting_instruction, generation_params):
    chunks = search_documents(history[-1]["text"])
    llm = get_llm(**generation_params)
    raw_response = llm.invoke(build_prompt(system_prompt, task_prompt, formatting_instruction, history, chunks))
    logging.debug("RAW LLM RESPONDE IS\n%s" % raw_response)
    response = "".join([row for row in raw_response.split("\n") if "`" not in row])

    try:
        response = json.loads(response)
    except json.decoder.JSONDecodeError as e:
        logging.error(str(e))
        logging.error("Raw LLM response:\n %s" % raw_response)
        return ["Error: got a malformatted answer from LLM"], []


    if not response["in_snippets"] and not response["in_chat"]:
        response["response"] = "Sorry, I was not provided with this information yet."
    return response["response"], chunks
