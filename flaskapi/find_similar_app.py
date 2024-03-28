import os
import json
import re
import gradio as gr
import pandas as pd
import requests
from argparse import ArgumentParser

URL = "http://127.0.0.1:5000/similarity"


def search(search_string: str):

    if not str.strip(search_string):
        return [{"chunk": "Provide non empty query text!", "metadata": {"bindinc_uri": "", "id": ""}}]

    api_key: str = os.getenv("SEARCH_API_KEY", "DUMMY_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}

    data = {
        "query": search_string
    }

    response = requests.post(URL, json=data, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {json.dumps(response.json(), indent=2)}")

    return response.json()


def search_pdf(search_string: str):
    data = search(search_string)
    return pd.json_normalize(data)


def clean_output(*args, **kwargs):
    return ([])


def item_seleted(evt: gr.SelectData):
    """Put Synopsis of the chosen item into the search"""
    val = evt.value
    m = re.search(r"Synopsis:\s+(.*)", val, re.IGNORECASE)
    if m:
        return m.groups()[0]
    return val


def get_examples():
    return [
        """Volg Dr. Ellen Garland en Dr. Michelle Fournet, twee wetenschappers die zich toelegden op de studie van bultruggen en hun sociale communicatie""",
        """Maak kennis met de Enchantimals, een groep schattige meisjes die een speciale band hebben met hun dier-besties""",
    ]


def demoapp():

    with gr.Blocks() as demoapp:
        gr.Markdown("# Search Similair Content")
        # with gr.Row():
        in_query = gr.TextArea(label="Query Text")
        with gr.Row():
            btn_cln = gr.Button("Clean", variant="secondary")
            btn_gen = gr.Button("Submit", variant="primary")
        # out_tab = gr.TextArea(label="Found Similar Content")
        init_tab = search_pdf("")
        out_tab = gr.Dataframe(
            value=init_tab, label="Found Similar Content", overflow_row_behaviour="paginate")
        btn_cln.click(clean_output, outputs=[out_tab])
        btn_gen.click(search_pdf, inputs=[in_query], outputs=[out_tab])

        out_tab.select(fn=item_seleted, outputs=[in_query])

        gr.Examples(
            examples=get_examples(),
            inputs=[in_query],
            run_on_click=True,
            fn=clean_output
        )
    return demoapp


def main_app():

    # parser = ArgumentParser()
    # parser.add_argument("file_path")

    # args = parser.parse_args()
    demo = demoapp()
    demo.launch(debug=True)


if __name__ == "__main__":
    main_app()
