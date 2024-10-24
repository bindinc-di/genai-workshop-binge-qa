"""
Gradio app for testing semantic search
"""
from argparse import ArgumentParser
from datetime import datetime
from argparse import ArgumentParser
import gradio as gr

from tqdm import tqdm
from timeit import default_timer
import chromadb
from chromadb.api import ClientAPI
import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

VERTEX_EMBEDDING_MODEL_NAME = "text-multilingual-embedding-002"

# default path to chroma DB files
CHROMA_DATA_PATH = "data/chromadb"

SEARCH_N_RESULTS = 5


_chroma_client = None

# dimensionality = 256

def embed_text(texts, dimensionality=None) -> list[list[float]]:
    """Embeds texts with a pre-trained, foundational model.

    Returns:
        A list of lists containing the embedding vectors for each input text
    """

    # A list of texts to be embedded.
    # texts = ["banana muffins? ", "banana bread? banana muffins?"]
    # The dimensionality of the output embeddings.
    # The task type for embedding. Check the available tasks in the model's documentation.
    task = "RETRIEVAL_DOCUMENT"

    model = TextEmbeddingModel.from_pretrained(VERTEX_EMBEDDING_MODEL_NAME)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)

    # print(embeddings)
    # Example response:
    # [[0.006135190837085247, -0.01462465338408947, 0.004978656303137541, ...], [0.1234434666, ...]],
    return [embedding.values for embedding in embeddings]


def item_as_html(item):
    return f"""<div>
    <p><strong>{item['id']}</strong> distance: {item['distance']}</p>
    <p>{item['document'][:500]}</p>
    </div>"""

def items_as_html(items):
    res = "<ul>"
    for item in items:
        res += f"<li>{item_as_html(item)}</li>"
    res +="</ul>"
    return res

def on_search_similar(query, collection_name):
     
    client = get_chroma_client()

    print(f"Searching for similar items ina collection {collection_name}...")
    similar_items = search_similar(query, collection_name, client)

    # return (item_as_html(item), items_as_html(similar_items[:5]))
    return items_as_html(similar_items)


def demoapp():
    with gr.Blocks(
        # theme=gr.themes.Glass(), 
        theme=gr.themes.Soft(),
        title="Search similar movies and series") as demo:

        similar_items = gr.State([])

        gr.Markdown("# Semantic search on ChromaDB")
        with gr.Group():
            gr.Markdown("## Chose a collection and enter search string")
            with gr.Row():
                query = gr.Textbox(label="Search string", placeholder="Enter text")
                # content_type=gr.Dropdown(label="Content type", choices=["movie", "series"], value="movie")
                collection_name=gr.Radio(label="collection", choices=["popcorn", "other"], value="popcorn")
                # btn_lookup = gr.Button("Search movie")
        
        with gr.Row():
            btn_find_similar = gr.Button("Search", variant="primary")
        
        with gr.Row():
            # gr.Markdown("## Main item")
            with gr.Row():
                # my_item = gr.Markdown(label="Main item")
                my_item  = gr.HTML(label="Main item")
        
        with gr.Group():
            gr.Markdown("## Similar items")
            with gr.Row():
                similar_items = gr.HTML()
                # similar_items = gr.DataFrame()
            # @gr.render(inputs=similar_items)
            # def show_similar(items):

            #     print(my_item)
            #     print(similar_items)

        
        btn_find_similar.click(on_search_similar, inputs=[query, collection_name], outputs=[similar_items])

        ex = gr.Examples(
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
            inputs=[query, collection_name],
            run_on_click=False,
            fn=on_search_similar,
            outputs=[similar_items],
        )
    return demo


  
def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
    
        # create the client if it doesn't exist
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    return _chroma_client

def search_similar(query:str, collection_name:str, client: ClientAPI, n_results=SEARCH_N_RESULTS):
    collection = client.get_collection(collection_name)

    emb = embed_text([query])

    similar_items = collection.query(emb, n_results=n_results)
    # print_similar(similar_items, main_id)
    print(f"Found {len(similar_items)} similar items")
    return (


        [ 
            # (id, round(float(dist), 3), doc)
            { "id": id, "document": doc, "distance": dist}
            for id, dist, doc in zip(
                similar_items['ids'][0], 
                similar_items["distances"][0],
                similar_items['documents'][0], 
            ) 
        ]

        # pd.DataFrame([ (id, round(float(dist), 3), doc)
        #     for id, dist, doc in zip(
        #         similar_items['ids'][0], 
        #         similar_items["distances"][0],
        #         similar_items['documents'][0], 
        #     ) 
        #     if int(id) != int(main_id)
        #     ], columns=("Id", "Distance", "Document"))
    )


def print_similar(similar_items, search_id):
    """Utility function: Shows found similar items"""
    for id, doc, dist in zip(similar_items['ids'][0], similar_items['documents'][0], similar_items["distances"][0]):
      if int(id) == int(search_id):
        continue
      print(f"{id} ({round(dist,3)}):\t{doc}\n")

def main():
    demo = demoapp()
    demo.launch(
        # share=False, 
        debug=True,
    )


if __name__ == "__main__":
    main()
