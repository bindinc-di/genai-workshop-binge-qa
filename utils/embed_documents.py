from __future__ import annotations

from argparse import ArgumentParser
import os
import json
from tqdm import tqdm


from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

VERTEX_EMBEDDING_MODEL_NAME = "text-multilingual-embedding-002"
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

def main(input_filepath, output_filepath=None):

    assert os.path.exists(input_filepath), f"File does not exist {input_filepath}"

    if not(output_filepath):
        output_filepath = os.path.splitext(input_filepath)[0] + "_embeddings.ndjson"

    with open(input_filepath, mode="r+") as file:
        line_count = sum(1 for _ in file)
        file.seek(0)
        with open(output_filepath, mode="w+") as output_file:
            for line in tqdm(file, total=line_count):
                if line.strip() == "":
                    continue
                doc = json.loads(line)
                if "page_content" in doc:
                    doc["embedding"] = embed_text([doc["page_content"]])[0]
                else:
                    print("Can't find page content in line:" % line)
                output_file.write(json.dumps(doc) + "\n")
                # output_file.flush()
    print(f"Data written to the file {output_filepath}")



if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("input_filepath")
    # parser.add_argument("output_filepath")

    args = parser.parse_args()
    main(args.input_filepath)

    # embed_text()