from argparse import ArgumentParser
import json
import os
from datetime import datetime
from glob import glob

import chromadb

from embed_documents import embed_text
from load_documents_to_chroma import create_collection

DEFAULT_CHROMADB_PATH = "data/chromadb"

DEFAULT_CHUNK_SIZE = 1000

def make_chunks(text, chunk_size=DEFAULT_CHUNK_SIZE):

    # TODO split into chunks using DEFAULT_CHUNK_SIZE
    return [text]

def load_files_to_chromadb_collection(input_dir, chromadb_path, collection_name):

    if not os.path.isdir(input_dir):
        raise ValueError(f"{input_dir} is not a directory" )
    
    ds = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(input_dir, f"small_text_documents_{ds}.ndjson")
    docs = []
    ix = 0
    with open(output_path, mode="w+", encoding="utf-8") as out_f:
        for filepath in glob(input_dir+"/**.txt", recursive=True):
            if os.path.isfile(filepath) and filepath.endswith(".txt"):

                with open(filepath, mode="r", encoding="utf-8") as f:
                    ix += 1
                    text = f.read()
                    # chunk embed and load to collection
                    for chunk_number, chunk in enumerate(make_chunks(text)):

                        chunk_id = str(ix *1000 + chunk_number)
                        
                        doc = {
                            "page_content": chunk,
                            "embedding" : embed_text([chunk])[0],
                            "id": chunk_id,
                            "metadata": {"id": str(ix), "chunk_number": chunk_number, "filename" : os.path.split(filepath)[-1]},
                        }

                        # dump to a NDJSON file
                        out_f.write(json.dumps(doc)+"\n")

                        docs.append(doc)
        print(f"Documents written to the pas {output_path}")

    if len(docs) >0 and chromadb_path and os.path.isdir(chromadb_path):
        collection = create_collection(chromadb_path, collection_name)
        collection.add(
            ids=[doc["id"] for doc in docs],
            embeddings =[doc["embedding"] for doc in docs],
            documents = [doc["page_content"] for doc in docs],
            metadatas = [doc["metadata"] for doc in docs],
        )
        print(f"{len(doc)} embeddings are added to collection {collection_name} of the chroma db on the path {chromadb_path}")
    else:
        print(f"Nothing to load to path {chromadb_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--collection-name", type=str, required=False)
    parser.add_argument("--chromadb-path", type=str, required=False)

    args = parser.parse_args()
    
    load_files_to_chromadb_collection(args.input_dir, args.chromadb_path, args.collection_name)

    print("Done.")

if __name__ == "__main__":
    main()