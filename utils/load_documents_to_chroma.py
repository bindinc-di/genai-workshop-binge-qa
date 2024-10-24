from argparse import ArgumentParser
import json
import os

import chromadb
DEFAULT_CHROMADB_PATH = "data/chromadb"

def remove_empty(doc: dict) -> dict:
    for k in list(doc.keys()):
        if not doc[k]:
            del doc[k]
    return doc

def create_collection(chromadb_path, collection_name):
    chroma_client = chromadb.PersistentClient(path=chromadb_path)
    for collection in chroma_client.list_collections():
        if collection.name == collection_name:
            chroma_client.delete_collection(name=collection_name)
            # collection = chroma_client.get_collection(name=collection_name)
            # collection.delete()
            # break

    collection = chroma_client.create_collection(name=collection_name)
    return collection


def reload_collection(input_filepath, chromadb_path, collection_name):
    
    collection = create_collection(chromadb_path, collection_name)

    with open(input_filepath, encoding="utf-8") as f:
        for ix, r in enumerate(f):
            doc = json.loads(r)
            # print(doc)
            # print(doc)
            if 'embedding' or 'page_content' not in doc:
                print(f"No embedding or page_content in line {ix}: {r}")
                continue

            collection.add(embeddings=[doc['embedding']], 
                           documents=[doc['page_content']],
                           metadatas=[remove_empty(doc.get('metadata', {}))],
                           ids=[str(doc.get('id') or doc.get('metadata', {}).get('id', ix))])

def test_search(chromadb_path, collection_name):
    chroma_client = chromadb.PersistentClient(path=chromadb_path)
    collection = chroma_client.get_collection(name=collection_name)

    # get some document by id
    my_id = 20992
    res = collection.get([str(my_id)], include=['embeddings', 'metadatas', 'documents'])
    emb = res['embeddings'][0]
    doc = res['documents'][0]
    print("Document with id :{}".format(my_id))
    print(doc)
    print("Embedding: {}".format(emb))

    # search similar documents
    similar_items = collection.query(emb, n_results=3)
    for id, doc in zip(similar_items['ids'][0], similar_items['documents'][0]):
        if int(id) == my_id:
            continue
        print(id + ":\t" + doc + "\n\n")


def load_file_to_chromadb_collection(input_filepath, chromadb_path, collection_name):
    # if os.path.isdir(input_filepath):
    #     doc_filepath = os.path.join(input_filepath, "docs.ndjson")
    #     for filename in os.listdir(input_filepath):
    #         if filename.endswith(".txt"):
    #             filepath = os.path.join(input_filepath, filename)
    #             with open(filepath, mode="r", encoding="utf-8") as in_f, \
    #                  open(doc_filepath, mode="r", encoding="utf-8") as out_f:
    #                 doc = {
    #                     "page_content": in_f.read(),
    #                     "id": filename,
    #                     "metadata": {}
    #                 }
    #                 out_f.write(json.dumps(doc) + "\n")
    #     reload_collection(doc_filepath, chromadb_path, collection_name=collection_name)


    # el
    if os.path.isfile(input_filepath) and input_filepath.endswith(".ndjson"):
        reload_collection(input_filepath, chromadb_path, collection_name=collection_name)
        test_search(chromadb_path, collection_name=collection_name)
    else:
        raise ValueError(f"Invalid input filepath: {input_filepath}")



def main():
    parser = ArgumentParser()
    parser.add_argument("input_filepath", type=str)
    parser.add_argument("--collection-name", type=str)
    parser.add_argument("--chromadb-path", type=str, default=DEFAULT_CHROMADB_PATH)

    args = parser.parse_args()
    
    load_file_to_chromadb_collection(args.input_filepath, args.chromadb_path, args.collection_name)

    print("Done.")

if __name__ == "__main__":
    main()