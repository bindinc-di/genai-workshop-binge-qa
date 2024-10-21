from argparse import ArgumentParser
import json
import chromadb
DEFAULT_CHROMADB_PATH = "data/chromadb"

def remove_empty(doc: dict) -> dict:
    for k in list(doc.keys()):
        if not doc[k]:
            del doc[k]
    return doc


def reload_collection(input_filepath, chromadb_path, collection_name):
    chroma_client = chromadb.PersistentClient(path=chromadb_path)
    for collection in chroma_client.list_collections():
        if collection.name == collection_name:
            chroma_client.delete_collection(name=collection_name)
            # collection = chroma_client.get_collection(name=collection_name)
            # collection.delete()
            # break

    collection = chroma_client.create_collection(name=collection_name)
    with open(input_filepath, encoding="utf-8") as f:
        for ix, r in enumerate(f):
            doc = json.loads(r)
            # print(doc)
            # print(doc)
            collection.add(embeddings=[doc['embedding']], documents=[doc['page_content']], metadatas=[remove_empty(doc['metadata'])], ids=[str(doc['metadata']['id'])])

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


def main(input_filepath, chromadb_path):
    reload_collection(input_filepath, chromadb_path, collection_name="popcorn")
    test_search(chromadb_path, collection_name="popcorn")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_filepath", type=str)
    parser.add_argument("--chromadb-path", type=str, default=DEFAULT_CHROMADB_PATH)

    args = parser.parse_args()
    
    main(args.input_filepath, args.chromadb_path)

    print("Done.")