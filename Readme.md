# A chat application prototyped with Xebia during the Gen AI workshor
Both the Chat app and the Similarity search API app are designed as services to be deployed in Cloud Run. See build/deployment scripts
`build_app.sh`
`build_api.sh`


## Demoapp
The repository also includes a lightweight wersion of a local chat app using the local Chroma service for RAG
The app can be used for experimenting with models, prompts, parameters and the context for RAG

### Installation
1. create and activate a new python virtual environment
2. Install dependencies
`pip install -m dev-requirements.txt`
3. Make a `data` subdirectory in the project directory. Copy Chroma database files from GCS to the path `data/chromadb`:
```
gcloud -m cp -r gs://bdc-binge-bot-recommendation-data/chromadb data/
```
This database contains documents made from Popcorn tips/articles combined with IMDB synopsis and plot of the relevant movies (one document per Popcorn item) embedded using the script `utils/embed_documents.py` For semantic search you should use the same embedding model with the same dimensionality

4. Create a `.env` file with the keys/credentials for the OpenAI API and Google Cloud
```
OPENAI_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=...
```
5. Additionally, put your text documents to be used in the generation in the path `data/custom_documents`.
These should be plain TXT files (one file per each logical document). These documents are ALWAYS sent to the prompt without filtering so don't put too much of these
6. Additionally, load plain text documents into the Chroma using the script `utils/load_plain_text_documents_to_chroma.py`
Please use a separate collecton `other` as the collection name because the script is cleaning the collection before reloading it, so all previous documents will be gone.

## Run
From the project directory start the Chat app using the command
```
python demoapp/chat_app.py
```
It will log the URL of the app's webserver in the console

Additionally you can experiment with the semantic search on the documents in Chroma using the Search app:
```
python demoapp/search_app.py
```

If you want to restore the Chroma database perform the steps from Installation step 3