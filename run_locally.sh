# Run semantic search API
# run Api server on http://127.0.0.1:5000
cd api_app
uvicorn app:app --reload

# Run the app
# adjust env SEARCH_BASE_URL=http://127.0.0.1:5000
streamlit run chat_app/app.py --server.headless true