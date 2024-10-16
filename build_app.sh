PROJECT_ID=speeltuin-327308
REGION=europe-west1
APP_NAME=binge-qa-app
docker buildx build . --platform linux/amd64 --file Dockerfile.app --tag gcr.io/$PROJECT_ID/$APP_NAME
# run locally
# docker run -it --rm  --env-file ".env" -p 8501:8501 gcr.io/$PROJECT_ID/$APP_NAME
SEARCH_APP_URL=https://binge-qa-api-312076928311.europe-west1.run.app
SEARCH_API_KEY=PekwbnJFh0ohMzNYftAOCIXJhoViDKqF

PORT=8501 && docker run \
-it --rm \
-p 8501:${PORT} \
-e PORT=${PORT} \
-e K_SERVICE=dev \
-e K_CONFIGURATION=dev \
-e K_REVISION=dev-00001 \
-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/gcp_secret.json \
-v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/keys/gcp_secret.json:ro \
-e SEARCH_BASE_URL=$SEARCH_APP_URL \
-e SEARCH_API_KEY=$SEARCH_API_KEY \
gcr.io/$PROJECT_ID/$APP_NAME

# docker tag $APP_NAME gcr.io/$PROJECT_ID/$APP_NAME
docker push gcr.io/$PROJECT_ID/$APP_NAME
gcloud config set project $PROJECT_ID
gcloud run deploy $APP_NAME --image gcr.io/$PROJECT_ID/$APP_NAME --platform managed --region $REGION --project $PROJECT_ID \
    --allow-unauthenticated --timeout=120 --port=8501 \
    --set-env-vars "SEARCH_BASE_URL=${SEARCH_APP_URL}"

# the variable SEARCH_API_KEY has to be set manually in the running service
# TODO use googlecloud secrets
