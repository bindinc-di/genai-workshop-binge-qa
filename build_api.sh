# PROJECT_ID=chrome-unity-259608
PROJECT_ID=speeltuin-327308
REGION=europe-west1
APP_NAME=binge-qa-api
docker buildx build . --platform linux/amd64 --file Dockerfile.api --tag gcr.io/$PROJECT_ID/$APP_NAME


# Run container locally
PORT=8080 && docker run \
-it --rm \
-p 9090:${PORT} \
-e PORT=${PORT} \
-e K_SERVICE=dev \
-e K_CONFIGURATION=dev \
-e K_REVISION=dev-00001 \
-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/gcp_secret.json \
-v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/keys/gcp_secret.json:ro \
gcr.io/$PROJECT_ID/$APP_NAME
#gcr.io/speeltuin-327308/binge-qa-api:latest

# Test-call the API with the shell command:
#curl --header "Content-Type: application/json" --data '{"query": "Horror Series op Netflix"}' http://0.0.0.0:9090/similarity

# docker tag $APP_NAME gcr.io/$PROJECT_ID/$APP_NAME
docker push gcr.io/$PROJECT_ID/$APP_NAME
gcloud config set project $PROJECT_ID
# gcloud run deploy $APP_NAME --image gcr.io/$PROJECT_ID/$APP_NAME --platform managed --region $REGION --allow-unauthenticated --timeout=900
gcloud run deploy $APP_NAME --image gcr.io/$PROJECT_ID/$APP_NAME --platform managed --project $PROJECT_ID --region $REGION --timeout=900