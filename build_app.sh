PROJECT_ID=speeltuin-327308
REGION=europe-west1
APP_NAME=binge-qa-app
docker buildx build . --platform linux/amd64 --file Dockerfile.app --tag $APP_NAME
# run locally
# docker run -it --rm  --env-file ".env" -p 5000 $APP_NAME
docker tag $APP_NAME gcr.io/$PROJECT_ID/$APP_NAME
docker push gcr.io/$PROJECT_ID/$APP_NAME
gcloud config set project $PROJECT_ID
gcloud run deploy $APP_NAME --image gcr.io/$PROJECT_ID/$APP_NAME --platform managed --region $REGION --allow-unauthenticated --timeout=120 --set-env-vars "SEARCH_BASE_URL=https://binge-qa-api-iobao7k2yq-ew.a.run.app"
