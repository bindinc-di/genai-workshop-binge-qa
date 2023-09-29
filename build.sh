PROJECT_ID=speeltuin-327308
REGION=europe-west1
APP_NAME=binge-qa-api
docker buildx build . --platform linux/amd64 --tag $APP_NAME
docker tag $APP_NAME gcr.io/$PROJECT_ID/$APP_NAME
docker push gcr.io/$PROJECT_ID/$APP_NAME
gcloud run deploy $APP_NAME --image gcr.io/$PROJECT_ID/$APP_NAME --platform managed --region $REGION --allow-unauthenticated --timeout=900
