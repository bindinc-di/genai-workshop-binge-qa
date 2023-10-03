PROJECT_ID=speeltuin-327308
REGION=europe-west1
APP_NAME=binge-qa-api
docker buildx build . --platform linux/amd64 --file Dockerfile.api --tag $APP_NAME
# run locally
# docker run -it --rm -p 5000 $APP_NAME 
docker tag $APP_NAME gcr.io/$PROJECT_ID/$APP_NAME
docker push gcr.io/$PROJECT_ID/$APP_NAME
gcloud config set project $PROJECT_ID
gcloud run deploy $APP_NAME --image gcr.io/$PROJECT_ID/$APP_NAME --platform managed --region $REGION --allow-unauthenticated --timeout=900
