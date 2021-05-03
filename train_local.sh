BUCKET_NAME="battulga-models"
REGION="us-central1"
JOB_NAME="aquarium28"
JOB_DIR="gs://$BUCKET_NAME/models"

gcloud ai-platform local train \
    --package-path trainer \
    --module-name trainer.task \
    --job-dir $JOB_DIR