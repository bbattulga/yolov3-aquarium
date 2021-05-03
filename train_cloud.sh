BUCKET_NAME="battulga-models"
REGION="us-central1"
JOB_NAME="aquarium38"
JOB_DIR="gs://$BUCKET_NAME/models"

gcloud ai-platform jobs submit training $JOB_NAME \
    --package-path trainer \
    --module-name trainer.task \
    --region $REGION \
    --python-version 3.5 \
    --runtime-version 1.14 \
    --job-dir $JOB_DIR \
    --scale-tier custom \
    --master-machine-type n1-standard-8 \
    --master-accelerator=type=nvidia-tesla-p4,count=1 \
    --stream-logs