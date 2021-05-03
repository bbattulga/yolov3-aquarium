from zipfile import ZipFile
from google.cloud import storage
import os
import logging


"""
    download dataset
"""
bucket_name = 'battulga-datasets'

source_blob_name = 'aquarium_dataset.zip'

destination = source_blob_name

# download
storage_client = storage.Client()

bucket = storage_client.get_bucket(bucket_name)

blob = bucket.blob(source_blob_name)

blob.download_to_filename(destination)

# extract
zf = ZipFile(destination)
zf.extractall()
zf.close()

"""
    download weights
"""
bucket_name = 'battulga-models'

source_blob_name = 'weights/yolo_weights.h5'
destination = 'yolo_weights.h5'

# download
storage_client = storage.Client()

bucket = storage_client.get_bucket(bucket_name)

blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination)
logging.warning('LISTDIR from download.py')
logging.warning(str(os.listdir()))
