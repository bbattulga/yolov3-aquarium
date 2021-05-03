import tarfile

with tarfile.open('./dist/trainer-0.4.tar.gz') as tf:
    tf.extractall()
