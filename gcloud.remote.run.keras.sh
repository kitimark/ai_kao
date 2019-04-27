gcloud ai-platform jobs submit training KAO$(Date +%y%m%d%H%M%S)\
--module-name=trainer.kao_bow \
--package-path=./trainer --job-dir=gs://ai-kao \
--region=asia-east1 --config=./cloudml.yaml --runtime-version=1.10