import os

stage = os.environ.get("IR2STAGE", "DEV")

passage_size = 400
tokenization_method = "spaces"
prepend_title_to_passage = True
