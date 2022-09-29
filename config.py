import os

stage = os.environ.get("IR2STAGE", "PROD")

passage_size = 400
tokenization_method = "spaces"
prepend_title_to_passage = True

# True to disable caching to a file
keep_in_memory = True if stage == "DEV" else False

ranking_size = 100
