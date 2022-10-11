import os

stage = os.environ.get("IR2STAGE", "DEV")

passage_size = 400
passage_size_longformer = 4050
tokenization_method = "spaces"
prepend_title_to_passage = True

# True to disable caching to a file
keep_in_memory = True if stage == "DEV" else False

ranking_size = 100
