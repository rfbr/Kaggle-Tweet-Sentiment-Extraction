import os

from tokenizers import ByteLevelBPETokenizer

MAX_LEN = 70
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 4
ROBERTA_MODEL = os.path.abspath("../roberta-base/")
TRAINING_FILE = os.path.abspath("../../data/train_fold.csv")
TEST_FILE = os.path.abspath("../../data/test.csv")
SAVED_MODEL_PATH = os.path.abspath("../saved_models")
SAMPLE_SUBMISSION_FILE = os.path.abspath("../../data/sample_submission.csv")
TOKENIZER = ByteLevelBPETokenizer(vocab_file=f"{ROBERTA_MODEL}/vocab.json",
                                  merges_file=f"{ROBERTA_MODEL}/merges.txt",
                                  lowercase=True,
                                  add_prefix_space=True)
