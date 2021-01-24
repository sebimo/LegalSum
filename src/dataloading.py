# Contains all the dataloading tasks for the training procedures
# First loading & processing of verdicts will be replaced by a version in a more bare-metal language,
# as those tasks can be easily parallelized.
# Preprocessing tasks will be loaded from preprocessing.py, as we are here concerned about data plumbing!
# Functionality:
#   - load json verdict, extract relevant text portions, tokenize & preprocess text
#      * space-based tokenization
#      * byte-pair tokenization
#  (- fetch relevant norm texts for a verdict, tokenize & preprocess text)
#   - dataloader that can be used in the model training

# TODO fix train, val, test split