# Defending-Supervised-Classifiers-against-Property-Inference-Attacks

This project contains code for paper "Defending-Supervised-Classifiers-against-Property-Inference-Attacks".

### File Description

##### main.py:

Use *PPCT* to train a final classifier in an iterative fashion.

##### data_clean.py

Generate an original dataset who has a specified property.

##### shadow_dataset_generate.py 

1. create_shadow_data(): Samples input from the sample domain.
2. datafilter(): Normalize samples and generate correct labels for them by *pilot mode*l.

##### net.py

Include the architecture of the classifier.

##### train.py

Include the training function and testing function of the classifier.