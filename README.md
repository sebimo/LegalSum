## Paper
This is the code used for the [Summarization of German Court Rulings](https://aclanthology.org/2021.nllp-1.19.pdf) paper.

The dataset will be made available shortly.

# LegalSum
Codebase for the summarization of German court rulings. This includes:

* data scraping & preprocessing pipeline
* GUI for manual data validation
* custom data loading & preprocessing
* model implementation & training routines
* evaluation routines

## Dataset
Around 100.000 guiding principles (legal summarizations) from German courts with a total number of with 300k-400k summarization sentences.

## Usage

Install the requirements with the ```environment.yml``` into a conda enviroment. 

* entry point for training is ```main.py```
* method evaluation is done with ```oracle.py``` (for the evaluation of the extractive labels) and ```evaluate.py``` for all the other methods

## Directory Structure

* ```src/``` contains all the preprocessing, dataloading, training and evaluation code for the extractive and abstractive summarization methods
* ```data/``` contains all the code for the acquisition of the dataset (scraping, processing, validation,...)

Not included in this repository is the dataset, which should be copied to ```data/dataset``` and some other binary files (information about them can be found within their folders).
