## Paper
This is the code used for the [Summarization of German Court Rulings](https://aclanthology.org/2021.nllp-1.19.pdf) paper.

The dataset is available via this [Dropbox link](https://www.dropbox.com/s/23mrviv5396rdl0/LegalSum.zip?dl=0). 

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

## Research

If you make use of this dataset in your research, we ask that you please cite our paper:

```
@inproceedings{glaser-etal-2021-summarization,
    title = "Summarization of {G}erman Court Rulings",
    author = "Glaser, Ingo  and
      Moser, Sebastian  and
      Matthes, Florian",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nllp-1.19",
    pages = "180--189",
    abstract = "Historically speaking, the German legal language is widely neglected in NLP research, especially in summarization systems, as most of them are based on English newspaper articles. In this paper, we propose the task of automatic summarization of German court rulings. Due to their complexity and length, it is of critical importance that legal practitioners can quickly identify the content of a verdict and thus be able to decide on the relevance for a given legal case. To tackle this problem, we introduce a new dataset consisting of 100k German judgments with short summaries. Our dataset has the highest compression ratio among the most common summarization datasets. German court rulings contain much structural information, so we create a pre-processing pipeline tailored explicitly to the German legal domain. Additionally, we implement multiple extractive as well as abstractive summarization systems and build a wide variety of baseline models. Our best model achieves a ROUGE-1 score of 30.50. Therefore with this work, we are laying the crucial groundwork for further research on German summarization systems.",
}
```
