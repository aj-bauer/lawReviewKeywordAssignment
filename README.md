# MESA8414: Label Maker

*Contributors*: Avi Bauer (aj-bauer) & Sarah Her (sarah-her)

Final Project for MESA8414 (Boston College), Fall 2025

## SYNOPSIS

This project is focused on higher education, and addresses the issues of topical analysis and automated keyword assignment for scholarly articles. Our motivation comes from a real-world scenario at Avi’s workplace, where library workers with limited knowledge of legal terminology are frequently tasked with assigning keywords from a large controlled vocabulary to law review articles as part of the archiving process. The learning curve for these library workers is steep, and could be greatly assisted by a system that provides keyword suggestions based on article abstracts. The dataset used in this project consists of metadata from approximately 2,100 law review articles from Boston College-published law reviews (see *Dataset Note*), which includes both article abstracts and human-assigned keywords drawn from a set of 108 legal terms.

In the notebook */notebooks/MESA8414_LabelMaker_ModelBuilding.ipynb*, the full subject vocabulary is first collapsed into a higher-level 30 term vocabulary using Latent Dirichlet Allcoation (LDA). Then a series of multi-label classifiers are trained and tested, and a Support Vector Machine (SVM) classifier was selected for deployment based on Hamming Loss. Cross-validation was used to select kernel and C (regularization parameter).

The resulting classifier, which accepts law review article abstracts and returns a list of relevant keywords from the 30-term controlled subject vocabulary, is hosted on Render.com, at https://mesa8414-label-maker.onrender.com/

### DATASET NOTE

The journals surveyed are the Boston College Law Review, the Boston College Environmental Affairs Law Review, the Boston College International and Comparative Law Review, the Boston College Journal of Law & Social Justice, and the Boston College Third World Law Journal. The articles represented in this dataset span publication dates from Jan 1962 - June 2022. All of the data in this dataset is public and available from https://lira.bc.edu.

## FILE STRUCTURE

```
├── README.txt
├── requirements.txt
├── app.py
├── data/
    └── docs.csv  <- required to train vectorizer
├── frontend/
├── index.html
├── model
    └── label_maker.skops
└── notebooks
    └── MESA8414_LabelMaker_ModelBuilding.ipynb
```
