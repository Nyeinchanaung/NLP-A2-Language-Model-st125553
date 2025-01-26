# NLP-A2-Language-Model-st125553
### Assignment 2: Language Model
### AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)

## GitHubLink:
-  https://github.com/Nyeinchanaung/NLP-A2-Language-Model-st125553 

## Content
- [Student Information](#student-information)
- [Files Structure](#files-structure)
- [How to run](#how-to-run)
- [Dataset](#dataset)
- [Model Training](#training)
- [Web Application](#application)

## Student Information
 - Name     : Nyein Chan Aung
 - ID       : st125553
 - Program  : DSAI

## Files Structure
1) The Jupytor notebook files
- st125553-LSTM-LM.ipynb
2) `app` folder  
- app.py (streamlit)
- `models` folder which contains four model exports and their metadata files.
 
## How to run
 - Clone the repo
 - Open the project
 - Open the `app` folder
 - `streamlit run app.py`
 - app should be up and run on `http://localhost:8501/`

## Dataset
- I am using the `Poetry Foundation Poems dataset` from kaggle.
- The datase was the collection of over 13,000 Poems from https://www.poetryfoundation.org.
- The dataset include Title, Poem (body), Poet and Tag. I used Poem(body) to train the model.
- After cleaning the data, there are 91,020 rows of data.
- I splited the data into 72816 rows for training, 9102 rows for each testing and validing in `DatasetDict`
- Kaggle Link: https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems.
- Credict To : https://www.poetryfoundation.org

## Training
- 
## Application
- 
## Screenshot
![Webapp1](ss1.png)
