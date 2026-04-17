# 📰 DJIA News Sentiment Analysis

> **A deep learning-based Natural Language Processing (NLP) model designed to classify financial news sentiment and capture market signals.**

---

## 📖 Problem Background
Stock prices are influenced by a multitude of factors, with public sentiment being one of the most crucial. To maximize returns in the stock market, it is essential to understand the information shaping traders' decisions. One of the most influential sources of this information is financial news, making it vital to catch market signals immediately as they are published.

## 🎯 Project Output
This project is developed to help traders quickly process and understand the vast amount of news published every minute. The model classifies news articles based on their underlying sentiment into three distinct categories:
* 🟢 **Positive**
* 🔴 **Negative**
* ⚪ **Neutral**

## 🗂️ Repository Outline

    📦 DJIA-News-Sentiment-Analysis
     ┣ 📜 README.md                  # Project Overview
     ┣ 📓 notebook.ipynb             # Notebook for building the model
     ┣ 📓 inference_notebook.ipynb   # Notebook for inferencing/predicting new data
     ┣ 📊 djia_news.csv              # Dataset for training the model
     ┣ 📊 new_data.csv               # New data which will be predicted
     ┗ 📂 deployment                 # Folder containing Python script for deployment

## 📊 Data
The dataset was extracted using the **Finnhub API**, covering a 90-day period and encompassing both news headlines and summaries.

## ⚙️ Methodology
This project implements the **Recurrent Neural Network (RNN)** algorithm. The modeling approach includes evaluating the network both with and without transfer learning, alongside other architectural improvements to achieve optimal performance.

## 💻 Tech Stack
* **Language:** Python 🐍
* **Tools:** Visual Studio Code

## 🚀 Reference & Resources
Access the trained model and view the project presentation below:

- 🔗 **[Trained Model on Google Drive](https://drive.google.com/file/d/1IbvP8YsYWCVUiCknzNyXxAajjjaFfnMM/view?usp=sharing)**  
- 📊 **[Project Presentation on Canva](https://www.canva.com/design/DAHBADencxA/3kFDskseL-g7FHfgjk7s5g/edit?utm_content=DAHBADencxA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)**

---
