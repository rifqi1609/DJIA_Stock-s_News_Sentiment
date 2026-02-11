# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Untuk menjalankan streamlit di eda
# "streamlit run eda.py"

def run():
    # Make title
    st.title("News Sentiment of DJIA Stock")
    st.image("deployment\image.jpg")
    st.markdown("## Background")
    st.markdown("""
    Stock price is influenced by many factors. One of the most crucial aspects is public sentiment of stocks. For gaining much money in the stock market, we need to understand the information spreading in trader's minds.
    One of the most influential source is stock news that we need to catch the signal immediately.       
    """)

    st.markdown("## Objective")
    st.markdown("""This pproject is developed for helping traders to understand a lot of news published every minutes. This project will classify news based on their sentiment namely, Positive, Negative and Neutral.
    """)

    # Data Preparation
    df = pd.read_csv('deployment/djia_news.csv')

    # Menampilkan Visualisasi EDA
    st.markdown("## Exploratory Data Analysis")

    # 1
    st.markdown("### 1. News Sentiment Proportion")
    # Count of each category
    counts = df['sentiment_label'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind='pie', 
                    autopct='%1.1f%%',
                    startangle=140)
    plt.title('News Sentiment Proportion')
    plt.ylabel('')
    st.pyplot(fig)
    st.markdown("""From the information above, we can see that "Neutral news is the most frequent published. If we want to see considerable move, it could happen when "Positive" and "Negative" news are published""")
    
    # 2
    st.markdown("### 2. Stock News Proportion")
    symbol_table = (df['symbol'].value_counts(normalize=True) * 100).reset_index()
    symbol_table.columns = ['Symbol', 'Percentage (%)']
    st.dataframe(symbol_table)
    st.markdown("""DJIA is the stock index of 30 blue chip stocks. As we can see that JPM, AMZN, AAPL and INTC is the the most frequent published above 5% amongst all stocks.
                """)

    # 3
    st.markdown("### 3. Top 10 Stocks with the most news")
    top_10=df['symbol'].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_10.plot(kind='bar', figsize=(10, 6), color='skyblue')
    plt.title('Top 10 Stocks with the most news')
    plt.xlabel('Ticker')
    plt.ylabel('Total of News')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
    st.markdown("""If we see the number of news, the top 4 mentioned before reach around 250 news within 30 days.
                """)

if __name__=='__main__':
    run()