# Flatiron Final Project - stock sentiment

This project explored whether it was possible to build prediction models for stock prices based on scraped text data - namely, articles from the New York Times and Twitter text. In particular, this text data was analyzed using the VADER sentiment analysis to derive positive/negative sentiment scores for every day (or, in the case of NYT articles, every day there was a relevant article). These scores were the features upon which I tried to predict movements in stock prices for Tesla, Netflix and Facebook.

The results of these models were, at best, middling (or, in the case of Facebook, simply not good). Yet, given that stock price movements are famously random, middling models are really not that bad: you could have made money with the Netflix and Tesla models.

Consider the performance of the Tesla model, illustrated below. Tesla stock dropped by more than 25% if the first half of 2019, presenting something of a challenge for a model seeking to make purchase type trades (perhaps I should try another model that can make shorts...). The Tesla model, rather smartly, doesn't make many trades over the period, and manages to come out out ahead: with $1,000, you would have made nearly $40, while a $1,000 buy and hold strategy would have lost you $260.

![Tesla model performance](https://github.com/max-miller/stock_sentiment_analyzer/blob/master/images/Tesla_performance.png?raw=true)

Red lines in the chart represent dates the model suggests taking a position. Some of those were good trades and some not, but it manages to break even despite the stock price suffering over the period.


# Data gathering
As with every data science project, data gathering and cleaning represented the largest challenges. Notebooks detailing the logic and basic outline of my data gathering code can be found in the appropriately titled 'data_gathering' folder. While the notebooks illustrate how I did things, I typically did the actual scraping from uncommented python files at the command line, which I have not uploaded here. I have also not uploaded any of the raw data, since I don't want to upload the text of thousands of NYT articles.

New York Times: scraping the NYT articles turned out to be a multi step process because the NYT api would only return the article's title and first paragraph, not the complete text. I ended up using the API to compile a list of urls and then scraping the text separately. A challenge with working with the NYT API/articles was filtering for topic. If you searched the API for a term like 'Tesla' it would return every article that had the word 'Tesla' somewhere in the text. This resulted in a lot of articles that weren't really relevant being returned: a real estate article where someone says they're looking for a place with a garage to park their Tesla, for instance.

Data gathering folder: NYT scraping notebook, quick stock scraping notebook, twitter scraping notebook, final processing notebook.



NYT_exploration notebook: quick text selection, sentiment analysis and visual exploration for text scraped from the New York Times.

sentiment_based_predictions: the bulk of the model work, with predictive models for Netflix, Tesla and Facebook that have... varying degrees of success.
