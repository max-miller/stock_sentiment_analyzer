# Flatiron Final Project - stock sentiment

This project explored whether it was possible to build prediction models for stock prices based on scraped text data - namely, articles from the New York Times and Twitter text. In particular, this text data was analyzed using the VADER sentiment analysis to derive positive/negative sentiment scores for every day (or, in the case of NYT articles, every day there was a relevant article). These scores were the features upon which I tried to predict movements in stock prices for Tesla, Netflix and Facebook.

The results of these models were, at best, middling (or, in the case of Facebook, simply not good). Yet, given that stock price movements are famously random, middling models are really not that bad: you could have made money with the Netflix and Tesla models.

Consider the performance of the Tesla model, illustrated below. Tesla stock dropped by more than 25% if the first half of 2019, presenting something of a challenge for a model seeking to make purchase type trades (perhaps I should try another model that can make shorts...). The Tesla model, rather smartly, doesn't make many trades over the period, and manages to come out out ahead: with $1,000, you would have made nearly $40, while a $1,000 buy and hold strategy would have lost you $260.

![Tesla model performance](https://github.com/max-miller/stock_sentiment_analyzer/blob/master/images/Tesla_performance.png?raw=true)

Red lines in the chart represent dates the model suggests taking a position. Some of those were good trades and some not, but it manages to break even despite the stock price suffering over the period.


# Data gathering
As with every data science project, data gathering and cleaning represented the largest challenges. Notebooks detailing the logic and basic outline of my data gathering code can be found in the appropriately titled 'data_gathering' folder. While the notebooks illustrate how I did things, I typically did the actual scraping from uncommented python files at the command line, which I have not uploaded here. I have also not uploaded any of the raw data, since I don't want to upload the text of thousands of NYT articles.

New York Times: scraping the NYT articles turned out to be a multi step process because the NYT api would only return the article's title and first paragraph, not the complete text. I ended up using the API to compile a list of urls and then scraping the text separately. A challenge with working with the NYT API/articles was filtering for topic. If you searched the API for a term like 'Tesla' it would return every article that had the word 'Tesla' somewhere in the text. This resulted in a lot of articles that weren't really relevant being returned: a real estate article where someone says they're looking for a place with a garage to park their Tesla, for instance.

I handled the really inclusive search in two ways. One was restricting which urls got added to my list of articles to scrape. If, for instance, the name I was searching for did not appear in the title, abstract or first paragraph, I did not scrape the full article. This helped filter out articles that weren't really relevant but where the company name I was searching for happened to appear once late in the text. I also filtered out articles where the company didn't appear a sufficient number of times. For easily scaled fixes, these worked pretty well, though not perfectly. One of the reasons I suspect my models for Facebook's stock didn't work as well was the difficulty of getting clean news sentiment; every other article these days mentions something that someone said on Facebook, so selecting out only the articles that are actually about Facebook is difficult.

Twitter scraping turned out to be a much harder task. The Twitter API is restrictive and also requires purchasing premium access if you plan to make more than 50 requests a month. A classmate suggested using Twint as an alternative. Twint looks like a great tool, but my use of it was marred by technical difficulties. In the end, some of the functionality seemed to not work on my machine. In particular, the function of Twint that is supposed to save down search results into a CSV, text file or Pandas dataframe, doesn't seem to work on my machine. My Twitter scraping notebook consists mostly of outlining my work-around to this, by using the system functionality to track the printed output of the python file, saving that output as a txt file and then reading the text back up to clean and insert into a dataframe.

# VADER sentiment analysis
I compiled sentiment scores based on the derived text using the VADER sentiment analyzer. VADER was particularly useful here because it's pretty robust to things like slang, misspellings or alternate spellings and the sorts of grammar/punctuation usage that typify language on things like social media. It's lexicon even includes non-word tokens like emoticons (the frowning emoticon :( has a sentiment rating of -2.2 on a -4 to +4 scale). This means it's pretty good at handling the text derived from Twitter, which is inherently pretty messy.

There were still some curious results from the sentiment analysis of the different sources of text. Consider the distribution sentiment scores derived from NYT articles about Tesla for a single year, 2018:

![Tesla NYT derived sentiment](https://github.com/max-miller/stock_sentiment_analyzer/blob/master/images/tesla_2018_sentiment.png?raw=true)

The analyzer typically is pretty sure whether an article is expressing positive or negative sentiment, with most of the scores clustered around positive or negative one (the scale goes from -1 to 1). In 2017, however, there were apparently many more ambiguous articles:

![Tesla NYT derived sentiment](https://github.com/max-miller/stock_sentiment_analyzer/blob/master/images/tesla_2017_sentiment.png?raw=true)

The spike around 1 is still visible, but there were many more articles that were, maybe, slightly negative, around -.5 or -.6, making the distribution plot look a lot less clean.

Another curious thing is that sentiment scores for Twitter text were almost always invariably positive. It's easy to find individual tweets that were negative, but compiled into a master text, the compound sentiment score for all of the tweets was basically always positive. In my final models, I ended up using the sub-scores for fractions of words that had positive and negative sentiment in addition to the compound score, because those scores provided much more variance.

# Model Building
Day to day stock performance is noisy - indeed famously random. Visual inspection doesn't suggest that there is a whole lot of value to be gleaned from the sentiment scores derived from news and twitter text. Consider a simple graph of rolling sentiment versus stock price:
![Tesla NYT derived sentiment vs stock close price](https://github.com/max-miller/stock_sentiment_analyzer/blob/master/images/stock_sentiment_vs_close.png?raw=true)


This looks mostly noisy and unhelpful, but this graph is a little misleading. Over the three and a half years of this data frame, there was a clear bounce following positive sentiment. On average the daily stock movement was low, but positive - around .04%. This should make some sense. Over the entire time period the stock price did rise, but with that rise spread out over so much time, the average day to day movement is pretty low.

Following a high sentiment value in the NYT, however, we see day to day movement of closer a bit more than .2% on average and following low sentiment in the NYT we see a drop of a bit more than .4% on average.

It ought to be said that these are small differences and that the averages are masking substantial variance within the groups - there are definitely instances of low sentiment days being followed by price increases and vice versa. Visually suggestive as these numbers are, the differences between these two groups is not quite statistically significant. Still, it looks like there might be some valuable information to be gleaned.

Before discussing the performance of the models, one last, very important note about avoiding reverse causality. It's not actually enough to find a connection between sentiment and stock performance - it needs to be actionable, and in order for that to be the case you need to avoid all instances where it might be the stock influencing the text rather than the other way around. If the stock is having a good day for instance, you might see a bunch of Tweets saying things like "wow, the stock is doing great today!" or maybe a late update story published on a news site that evening saying that the stock performed well that day. Of course, you can't actually use the reactions to the performance to predict the performance itself. To avoid any risk of that, I've avoided all intra-day comparisons. The text sentiment is calculated for an entire day and then used it to predict performance the *next* day.

With an eye towards how to actually operationalize a model with a trading strategy, I chose to create classification models - something like "will the next day see above or below average performance". You could then choose to trade on that information in the next day.

Something else I noticed is that when you consider the sentiment coming from different sources, there seemed to be some sort of non-linear/interaction effect. Between days with high and low negativity scores from Twitter, there is a much bigger difference in price movement when the NYT sentiment was low than when it was high. This suggests some sort of interaction here and that a logistic regression may not be the most appropriate approach. I ended up trying various tree, KNN and random forest models and had the most success with the random forest models.

I trained my model on the historical data from 2016 to the end of 2018 and then tested their performance on the first half of 2019. My best performing Tesla models worked pretty well, breaking even or even making a little money while the stock was down more than 25% over the period. It's important to note that these models weren't successful in the sense that they correctly identified every day the stock went up, or that they never accidentally suggested you trade on a day that the stock actually fell. In an environment where the stock mostly fell, these models succeeded mostly by advising you not to trade at all and getting it right a bit more often than not when it did suggest you trade.

![Tesla model performance](https://github.com/max-miller/stock_sentiment_analyzer/blob/master/images/Tesla_performance.png?raw=true)

My best Netflix models also modestly outperformed the simple buy and hold strategy, but in marked contrast to the Tesla model, it succeeded by taking a lot of positions - indeed taking positions most days! For a stock that is generally increasing in value over the period, this makes a certain amount of sense! The Netflix models won their modest edge over the simple strategy by managing to avoid a couple of significant down stretches in an otherwise positive growth environment:

![Netflix model performance](https://github.com/max-miller/stock_sentiment_analyzer/blob/master/images/Netflix%20performance.png?raw=true)

My Tesla model clearly outperformed the simple buy and hold strategy. My Netflix modestly outperformed the buy and hold strategy, but the difference between the two performances is small enough that it might easily be attributable to chance rather than to the strength of the model. (Still, matching market returns is not a horrible result, given that efficient markets theory suggests that the task should not be possible at all.) By contrast my Facebook models were unambiguous failures, netting far less than the simple buy and hold strategy:

![Facebook model performance](https://github.com/max-miller/stock_sentiment_analyzer/blob/master/images/facebook_performance.png?raw=true )

Two possible reasons this might be the case. 1) Facebook is a more financially mature company. Where in the case of Tesla, stock performance is generally untethered from traditional metrics of financial performance (they've never really 'turned a profit' exactly), Facebook stock performance is easier to align with its balance sheet. 2) Actually filtering for only articles relevant to Facebook was substantially harder, as I discussed. The facebook data may be much noisier than the Tesla or Netflix data.

# Next Steps
Where would you go from here? Obvious next steps include:
- Better text-cleaning practice: particularly for companies like Facebook or Twitter which will see a lot of mentions in articles that aren't strictly about them.
- More text sources: both to help fill out the data footprint of companies that don't appear in the news very often and also to incorporate more distinct sources of data (it's reasonable to think that the NYT and Twitter represent two different sorts of data with different pieces of information and biases. The NYT and AP or Reuters might be highly correlated, however)
- A multi-class predictive model which tries to call out days to short the stock in addition to days to take long positions
- Try to incorporate timestamps, or other markers to incorporate intra-day estimates/performance and suggestions for same day positions to take.
