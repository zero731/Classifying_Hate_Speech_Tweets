 
    
    

def check_unique(df, col, dropna=False):
    
    """Takes in a Pandas DataFrame and specific column name and returns a Pandas DataFrame 
    displaying the unique values in that column as well as the count of each unique value. 
    Default is to also provide a count of NaN values.
    """
    
    import pandas as pd
    
    unique_vals = pd.DataFrame()
    unique_vals['count'] = pd.Series(df[col].value_counts(dropna=dropna))
    unique_vals['%'] = pd.Series(df[col].value_counts(normalize=True, dropna=dropna))
    
    return unique_vals






def clean_tweet(tweet, stop_list):
    """Takes in a tweet in the form of a string and cleans it of tags, urls,
       '&amp;' (which denotes '&'), and 'RT's. Returns a list of tokens with everything lower case
       and any words and punctuation from the specified stop list removed."""
    
    import nltk
    from nltk import regexp_tokenize
    import re
    
    
    # Remove twitter tags (@s)
    tweet = re.sub(r'@([_0-9a-zA-Z])\w+', ' ', tweet)
    
    # Remove urls
    tweet = re.sub(r'http\S+', ' ', tweet)
    
    # Remove all ''&#0000's
    tweet = re.sub(r'(&#[0-9]+)', ' ', tweet)
    
    # Remove all '&xt's
    tweet = re.sub(r'(&[a-z]t)', ' ', tweet)
    
    # Replace all '&amp;' with 'and'
    tweet = re.sub(r'(&amp;)', 'and', tweet)
    
    # Remove all 'RT's
    tweet = re.sub(r'(RT)', ' ', tweet)
    
    # Remove all versions of ellipses that string together separate words
    tweet = re.sub(r'(\.\.+)', ' ', tweet)
    
    # Make everything lower case
    tweet = tweet.lower()
    
    # Split into tokens - this tokenizer
    pattern = r"([a-zA-Z]+(?:'[a-z]+)?)"
    tokens = regexp_tokenize(tweet, pattern)
    
    # Remove stopwords and punctuation
    stopped_tokens = [w for w in tokens if w not in stop_list]
    
    return stopped_tokens







def get_token_list(df, col, freq=False):
    """Takes in a DataFrame and column that contains tokenized texts 
       and returns a list containing all the tokens (including duplicates) from that 
       column. If show_freq=True, the function will also print out the number of 
       unique tokens and the top 25 most common words as well as their counts based
       on nltk's FreqDist."""
    
    import pandas as pd
    from nltk import FreqDist
    
    ## Create list of all tokens
    tokens = []
    for text in df[col].to_list():
        tokens.extend(text)

    if freq:
    # Make a FreqDist from token list
        fd = FreqDist(tokens)
    
    # Display length of the FreqDist (# of unique tokens) and 25 most common words
        print('\n********** {} Summary **********\n'.format(col))
        print('Number of unique words = {}'.format(len(fd)))
        display(pd.DataFrame(fd.most_common(25), columns=['token', 'count']))
    
    return tokens






def plot_wordcloud(tokens, title=None):
    """Takes in a list of tokens and returns a wordcloud visualization of the most common words."""
    
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    wordcloud = WordCloud(collocations=False, colormap='gist_rainbow',
                          min_font_size=7)
    wordcloud.generate(','.join(tokens))
    
    plt.figure(figsize = (14, 14))
    if title:
        plt.title(('Most Common Words for ' + title), fontsize=28, fontweight='bold')
    plt.imshow(wordcloud) 
    plt.axis('off');
    
    return wordcloud