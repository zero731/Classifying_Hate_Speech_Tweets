 
    
    

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






def plot_class_distr(df, descr='', save=False, fig_name=None):
    
    """Takes in a Pandas DataFrame and optionally a description of the DataFrame for modifying
       the figure title. Plots the distribution of class labels from the DataFrame. Option to save
       the resulting image to Figures folder of current notebook.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    fig_filepath = 'Figures/'
    
    plt.figure(figsize=(7,5))
    df.groupby('class').tweet.count().plot.bar(ylim=0)
    plt.xlabel('Class', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.title('Distribution of Tweet Classes {}'.format(descr), fontsize=18, fontweight='bold')
    
    if save:
        plt.savefig(fig_filepath+fig_name, bbox_inches = "tight")
        
    plt.show()






def clean_tweet(tweet):
    """Takes in a tweet in the form of a string and cleans it of tags, urls,
       '&amp;' (which denotes '&'), and 'RT's. Returns the tweet with these removed."""
    
    import nltk
    import re
    
    
    # Remove twitter tags (@s)
    tweet = re.sub(r'@([_0-9a-zA-Z])\w+', ' ', tweet)
    
    # Remove urls
    tweet = re.sub(r'http\S+', ' ', tweet)
    
    # Remove all emojis encoded as '&#0000's
    tweet = re.sub(r'(&#[0-9]+)', ' ', tweet)
    
    # Remove all '&xt's
    tweet = re.sub(r'(&[a-z]t)', ' ', tweet)
    
    # Replace all '&amp;' with 'and'
    tweet = re.sub(r'(&amp;)', 'and', tweet)
    
    # Remove all 'RT's
    tweet = re.sub(r'(RT)', ' ', tweet)
    
    # Remove all versions of periods and ellipses that string together separate words
    tweet = re.sub(r'[a-zA-Z0-9]\.+[a-zA-Z0-9]', ' ', tweet)
    
    # Remove all '#'s
    tweet = re.sub(r'(#)', ' ', tweet)
    
    # Remove numeric symbols
    tweet = re.sub(r'[0-9]', ' ', tweet)
    
    return tweet







def tokenize_tweet(tweet, stop_list):
    """Takes in a tweet in the form of a string and cleans it of tags, urls,
       '&amp;' (which denotes '&'), and 'RT's. Returns a list of tokens with everything lower case
       and any words and punctuation from the specified stop list removed."""
    
    import nltk
    from nltk import regexp_tokenize
    import re
    
    
    # Clean the tweet
    tweet = clean_tweet(tweet)
    
    # Make everything lower case
    tweet = tweet.lower()
    
    # Split into tokens
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





def lemma_text(token_list):
    
    """Takes in a list of tokens and returns them joined as a lemmatized string using nltk.stem's
       WordNetLemmatizer.
    """
    
    from nltk.stem import WordNetLemmatizer 

    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    lemmatized_output = ' '.join([lemmatizer.lemmatize(word) for word in token_list])
    
    return lemmatized_output





def plot_wordcloud(tokens, title=None, save=False, fig_name=None):
    """Takes in a list of tokens and returns a wordcloud visualization of the most common words."""
    
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    fig_filepath = 'Figures/'
    
    wordcloud = WordCloud(collocations=False, colormap='gist_rainbow',
                          min_font_size=7)
    wordcloud.generate(','.join(tokens))
    
    plt.figure(figsize = (12, 12))
    if title:
        plt.title(('Most Common Words for ' + title), fontsize=28, fontweight='bold')
    plt.imshow(wordcloud) 
    plt.axis('off')
    
    if save:
        plt.savefig(fig_filepath+fig_name, bbox_inches = "tight")
    
    plt.show()
    
    return wordcloud






def eval_classifier(clf, X_test, y_test, model_descr='',
                    target_labels=['Hate Speech', 'Offensive', 'Neither'],
                    cmap='Blues', normalize='true', save=False, fig_name=None):
    
    """Given an sklearn classification model (already fit to training data), test features, and test labels,
       displays sklearn.metrics classification report and confusion matrix. A description of the model 
       can be provided to model_descr to customize the title of the classification report.
    """
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, plot_confusion_matrix
    
    
    fig_filepath = 'Figures/'
    
    ## get model predictions
    y_hat_test = clf.predict(X_test)
    
    
    ## Classification Report
    report_title = 'Classification Report: {}'.format(model_descr)
    divider = ('-----' * 11) + ('-' * (len(model_descr) - 31))
    report_table = classification_report(y_test, y_hat_test,
                                         target_names=target_labels)
    print(divider, report_title, divider, report_table, divider, divider, '\n', sep='\n')
    
    
    ## Make Subplots for Figures
    fig, axes = plt.subplots(figsize=(10,6))
    
    ## Confusion Matrix
    plot_confusion_matrix(clf, X_test, y_test, 
                          display_labels=target_labels, 
                          normalize=normalize, cmap=cmap, ax=axes)
    
    axes.set_title('Confusion Matrix:\n{}'.format(model_descr),
                   fontdict={'fontsize': 18,'fontweight': 'bold'})
    axes.set_xlabel(axes.get_xlabel(),
                       fontdict={'fontsize': 12,'fontweight': 'bold'})
    axes.set_ylabel(axes.get_ylabel(),
                       fontdict={'fontsize': 12,'fontweight': 'bold'})
    axes.set_xticklabels(axes.get_xticklabels(),
                       fontdict={'fontsize': 10,'fontweight': 'bold'})
    axes.set_yticklabels(axes.get_yticklabels(), 
                       fontdict={'fontsize': 10,'fontweight': 'bold'})
    
    
    if save:
        plt.savefig(fig_filepath+fig_name, bbox_inches = "tight")
    
    fig.tight_layout()
    plt.show()

    return fig, axes




def fit_grid_clf(model, params, X_train, y_train, X_test, y_test,
                 model_descr='', score='accuracy',
                 save=False, fig_name=None):
    
    """Given an sklearn classification model, hyperparameter grid, X and y training data, 
       and a GridSearchCV scoring metric (default is 'accuracy', which is the default metric for 
       GridSearchCV), fits a grid search of the specified parameters on the training data and 
       returns the grid object. Function also takes in X_test and y_test to get predictions and 
       evaluate model performance on test data. Prints out parameters of the best estimator as well 
       as its classification report and confusion matrix. A description of the model can be provided
       to model_descr to customize the title of the classification report.
    """
    
    from sklearn.model_selection import GridSearchCV
    import datetime as dt
    from tzlocal import get_localzone
    
    fig_filepath = 'Figures/'
    
    start = dt.datetime.now(tz=get_localzone())
    fmt= "%m/%d/%y - %T %p"
    
    print('---'*20)    
    print(f'***** Grid Search Started at {start.strftime(fmt)}')
    print('---'*20)
    print()
    
    grid = GridSearchCV(model, params, scoring=score, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    end = dt.datetime.now(tz=get_localzone())
    
    print(f'\n***** Training Completed at {end.strftime(fmt)}')
    print(f"\n***** Total Training Time: {end-start}")
    print('\n')
    
    print('Best Parameters:')
    print(grid.best_params_)
    print('\n')
    eval_classifier(grid.best_estimator_, X_test, y_test, model_descr,
                    save=save, fig_name=fig_name)
    
    if save:
        plt.savefig(fig_filepath+fig_name, bbox_inches = "tight")
   
    plt.show()
    
    return grid







def plot_feat_importance(model, clf_step_name, vec_step_name, model_title='', save=False, fig_name=None):
    
    """Takes in an sklearn classifier already fit to training data, the name of the step for that model
       in the modeling pipeline, the vectorizer step name, and optionally a title describing the model. 
       Returns a horizontal barplot showing the top 20 most important features in descending order.
    """

    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    import matplotlib.pyplot as plt
    
    fig_filepath = 'Figures/'
    
    feature_importances = (
        model.named_steps[clf_step_name].feature_importances_)
    
    feature_names = (
        model.named_steps[vec_step_name].vocabulary_) 
    
    importance = pd.Series(feature_importances, index=feature_names)
    plt.figure(figsize=(8,6))
    fig = importance.sort_values().tail(20).plot(kind='barh')
    fig.set_title('{} Feature Importances'.format(model_title), fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)
    
    if save:
        plt.savefig(fig_filepath+fig_name, bbox_inches = "tight")

    plt.show()
    
    return fig






def plot_coefficients(model, clf_step_name, vec_step_name,
                      class_label, model_title='', top_features=10,
                      save=False, fig_name=None):
    
    """Takes in an sklearn classifier already fit to training data, the name of the step for that model
       in the modeling pipeline, the vectorizer step name, a class label, and optionally a title describing the model. 
       Returns a horizontal barplot showing the top 20 most important features by coefficient weight (10 most 
       positive and 10 most negative).
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    fig_filepath = 'Figures/'
    
    ## Get the coefficients for the specified class label
    feature_coefs = (
        model.named_steps[clf_step_name].coef_[class_label])
    
    ## Get the vocabulary from the fit vectorizer
    feature_names = (
        model.named_steps[vec_step_name].vocabulary_) 
    # Create a version of the vocab dict with keys and values swapped
    vocab_swap = (
        {value:key for key, value in feature_names.items()}) 

    
    ## Store the top 10 positive coefficients and their indices
    pos_10_index = (
        np.argsort(model.named_steps[clf_step_name].coef_[class_label])[-top_features:])
    pos_10_coefs = (
        np.sort(model.named_steps[clf_step_name].coef_[class_label])[-top_features:])
    
    ## Store the top 10 negative coefficients and their indices
    neg_10_index = (
        np.argsort(model.named_steps[clf_step_name].coef_[class_label])[:top_features])
    neg_10_coefs = (
        np.sort(model.named_steps[clf_step_name].coef_[class_label])[:top_features])
    
    ## Combine top positive and negative into one list for indices and one for coefs
    top_20_index = list(pos_10_index) + list(neg_10_index)
    top_20_coefs = list(pos_10_coefs) + list(neg_10_coefs)

    
    ## Get list of top predictive words and use it as index for series of coef values
    top_words = []

    for i in top_20_index:
        top_words.append(vocab_swap[i])

    top_20 = pd.Series(top_20_coefs, index=top_words)
    
    
    ## Create plot
    plt.figure(figsize=(8,6))
    
    # Color code positive coefs blue and negative red
    colors = ['blue' if c < 0 else 'red' for c in top_20]
    
    # Adjust title according to specified class code
    class_dict = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    title_class = class_dict[class_label]
    
    fig = top_20.sort_values().plot(kind='barh', color=colors)
    fig.set_title('Top Words for Predicting {} - {}'.format(title_class, model_title),
                  fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)
    
    if save:
        plt.savefig(fig_filepath+fig_name+'_'+title_class.replace(' ', '_'), bbox_inches = "tight")
    
    plt.show()
    
    return fig






def print_full_tweet(df, col='text', title=''):
    
    """Takes in a dataframe and column name (default name is 'text') to print output
       displaying the total number of tweets and the full (non-truncated) text. Can 
       provide a title describing what sort of tweets are being output.
    """
    
    import pandas as pd
    
    print('***'*20, '\n')
    print(title, '\n')
    print('Number of tweets:', len(df), '\n')
    print('***'*20, '\n')
    
    for i in range(len(df)):
        print(i , df.iloc[i]['text'], '\n', '---'*20, '\n')