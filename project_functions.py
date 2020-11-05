
def check_unique(df, col, dropna=False):
    
    """Takes in a Pandas DataFrame and specific column name and returns a Pandas DataFrame 
    displaying the unique values in that column as well as the count of each unique value. 
    Default is to also provide a count of NaN values.
    
    Args:
        df (Pandas DataFrame): DataFrame containing the column to check the unique values of.
        col (str): Name of the column you want to check.
        dropna (bool, default=False): Whether or not to drop null values from list of values.
    
    Returns:
        DataFrame: Pandas DataFrame with columns for the unique values in the specified column, 
            the number of occurrences of each unique value in that column, and the percentage of 
            the column made up by each unique value.
    
    Example:
        >>> df = pd.DataFrame({'a': [2, 4, 4, 6],
                               'b': [2, 1, 3, 4]})

        >>> check_unique(df, col='a', dropna=False)
        
            count   %
        4   2   0.50
        6   1   0.25
        2   1   0.25
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
    
    Args:
        df (Pandas DataFrame): DataFrame from which to plot class distributions.
        descr (str): A description of the DataFrame for customizing plot title.
        save (bool, default=False): Whether to save the returned figure.
        fig_name (str, optional): What to name the file if the image is being saved.
        
    Returns:
        figure: Matplotlib.pyplot bar plot figure showing the distribution of the data by class.
    
    Example:
        >>> plot_class_distr(df=data, descr='Example Dataset', save=True, fig_name='class_distr_example')
    
        """
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    fig_filepath = 'Figures/'
    
    fig = plt.figure(figsize=(7,5))
    df.groupby('class').tweet.count().plot.bar(ylim=0)
    fig = plt.xlabel('Class', fontsize=14, fontweight='bold')
    fig = plt.ylabel('Count', fontsize=14, fontweight='bold')
    fig = plt.xticks(rotation=45, fontsize=12, fontweight='bold')
    fig = plt.yticks(fontsize=12)
    fig = plt.title('Distribution of Tweet Classes {}'.format(descr), fontsize=18, fontweight='bold')
    
    if save:
        plt.savefig(fig_filepath+fig_name, bbox_inches = "tight")
        
    plt.show()

    return fig




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







def tokenize_tweet(tweet, stop_list, pattern = r"([a-zA-Z]+(?:'[a-z]+)?)"):
    """Takes in a tweet in the form of a string and cleans it of tags, urls,
       '&amp;' (which denotes '&'), and 'RT's. Returns a list of tokens with everything lower case
       and any words and punctuation from the specified stop list removed.
       
    Args:
        tweet (str): Tweet text to be tokenized.
        stop_list (iterable of strings): Strings to remove from the list of tokens before it is returned.
        pattern (regex str, default=r"([a-zA-Z]+(?:'[a-z]+)?)"): Pattern to tokenize by. 
            Default pattern keeps words with apostrophes as a single token, rather than splitting at apostrophe.
    
    Returns:
        iterable: List of tokens as strings.
    
    Example:
        >>> stopwords = stopwords.words('english')
        >>> t = "This ain't a real tweet"
        >>> tokenize_tweet(t, stop_list=stopwords)
        ["ain't", 'real', 'tweet']
    
       """
    
    import nltk
    from nltk import regexp_tokenize
    import re
    
    
    # Clean the tweet
    tweet = clean_tweet(tweet)
    
    # Make everything lower case
    tweet = tweet.lower()
    
    # Split into tokens
#     pattern = r"([a-zA-Z]+(?:'[a-z]+)?)"
    tokens = regexp_tokenize(tweet, pattern)
    
    # Remove stopwords and punctuation
    stopped_tokens = [w for w in tokens if w not in stop_list]
    
    return stopped_tokens







def get_token_list(df, col, freq=False):
    """Takes in a DataFrame and column that contains tokenized texts 
       and returns a list containing all the tokens (including duplicates) from that 
       column. If freq=True, the function will also print out the number of 
       unique tokens and the top 25 most common words as well as their counts based
       on nltk's FreqDist.
       
    Args:
        df (Pandas DataFrame): DataFrame from which to obtain tokenized text.
        col (str): Name of the column that contains the text to tokenize.
        freq (bool, default=False): Whether to print summary of token list.
    
    Returns:
        iterable: List of tokens as strings.
    
    Example:
        >>> df = pd.DataFrame({'numbers': [2, 4],
                   'text': [['an', 'example'],
                           ['another', 'example']]})

        >>> example_tokens = get_token_list(df, col='text', freq=True)
        >>> example_tokens
        
        ********** text Summary **********

        Number of unique words = 3
        token   count
        0   example 2
        1   an      1
        2   another 1
        
        ['an', 'example', 'another', 'example']
    
       """
    
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





def plot_wordcloud(tokens, title=None, save=False, fig_name=None, collocations=False):
    """Takes in a list of tokens and returns a wordcloud visualization of the most common words.
    
    Args:
        tokens (iterable of strings): List of tokens.
        title (str): A description of the tokens for customizing plot title.
        save (bool, default=False): Whether to save the returned figure.
        fig_name (str, optional): What to name the file if the image is being saved.
        collocations (bool, default=False): Whether to include collocations (bigrams) of two words.
    
    Returns:
        figure: WordCloud plot showing the most common tokens.
    
    Example:
        >>> plot_wordcloud(tokens=my_tokens, descr='My Tokens', save=True, fig_name='my_tokens_wordcloud')
    
    """
    
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    fig_filepath = 'Figures/'
    
    wordcloud = WordCloud(collocations=collocations, colormap='gist_rainbow',
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
       
       
    Args:
        clf (estimator): Fitted classifier.
        X_test (series or array): Subset of X data used for testing.
        y_test (series or array): Subset of y data used for testing.
        model_descr (str): A description of the model for customizing plot title.
        target_labels (list of strings, default=['Hate Speech', 'Offensive', 'Neither']): List of class labels 
            used for formatting tick labels.
        cmap (str, default='Blues'): Specifies a color map that can be used by sklearn's plot_confusion_matrix.
        normalize (str, {'true', 'pred', 'all', None}, default='true'): Whether to normalize the
        confusion matrix over the true (rows), predicted (columns) conditions or all the population. 
        If None, confusion matrix will not be normalized.
        save (bool, default=False): Whether to save the returned figure.
        fig_name (str, optional): What to name the file if the image is being saved.
    
    Returns:
        display: Sklearn classification report and confusion matrix.
    
    Example:
        >>> eval_classifier(clf=my_model, X_test, y_test, model_descr='My Model',
                    target_labels=['Hate Speech', 'Offensive', 'Neither'],
                    cmap='Blues', normalize='true', save=true, fig_name='my_model_eval')
    
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




def fit_grid_clf(clf, params, X_train, y_train, X_test, y_test,
                 model_descr='', score='accuracy'):
    
    """Given an sklearn classification model, hyperparameter grid, X and y training data, 
       and a GridSearchCV scoring metric (default is 'accuracy', which is the default metric for 
       GridSearchCV), fits a grid search of the specified parameters on the training data and 
       returns the grid object. Function also takes in X_test and y_test to get predictions and 
       evaluate model performance on test data. Prints out parameters of the best estimator as well 
       as its classification report and confusion matrix. A description of the model can be provided
       to model_descr to customize the title of the classification report.
       
    Args:
        clf (estimator): Fitted classifier.
        params (dict): Dictionary with parameters names (`str`) as keys and lists of 
            parameter settings to try as values.
        X_train (series or array): Subset of X data used for training.
        y_train (series or array): Subset of y data used for training.
        X_test (series or array): Subset of X data used for testing.
        y_test (series or array): Subset of y data used for testing.
        model_descr (str): A description of the model for customizing plot title.
        score (str, default='accuracy'): A string indicating a scoring method compatible with 
            sklearn.model_selection's GridSearchCV.
    
    Returns:
        grid: Fitted GridSearchCV object
    
    Example:
        >>> param_grid = {'param_name_1':[(1,1),(1,2),(1,3)],
                          'param_name_2':[0.005, 2, 3],
                         }
        >>> fit_grid_clf(clf=my_model, params=param_grid, X_train, y_train, X_test, y_test,
                 model_descr='My Model', score='accuracy')
    
    """
    
    from sklearn.model_selection import GridSearchCV
    import datetime as dt
    from tzlocal import get_localzone
    
    
    start = dt.datetime.now(tz=get_localzone())
    fmt= "%m/%d/%y - %T %p"
    
    print('---'*20)    
    print(f'***** Grid Search Started at {start.strftime(fmt)}')
    print('---'*20)
    print()
    
    grid = GridSearchCV(clf, params, scoring=score, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    end = dt.datetime.now(tz=get_localzone())
    
    print(f'\n***** Training Completed at {end.strftime(fmt)}')
    print(f"\n***** Total Training Time: {end-start}")
    print('\n')
    
    print('Best Parameters:')
    print(grid.best_params_)
    print('\n')
    eval_classifier(grid.best_estimator_, X_test, y_test, model_descr)
    
    return grid







def plot_feat_importance(clf, clf_step_name, vec_step_name, model_title='', save=False, fig_name=None):
    
    """Takes in an sklearn classifier already fit to training data, the name of the step for that model
       in the modeling pipeline, the vectorizer step name, and optionally a title describing the model. 
       Returns a horizontal barplot showing the top 20 most important features in descending order.
         
    Args:
        clf (estimator): An sklearn Pipeline with a vectorizer steps and final step is a fitted classifier.
        clf_step_name (str): The name given to the classifier step of the pipe.
        vec_step_name (str): The name given to the vectorizer step of the pipe.
        model_title (str): A description of the model for customizing plot title.
        save (bool, default=False): Whether to save the returned figure.
        fig_name (str, optional): What to name the file if the image is being saved.
    
    Returns:
        figure: Matplotlib.pyplot bar plot figure showing the feature importance values for the 
            20 most important features.
    
    Example:
        >>> plot_feat_importance(clf=my_model, clf_step_name='clf', vec_step_name='vec',
                                 model_title='My Model', save=True, fig_name='my_model_feat_import')
    
    """

    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    import matplotlib.pyplot as plt
    
    fig_filepath = 'Figures/'
    
    feature_importances = (
        clf.named_steps[clf_step_name].feature_importances_)
    
    feature_names = (
        clf.named_steps[vec_step_name].vocabulary_) 
    
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






def plot_coefficients(clf, clf_step_name, vec_step_name,
                      class_label, model_title='', top_features=10,
                      save=False, fig_name=None):
    
    """Takes in an sklearn classifier already fit to training data, the name of the step for that model
       in the modeling pipeline, the vectorizer step name, a class label, and optionally a title describing the model. 
       Returns a horizontal barplot showing the top 20 most important features by coefficient weight (10 most 
       positive and 10 most negative).
       
    Args:
        clf (estimator): An sklearn Pipeline with a vectorizer steps and final step is a fitted classifier.
        clf_step_name (str): The name given to the classifier step of the pipe.
        vec_step_name (str): The name given to the vectorizer step of the pipe.
        class_label (int): Integer representing numerically encoded class of interest.
        model_title (str): A description of the model for customizing plot title.
        top_features (int, default=10): Number of top positive and top negative coefficients to plot
            (so default of 10 returns bar plot with 20 bars total).
        save (bool, default=False): Whether to save the returned figure.
        fig_name (str, optional): What to name the file if the image is being saved.
    
    Returns:
        figure: Matplotlib.pyplot bar plot figure showing the coefficient weights for the top
            20 most important features.
    
    Example:
        >>> plot_coefficients(clf=my_model, clf_step_name='clf', vec_step_name='vec',
                                 class_label=0, model_title='My Model', top_features=10,
                                 save=True, fig_name='my_model_coeffs')
    
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    fig_filepath = 'Figures/'
    
    ## Get the coefficients for the specified class label
    feature_coefs = (
        clf.named_steps[clf_step_name].coef_[class_label])
    
    ## Get the vocabulary from the fit vectorizer
    feature_names = (
        clf.named_steps[vec_step_name].vocabulary_) 
    # Create a version of the vocab dict with keys and values swapped
    vocab_swap = (
        {value:key for key, value in feature_names.items()}) 

    
    ## Store the top 10 positive coefficients and their indices
    pos_10_index = (
        np.argsort(clf.named_steps[clf_step_name].coef_[class_label])[-top_features:])
    pos_10_coefs = (
        np.sort(clf.named_steps[clf_step_name].coef_[class_label])[-top_features:])
    
    ## Store the top 10 negative coefficients and their indices
    neg_10_index = (
        np.argsort(clf.named_steps[clf_step_name].coef_[class_label])[:top_features])
    neg_10_coefs = (
        np.sort(clf.named_steps[clf_step_name].coef_[class_label])[:top_features])
    
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
       
    Args:
        df (Pandas DataFrame): DataFrame containing the column with the text you want to print.
        col (str, default='text'): Name of the column containing the text to be printed.
        title (str): A description of the type of tweets for customizing plot title.
    
    Returns:
        display: Simply prints out a non-truncated version of the specified texts.
    
    Example:
        >>> df = pd.DataFrame({'numbers': [2, 4, 4],
                   'text': ['A simple example.',
                            'And another.',
                            'One more string.']})

        >>> print_full_tweet(df, col='text', title='Example for Docstring')
        
        ************************************************************ 

        Example for Docstring 

        Number of tweets: 3 

        ************************************************************ 

        0 A simple example. 
         ------------------------------------------------------------ 

        1 And another. 
         ------------------------------------------------------------ 

        2 One more string. 
         ------------------------------------------------------------ 
    
    """
    
    import pandas as pd
    
    print('***'*20, '\n')
    print(title, '\n')
    print('Number of tweets:', len(df), '\n')
    print('***'*20, '\n')
    
    for i in range(len(df)):
        print(i , df.iloc[i]['text'], '\n', '---'*20, '\n')