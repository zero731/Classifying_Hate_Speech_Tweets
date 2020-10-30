# Mod 4 Final Project - 
# Classifying Hate Speech Tweets

#### Author: Max Steele

The goal of this project was to build a classifier to predict 

## Data
The data used were obtained from <a href=""> </a>. The dataset contains the following columns/ information: 
* 




## Methods
I followed the OSEMN data science process to approach this problem. 


### Tweet Processing







### Modeling
Classifiers were fit using Scikit-Learn (for random forests, Multinomial Naive Bayes, and LinearSVC) or XGBoost (for XGB classifiers). All types of classifiers were first fit using default parameters, then tuned to optimize accuracy and then balanced accuracy using GridSearchCV to test a grid of hyperparameter values.

Model quality and performance were primarily assessed based on overall accuracy and the recall for the 'Hate Speech' class.


## Results


### Random Forest

###

###

### XGradient Boosted



### Interpretation of Best Classifier




## Conclusions and Recommendations




## Future Work
