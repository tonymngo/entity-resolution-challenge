
# Problem Description
Entity resolution, often known in the context of record linkage problem, is the task to find matching entities across different databases. Records are differed due to several reasons, including different identifiers or data collection processes. In this exercise, I will present my approach to resolve the identities of restaurants between Foursquare and Locu databases (both of these companies provide listings of restaurants)

# The Data
Foursquare and Locu data each contain 1,000 records, in which 600 records in each datasets are used for training and the rest are used for testing. Most of the data have been streamlined to have similar data structures, meaning they each contain name, phone number, street address, postal code, longitude, latitude, region, etc. in JSON format. However, a large portion of the data are missing, and there is no guarantee same restaurants have exact same name or address across databases; in fact, some matching records can have totally different names, address or postal codes.

# Approach
By looking at the data, I noticed the phone number formats are different for locu and foursquare data, hence my first step was to bring these phone numbers to the same format by keeping only digits of the phone number.

In order to avoid pairwise comparison of all venues, I use approximate nearest neighbors technique and hash the values of each data point in locu and foursquare files using MinHash. The data I used to hash include words in name, street address, phone, postal code, and any combination of name with 2 words. The nearest neighbors are selected based on a threshold of Jaccard similarity (in my case I used 0.05). As a results, I am able to reduce the number of comparisons by three times (instead of comparing 320,000 pairs of train data and 160,000 pairs of test data, I compared 112,000 pairs of train data and 52,000 pairs of test data). The disadvantage of this technique is that with small probability we might miss comparison of relevant pairs, hence affecting the recall score.

After getting the nearest neighbors of each pair, I used a supervised learning approach, treating the matches_train.csv pairs as the ground truth. I read locu and foursquare json files into Pandas dataframe and joined them using the MinHash values. I then created appropriate features to train the model, including:

- Levenshtein distance of restaurant name
- Levenshtein distance of street address
- Whether both addresses exist (0/1)
- Longitude absolute differences
- Latitude absolute differences
- Whether both phones exist (0/1)
- Whether phones match (0/1)

Because we have a very imbalanced dataset, I chose appropriate models to handle this situation, including:

- Editted Nearest Neighbors with Random Forest - which is also my final chosen model because it gives the highest cross-validation f1-score
- Editted Nearest Neighbors with Logistic Regression
- Condensed Nearest Neighbors with Random Forest
- Random Undersampling with Random Forest
- Random Forest with balanced class weights

At this point, the output of model can contain pairs that are not exclusive, meaning one locu_id can be matched to more than one foursquare_id. In order to enforce exclusivity, I used the maximum weighted bipartile matching algorithm implemented in networkx package, with the edges' weights being the predicted proabilities by my Random Forest model. The output gave exclusive pairs with highest sum of predicted probabilities. 

I tuned the model a final time and fit on all training examples. The precision, recall, and f1-score of training set are as follow:

- precision = 100%
- recall = 96.94%
- f1-score = 98.45%

I preprocessed the test set and used this model to predict the matched pairs, then output to matches_test.csv in the current folder. The results of testing are as follow:

- precision = 99.57%
- recall = 97.08%
- f1-score = 98.31%

Final parameters of model:
```python
EditedNearestNeighbours(n_neighbors= 5),
RandomForestClassifier(n_estimators=300,max_depth=5,max_leaf_nodes=24)
```

The 3 most important features for my model are:

- Levenshtein distance of restaurant name
- Longitude absolute differences
- Latitude absolute differences

Note about the python function: the main get_matches function imports supporting functions from support_functions.py (which is also included in the zip file). The total time of training and predicting the outcome takes approximately 4 minutes on a laptop with Core i7 chip, 2.5 GHz CPU, 8 GB RAM. 