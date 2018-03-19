import json
import csv
import pandas as pd
import numpy as np
import re

from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from support_functions import *

"""
This assignment can be done in groups of 3 students. Everyone must submit individually.

Write down the UNIs of your group (if applicable)

Name : Tung Ngo
Uni  : tn2364

"""


def get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path):
    """
        In this function, You need to design your own algorithm or model to find the matches and generate
        a matches_test.csv in the current folder.

        you are given locu_train, foursquare_train json file path and matches_train.csv path to train
        your model or algorithm.

        Then you should test your model or algorithm with locu_test and foursquare_test json file.
        Make sure that you write the test matches to a file in the same directory called matches_test.csv.

    """
    pd.options.mode.chained_assignment = None
    fs = pd.read_json(foursquare_train_path)
    locu = pd.read_json(locu_train_path)
    truth = pd.read_csv(matches_train_path)
    fs_test = pd.read_json(foursquare_test_path)
    locu_test = pd.read_json(locu_test_path)
    
    # Format phone numbers in train and test set
    locu.phone = format_phone(locu.phone)
    locu_test.phone = format_phone(locu_test.phone)
    fs.phone = format_phone(fs.phone)
    fs_test.phone = format_phone(fs_test.phone)
    
    # Construct train_set
    relevant_comb = find_relevant_comb(locu,fs,threshold=0.05)
    train_set = create_train_set(locu,fs,truth,relevant_comb)
    train_set = create_features(train_set)
        
    #Construct test_set
    relevant_comb_test = find_relevant_comb(locu_test,fs_test,threshold=0.05)
    test_set = create_test_set(locu_test,fs_test,relevant_comb_test)
    test_set = create_features(test_set)
    
    X_train = train_set.drop(['match','locu_id','foursquare_id'],axis=1).values
    y_train = train_set['match'].values
    X_test = test_set.drop(['locu_id','foursquare_id'],axis=1).values
    
    # Editted Nearest Neighbors with Random Forest Model
    enn_pipe_rf = make_imb_pipeline(EditedNearestNeighbours(n_neighbors= 5),
                                  RandomForestClassifier(n_estimators=300,max_depth=5,
				  random_state=5,max_leaf_nodes=24,n_jobs=3))
    enn_pipe_rf.fit(X_train,y_train)
    predicted_proba_train = enn_pipe_rf.predict_proba(X_train)
    predicted_proba_test = enn_pipe_rf.predict_proba(X_test)
    graph_structure_train,locu_ids_train = get_graph_structure(predicted_proba_train,0.5,train_set)
    matches_train = bipartile_match(graph_structure_train,locu_ids_train)
    truth_list = [tuple(x) for x in truth.values]
    compute_metrics(truth_list,matches_train)
    graph_structure_test,locu_ids_test = get_graph_structure(predicted_proba_test,0.5,test_set)
    matches_test = bipartile_match(graph_structure_test,locu_ids_test)
    
    # Output csv and return
    pd.DataFrame(matches_test,columns=['locu_id','foursquare_id']).to_csv('matches_test.csv',index=False)
    return matches_test
    
    
    
