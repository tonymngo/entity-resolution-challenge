#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tungngo
"""

import pandas as pd
import numpy as np
import re
from datasketch import MinHash, MinHashLSH
import networkx as nx
from nltk.metrics.distance import edit_distance
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from itertools import combinations


def format_phone(phones):
    phone_fm = []
    for number in phones:
        if number:
            if number == '':
                phone_fm.append(None)
            else:
                digits = re.findall('\d+',str(number))
                if len(digits) > 1:
                    phone_fm.append(''.join(digits))
                else:
                    phone_fm.append(digits[0])
        else:
            phone_fm.append(None)
    return phone_fm


def MinHashFunc(df,num_perm,stop_words):
    """
    Take in dataframe with relevant columns to append and hash row based on these columns
    Return: dictionary of MinHash objects for all rows
    """
    row_hash = {}
    stemmer = WordNetLemmatizer()
    # Iterate through rows in table and min hash each row, then return the dictionary of all rows and MinHash objects
    for i, row in df.iterrows():
        row_hash[i] = MinHash(num_perm=num_perm)
        split_name = re.sub('[^A-Za-z0-9]+', ' ', row['name'].lower()).split()
        stop_words_remove = [w for w in split_name if not w in stop_words]
        name_stem = [stemmer.lemmatize(w) for w in stop_words_remove]
        name_comb = [''.join(w) for w in list(combinations(name_stem,2))]
        split_add = re.sub('[^A-Za-z0-9]+', ' ', row['street_address'].lower()).split()
        row_values = name_stem + name_comb + split_add + [row['phone']]*2 + [row['postal_code']]
#        print(row_values)
        for j in row_values:
            try:
                row_hash[i].update(j.encode('utf8'))
            except AttributeError:
                continue
    return row_hash


def find_relevant_comb(locu,fs,threshold=0.2,num_perm=128,weights=(0.3,0.7),stop_words=set(stopwords.words('english'))):
    # MinHash all rows using relevant columns
    to_hash_locu = locu[['name','street_address','phone','postal_code']]
    to_hash_fs = fs[['name','street_address','phone','postal_code'  ]]
    row_hash_locu = MinHashFunc(to_hash_locu,num_perm,stop_words)
    row_hash_fs = MinHashFunc(to_hash_fs,num_perm,stop_words)
    
    # Approximate nearest neighbors with LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm,weights=weights)
    for i, value in  row_hash_fs.items():
        lsh.insert(i,value)
    
    #Query relevant combination of locu_id and fs_id
    relevant_comb = []
    for i in locu.index.values:
        locu_id = locu.iloc[i]['id']
        fs_id_list = [fs.iloc[id]['id'] for id in lsh.query(row_hash_locu[i])]
        relevant_comb += list((locu_id,fs_id) for fs_id in fs_id_list)
    return relevant_comb


def create_train_set(locu,fs,truth,relevant_comb):
    # Construct train set by appending not_matched df with matched df
    truth_list = [tuple(x) for x in truth.values]
    matched = pd.DataFrame(list(set(truth_list)&set(relevant_comb)),columns=['locu_id','foursquare_id'])
    not_matched = pd.DataFrame(list(set(relevant_comb)-set(truth_list)),columns=['locu_id','foursquare_id'])
    matched['match']=1
    not_matched['match']=0
    
    train = pd.concat([matched,not_matched]).reset_index(drop=True)
    train = train.merge(locu,left_on='locu_id',right_on='id')
    train = train.merge(fs, left_on='foursquare_id',right_on='id',suffixes=('_locu','_fs'))
    train = train[['locu_id','foursquare_id','latitude_locu','longitude_locu','name_locu','street_address_locu','phone_locu',
                  'latitude_fs','longitude_fs','name_fs','street_address_fs','phone_fs','match']]
    train = train.fillna(0)
    
    return train


def create_test_set(locu_test,fs_test,relevant_comb):
    comb_to_check = pd.DataFrame(relevant_comb,columns=['locu_id','foursquare_id'])
    comb_to_check = comb_to_check.merge(locu_test,left_on='locu_id',right_on='id')
    comb_to_check = comb_to_check.merge(fs_test, left_on='foursquare_id',right_on='id',suffixes=('_locu','_fs'))
    comb_to_check = comb_to_check[['locu_id','foursquare_id','latitude_locu','longitude_locu','name_locu','street_address_locu','phone_locu',
                  'latitude_fs','longitude_fs','name_fs','street_address_fs','phone_fs']]
    test = comb_to_check.fillna(0)
    return test


#Create new features
def create_features(df):
    name_dist = []
    add_exist = []
    add_dist = []
    long_dist = []
    lat_dist = []
    phone_exist = []
    phone_match = []
    
    for i,row in df.iterrows():
        # edit distance of names
        name_dist.append(edit_distance(row['name_locu'].lower(),row['name_fs'].lower()))
        
        # address' edit distance and whether address exists
        if (row['street_address_locu']!='' and row['street_address_fs']!=''):
            add_dist.append(edit_distance(row['street_address_locu'].lower(),row['street_address_fs'].lower(),substitution_cost=2))
            add_exist.append(1)
        else:
            add_dist.append(9999)
            add_exist.append(0)
        
        # longitude and latitude differences
        long_dist.append(abs(row['longitude_locu']-row['longitude_fs']))
        lat_dist.append(abs(row['latitude_locu']-row['latitude_fs']))
        
        # same phone and whether phone exists (i.e. both phones are not 0s)
        if (row['phone_locu']!=0 or row['phone_fs']!=0):
            if (row['phone_locu'] == row['phone_fs']):
                phone_exist.append(1)
                phone_match.append(1)
            else:
                phone_exist.append(1)
                phone_match.append(0)
        else:
            phone_exist.append(0)
            phone_match.append(0)
    
    # return new dataframe as result
    if 'match' in df.columns:
        new_df = df[['locu_id','foursquare_id','match']]
    else:
        new_df = df[['locu_id','foursquare_id']]
    new_df['name_dist'],new_df['add_dist'],new_df['add_exist'],new_df['long_dist'],new_df['lat_dist'],new_df['phone_exist'],new_df['phone_match'] = \
        name_dist, add_dist, add_exist, long_dist, lat_dist, phone_exist, phone_match
    new_df.reset_index(drop=True,inplace=True)
    return new_df


def get_graph_structure(pred_proba, threshold, original_df):
    """
    Input: predicted probabilities, threshold of predicted proba, test set
    Return: prediction with probabilies which will be used as nodes, edges, and weights of bipartile graph
    """
    
    proba_df = pd.DataFrame(pred_proba)
    proba_df = proba_df[proba_df[1]>=threshold]
    
    full_pred = proba_df.join(original_df)[['locu_id','foursquare_id',1]]
    locu_id_list = full_pred['locu_id'].values
    model_pred = [tuple(x) for x in full_pred.values]
    return model_pred,locu_id_list


# Build graph and enforce exclusivity with bipartile matching
def bipartile_match(edges,locu_id_list):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    bipartile_matching = nx.max_weight_matching(G)
    
    # Get the list of 
    matches = []
    for key,value in  bipartile_matching.items():
        if key in locu_id_list:
            matches.append((key,value))
    return matches


def compute_metrics(truth,prediction):
    precision = len(set(prediction)&set(truth))/len(prediction)
    recall = len(set(prediction)&set(truth))/len(truth)
    f1_score = 2/(1/recall + 1/precision)
    print("precision =",precision)
    print("recall =",recall)
    print("f1-score =",f1_score)