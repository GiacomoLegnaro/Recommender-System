#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Short summary that describe what the algorithm does
"""

import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from class_definition_1m import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from itertools import imap, product

__author__ = 'Giacomo Legnaro'
__version__ = '0.0'
__email__ = 'g.legnaro@gmail.com'
__status__ = 'Production'
__date__ = 'September 1st, 2016'
__studentID__ = '1724522'


def Offline():
    # -------------------------------------------------------------------------
    # Loading the data
    # -------------------------------------------------------------------------

    # Store data
    user = []
    item = []
    rating = []
    rating_test = []

    # Load the dataset
    d = Data()
    # d.loadUser('ml-100k/u.user', user)
    # d.loadItem('ml-100k/u.item', item)
    # d.loadRatings('ml-100k/u.data', rating)

    print 'Loading the dataset...'
    d.loadUser('ml-1m/users.dat', user)
    d.loadItem('ml-1m/movies.dat', item)
    d.loadRatings('ml-1m/ratings.dat', rating)
    print 'Load Done!\n'

    n_users = len(user)
    n_items = len(item)
    print 'Number of users = '+str(n_users) + \
          ' | Number of movies = '+str(n_items)

    print 'Splitting dataset: train 80% - test 20%'
    rating, rating_test = train_test_split(rating, test_size=0.2)
    print 'Splitting done!\n'

    print 'Size train set = '+str(len(rating)) + \
          ' | Size test set = '+str(len(rating_test))

    print 'Creating a users dictionary to correspond ml user_id ' + \
          'and our user_id'
    users_idx = {}
    for u in user:
        users_idx[u.id] = u.idx

    print 'Creating a items dictionary to correspond ml item_id ' + \
          'and our item_id'
    items_idx = {}
    for i in item:
        items_idx[i.id] = i.idx

    # -------------------------------------------------------------------------
    # Computational part
    # -------------------------------------------------------------------------

    print 'Create a utility matrix - train set'
    # The utility matrix stores the rating for each user-item pair in the
    # matrix form.
    utility = np.zeros((n_users, n_items)) * np.nan
    # we use nan values to simply some computational calculus respect to zero
    for r in tqdm(rating):
        utility[users_idx[r.user_id] - 1][items_idx[r.item_id] - 1] = r.rating
    # user_id and item_id start from 1 and not from zero
    print utility

    print 'Create a utility matrix - test set'
    utility_test = np.zeros((n_users, n_items)) * np.nan
    for r in tqdm(rating_test):
        utility_test[users_idx[r.user_id] - 1][items_idx[r.item_id] - 1] = \
            r.rating
    # user_id and item_id start from 1 and not from zero
    print utility_test

    print 'Computing the items cosine similarity...'
    utility_num = np.nan_to_num(utility)
    similarity = cosine_similarity(utility_num.T)

    # Compute the mean of all the dataset
    mu = float(np.nanmean(utility))

    print 'Computing utility baseline matrix.. [Waiting time ~ 15 min]'
    # ranking items
    # Consider the item columns utility matrix and subtract the mean of the
    # all dataset. Compute the mean of the ranked values.
    bi = []
    for i in tqdm(range(0, n_items)):
        bi.append(np.nansum(utility[:, i]-mu) /
            (25 + sum(~np.isnan(utility[:, i]))))

    # Consider the user rows utility matrix and subtract the mean of the
    # all dataset adn the corresponding baseline items value.
    # Compute the mean of the ranked values.
    bu = []
    for u in tqdm(range(0, n_users)):
        bu.append(np.nansum(utility[u, :]-mu-bi) /
            (10 + sum(~np.isnan(utility[u, :]))))

    # Make the utility baseline given by the sum of the user with the item and
    # the general mean
    utility_base = mu + np.array(
        list(imap(sum, product(bi, bu)))).reshape(n_items, n_users).T
    print 'Utility baseline matrix - Done!\n'

    # -------------------------------------------------------------------------
    # Prediction part
    # -------------------------------------------------------------------------

    print 'Predict the ratings for the test set'
    # Predict ratings for test set and find the root mean squared error
    y_true = []
    y_pred = []
    for u in tqdm(range(0, n_users)):
        for i in range(0, n_items):
            if ~np.isnan(utility_test[u, i]):
                num = np.nansum((utility[u, :] - utility_base[u, :]) *
                                similarity[:, i])
                # Consider the utility train row of a certain users, subtract
                # the predicted values and weight the movies by similarity
                den = sum(abs(similarity[~np.isnan(utility_test[u, :]), i]))
                # the absolute sum of similarity of the movies considered
                if den != 0:
                    y_pred.append(num / den + utility_base[u, i])
                else:
                    y_pred.append(utility_base[u, i])
                y_true.append(utility_test[u][i])

    # Last step
    RMSE = (mean_squared_error(y_true, y_pred))**0.5
    print "Root Mean Squared Error: %f" % RMSE

    # -------------------------------------------------------------------------
    # Store the Computational result
    # -------------------------------------------------------------------------
    similarity.tofile('output/similarity')
    utility_base.tofile('output/utility_base')

    with open('output/mu', 'wb') as fp:
        pickle.dump(mu, fp)

    with open('output/bi', 'wb') as fp:
        pickle.dump(bi, fp)

    with open('output/items_idx', 'wb') as fp:
        json.dump(items_idx, fp)


def Online():
    user = []
    item = []
    # Load the dataset
    d = Data()
    d.loadUser('ml-1m/users.dat', user)
    d.loadItem('ml-1m/movies.dat', item)

    n_users = len(user)
    n_items = len(item)

    rating_nu = []
    d.loadRatings('input/ratings.dat', rating_nu)

    similarity = np.fromfile('output/similarity')
    similarity = similarity.reshape(n_items, n_items)
    # utility_base = np.fromfile('output/utility_base')
    # utility_base = utility_base.reshape(n_users, n_items)
    with open('output/bi', 'rb') as fp:
        bi = pickle.load(fp)

    with open('output/mu', 'rb') as fp:
        mu = pickle.load(fp)

    with open('output/items_idx', 'rb') as fp:
        items_idx = json.load(fp)

    utility_nu = np.zeros((1, n_items)) * np.nan
    for r in rating_nu:
        utility_nu[0][items_idx[str(r.item_id)] - 1] = r.rating

    # mu = float(np.nanmean(utility_nu))
    num = np.nansum(utility_nu[0] - mu - bi)
    den = 10 + np.sum(~np.isnan(utility_nu))
    bu_nu = num/den

    # bu_nu = np.nansum(utility_nu[0, :]-mu-bi) / \
    #             (1e-9+sum(~np.isnan(utility_nu[0, :])))

    utility_base_nu = mu+bu_nu+bi

    y_pred = []
    y_pred_id = []
    for i in range(n_items):
        if np.isnan(utility_nu[0, i]):
            num = np.nansum((utility_nu[0] - utility_base_nu) *
                            similarity[:, i])
            den = sum(similarity[~np.isnan(utility_nu[0, :]), i])
            y_pred_id.append(i)
            if den != 0:
                y_pred.append(num / den + utility_base_nu[i])
            else:
                y_pred.append(utility_base_nu[i])

    items_title = {}
    for i in item:
        items_title[i.idx] = i.title

    top50 = np.argsort(y_pred)[-50:][::-1]
    print 'The 50 top movie recommended are:'
    for t in top50:
        print '# ', str(np.where(top50 == t)[0][0]+1), \
            '\t- ', items_title[y_pred_id[t]+1]


if __name__ == "__main__":
    print ('------   MovieLens 1M   ------')
    print ('\t 0: \tPress 0 to evaluate the recommender system offline')
    print ('\t 1: \tPress 1 to consider a new user')
    sel = int(raw_input('Choose: '))

    if sel == 0:
        Offline()
    else:
        Online()
