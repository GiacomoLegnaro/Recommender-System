#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Short summary that describe what the algorithm does
"""


__author__ = 'Giacomo Legnaro'
__version__ = '1.0'
__email__ = 'g.legnaro@gmail.com'
__status__ = 'Production'
__date__ = 'September 1st, 2016'

import re

class User:
    """
    Demographic information about the users
    """
    def __init__(self, idx, id, gender, age, occupation, zip):
        self.idx = int(idx)
        self.id = int(id)
        self.gender = gender
        self.age = int(age)
        self.occupation = int(occupation)
        self.zip = zip
        self.avg_rate = 0.0
        self.std_rate = 0.0


class Item:
    """
    Information about the items (movies)
    """
    def __init__(self, idx, id, title, genre):
        self.idx = int(idx)
        self.id = int(id)
        self.title = title
        self.genre = genre


class Rating:
    """
    The full u data set:
     - 100000 ratings
     - 943 users
     - 1682 items
    Each user has rated at least 20 movies. Users and items are numbered
    consecutively from 1. The data is randomly ordered.
    """
    def __init__(self, user_id, item_id, rating, timestamp):
        self.user_id = int(user_id)
        self.item_id = int(item_id)
        self.rating = int(rating)
        self.timestamp = timestamp


class Data:
    """
    Class to import correctly the data.
    We define three functions to divide the work to each type of data
    """
    def loadUser(self, file, u):
        f = open(file, 'r')
        text = f.read()
        raw_data = re.split('\n', text)
        ct = 0
        for line in raw_data:
            obs = line.split('::')
            if len(obs) == 5:
                ct += 1
                u.append(User(ct, obs[0], obs[1], obs[2], obs[3], obs[4]))
        f.close()

    def loadItem(self, file, i):
        f = open(file, 'r')
        text = f.read()
        raw_data = text.split('\n')
        ct = 0
        for line in raw_data:
            obs = line.split('::')
            if len(obs) == 3:
                ct += 1
                i.append(Item(ct, obs[0], obs[1], obs[2]))
        f.close()

    def loadRatings(self, file, r):
        f = open(file, 'r')
        text = f.read()
        raw_data = text.split('\n')
        for line in raw_data:
            obs = line.split('::')
            if len(obs) == 4:
                r.append(Rating(obs[0], obs[1], obs[2], obs[3]))
        f.close()
