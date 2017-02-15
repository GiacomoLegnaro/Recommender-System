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
    def __init__(self, id, age, gender, occupation, zip):
        self.id = int(id)
        self.age = int(age)
        self.gender = gender
        self.occupation = occupation
        self.zip = zip
        self.avg_rate = 0.0
        self.std_rate = 0.0


class Item:
    """
    Information about the items (movies)
    """
    def __init__(self, id, title, release_date, video_release_date, imdb_url,
                 unknown, action, adventure, animation, children, comedy,
                 crime, documentary, drama, fantasy, film_noir, horror,
                 musical, mystery, romance, sci_fi, thriller, war, western):
        self.id = int(id)
        self.title = title
        self.release_date = release_date
        self.video_release_date = video_release_date
        self.imdb_url = imdb_url
        self.unknown = int(unknown)
        self.action = int(action)
        self.adventure = int(adventure)
        self.animation = int(animation)
        self.children = int(children)
        self.comedy = int(comedy)
        self.crime = int(crime)
        self.documentary = int(documentary)
        self.drama = int(drama)
        self.fantasy = int(fantasy)
        self.film_noir = int(film_noir)
        self.horror = int(horror)
        self.musical = int(musical)
        self.mystery = int(mystery)
        self.romance = int(romance)
        self.sci_fi = int(sci_fi)
        self.thriller = int(thriller)
        self.war = int(war)
        self.western = int(western)


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
        for line in raw_data:
            obs = line.split('|')
            if len(obs) == 5:
                u.append(User(obs[0], obs[1], obs[2], obs[3], obs[4]))
        f.close()

    def loadItem(self, file, i):
        f = open(file, 'r')
        text = f.read()
        raw_data = text.split('\n')
        for line in raw_data:
            obs = line.split('|')   # line.split('|', 5)
            if len(obs) == 24:
                i.append(Item(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5],
                         obs[6], obs[7], obs[8], obs[9], obs[10], obs[11],
                         obs[12], obs[13], obs[14], obs[15], obs[16], obs[17],
                         obs[18], obs[19], obs[20], obs[21], obs[22], obs[23]))
        f.close()

    def loadRatings(self, file, r):
        f = open(file, 'r')
        text = f.read()
        raw_data = text.split('\n')
        for line in raw_data:
            obs = line.split('\t')
            if len(obs) == 4:
                r.append(Rating(obs[0], obs[1], obs[2], obs[3]))
        f.close()
