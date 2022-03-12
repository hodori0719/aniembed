#!/usr/bin/env python
# coding: utf-8

import requests

from bs4 import BeautifulSoup

import time

URL = "https://myanimelist.net/users.php"
LOOPS = 100
LARGE_LOOPS = 100
TEN_MINUTES = 600

# The following line initializes the set of users to blank. In an IPYNB, only do this once since keeping this uncommented will only sample the users online at runtime
userlist = set(())

# We could use tor here to pull more requests simultaneously, but I've found this only gives around 11% more usernames with 10 times the calls
for i in range(LARGE_LOOPS):
    print("Starting cycle number " + str(i))
    for j in range(LOOPS):
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.find(id="content")
        usernames = results.find_all("a", itemprop='', string=True)
        for user in usernames:
            userlist.add(user.text)
        time.sleep(1)
    print("Current cycle complete, number " + str(i) + ", sleeping for 10 minutes")
    time.sleep(TEN_MINUTES)

# Note, although we certainly won't use all of these usernames, it's generally a bad idea to filter the data pre-processing since userdata may change wildly between scraping and analysis

with open("users.txt", 'w') as outfile:
    for username in userlist:
        outfile.write(username + "\n")
