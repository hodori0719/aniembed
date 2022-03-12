# AniEmbed

AniEmbed is a collection of Python scripts designed to generate neural-network based anime recommendations based on API data from the website MyAnimeList.net. AniEmbed comes in several disjoint parts: `embeddings.py`, which handles the neural network, the API download scripts for anime and anime rating data, and a simple web scraper for gathering usernames.

A front-end web app to allow queries on static results is in development.

## embeddings.py

A similarity graph generated for the top 100 anime of all time on MyAnimeList using `embeddings.py`:
![Top 100](https://myoctocat.com/assets/images/base-octocat.svg)

`embeddings.py` uses neural network embeddings based on Apache MXNet to represent anime as 50-dimensional vectors. These vectors are trained on a neural network which takes lists of user ratings as inputs; the closer a single user rated two anime together, the more "similar" they are. This similarity index is calculated using the geometric mean of the offsets of the userscores from the mean scores, to account for popularity bias. You can find a simplified TensorFlow version of the neural net definition in the `embeddings.ipynb` Jupyter Notebook, but the data iterator is in `embeddings.py`.

The closer the dot product of two anime vectors is to 1, the more similar they will be. Since the resulting matrix is of n x 50 dimensions, n being the number of titles, this is impossible to visualize normally; thus, these vectors are mapped onto two dimensions using TSNE for better visualization.

An example visualization trained on the 100 top anime with 100000 user lists is shown above. Anime closer to each other will be more similar, and those farther will be less. Notice the shonen cluster on the left, for instance, and the slice-of-life cluster to the right. (A few interesting observations: all of Kyoto Animation's titles (Hyouka, Clannad, KnK) are on right top of each other except Violet Evergarden, which is closer to the "edgelord" cluster with Elfen Lied and Evangelion! We also see popular shows like Konosuba S1 being closer to the shonen/isekai clusters, but S2 shifts back towards the slice of life category.)

To generate a graph, simply do:
```bash
python3 embeddings.py
```
after adjusting the global learning parameters in the heading. You will need files at the directories dictated by `ANIME_LIST`, `ANIME_DATA`, and a pandas dataframe pickle `ANIME_DATA_LIST` produced by `lists_download.ipynb`.

## anime_download.ipynb, lists_download.ipynb

To run either of these items, you need a MyAnimeList API authentication token, which has been omitted from the public releases of these Notebooks. There is an official (but poor, IMO) [documentation of the MAL RESTful API](https://myanimelist.net/apiconfig/references/api/v2). 

*WARNING*: MAL has no officially documented call limit, but generally waiting 500-1000ms between calls should be fine. Any more will lead to a temporary ban.

## anime_download.ipynb, lists_download.ipynb

To run either of these items, you need a MyAnimeList API authentication token, which has been omitted from the public releases of these Notebooks. There is an official (but poor, IMO) [documentation of the MAL API](https://myanimelist.net/apiconfig/references/api/v2). 

Warning: MAL uses a RESTful API and has no officially documented call limit, but generally waiting 500-1000ms between calls should be fine. Any more will lead to a temporary ban.

## malscraper.py

Since the MAL API does not provide any method of retrieving lists of usernames, and there is no advanced search function for this on the website, brute force scraping is necessary to gather usernames. Based on MAL's "Online Today" metric, this scraper should catch upwards of 80% of users active on a given day. 

The scraper repeatedly requests the active users page at [https://myanimelist.net/users.php](https://myanimelist.net/users.php). Since most inactive accounts on MAL have completely empty lists, it is not efficient to attempt to gather usernames through tree search. Still, this suffices for the purposes of this analysis. A list of over 100,000 usernames pulled from the scraper is provided in `test-data/users-long.txt`.

## Misc

A front-end app using React.js to traverse through a pre-generated embedding graph is in development, but it's on hold until I'm able to resolve problems with MXNet. Feel free to friend me on MAL @Ho-kagoTeaTime :)
