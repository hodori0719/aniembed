# Packages

import json
import time

import pandas as pd
import numpy as np

import mxnet as mx

from sklearn import manifold 

import requests

import urllib3

import plotly.express as px

import dash
from dash import dcc
from dash import html

from PIL import Image
from io import BytesIO 

from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from tqdm import tqdm

import logging

# Constants

# List of MAL ID's and metadata
ANIME_LIST = "animelist.txt"
ANIME_DATA = "animes.json"

# Pandas dataframe import
ANIME_DATA_FILE = "123.pkl"

# Exclude people who have watched too few anime or have no opinion about what they watch 
MIN_ANIME_REQUIRED = 15
BOTTER_LIMIT = 9.9

# Embeddings
N_EMBEDDING = 16
N_BATCH = 2**12
N_EPOCH = 100

LEARN_RATE = 0.005

SELECTION_RESAMPLING_FACTOR = 10
N_ACCOUNTS_PER_EPOCH = 100
N_ANIME_PER_ACCOUNT = 15
SAVE_ACCOUNTS = True

AVERAGE_METHOD = 'geom'     # 'geom', 'harm'
EPS_HARM = 10**-5
EPS_GEOM = 10**-5


SEED = 51
MX_CONTEXT = 'CPU'

##
# Image Output
ICON_ZOOM = 0.03

##
# Misc

# Path to Anime images
ANIME_IMAGE_URL = 'http://ddragon.leagueoflegends.com/cdn/10.20.1/img/champion/'

# Logging-Level
LOGGING_LEVEL = 'INFO'



###
# Neural net
def neuralNet(nAnime, nDimEmbedding):
    X = mx.sym.Variable('data')
    y = mx.sym.Variable('label')
    
    symEmb = mx.sym.Embedding(data = X, input_dim = nAnime, output_dim = nDimEmbedding)
    
    symEmbChamp1 = mx.sym.slice_axis(symEmb, 1, 0, 1)
    symEmbChamp2 = mx.sym.slice_axis(symEmb, 1, 1, 2)
    
    symEmbReshape1 = mx.sym.reshape(symEmbChamp1, (-1, nDimEmbedding))
    symEmbReshape2 = mx.sym.reshape(symEmbChamp2, (-1, nDimEmbedding))
    
    symDotProdAngle = mx.sym.sum(symEmbReshape1 * symEmbReshape2, axis = 1, keepdims = True)
    
    symError = mx.sym.LinearRegressionOutput(symDotProdAngle, y)
    
    return(symError)


###
# Iterator
class embeddingIterator(mx.io.DataIter):
    ###
    # Initializer
    def __init__(self, ratingDataFiltered, batchSize = N_BATCH, meanType = AVERAGE_METHOD, selectionResamplingFactor = SELECTION_RESAMPLING_FACTOR, nAccountsPerEpoch = N_ACCOUNTS_PER_EPOCH, nAnimePerAccount = N_ANIME_PER_ACCOUNT, saveAccounts = SAVE_ACCOUNTS, EPS_HARM = EPS_HARM, EPS_GEOM = EPS_GEOM):
        self.ratingValues = ratingDataFiltered.values
        self.batchSize = batchSize
        self.meanType = meanType
        self.selectionResamplingFactor = selectionResamplingFactor
        self.nAccountsPerEpoch = nAccountsPerEpoch
        self.nAnimePerAccount = nAnimePerAccount
        self.saveAccounts = saveAccounts
        
        self.EPS_HARM = EPS_HARM
        self.EPS_GEOM = EPS_GEOM
        
        # Calculate probabilities for random anime selection
        self.ratios = self.ratingValues / self.ratingValues.sum(axis = 1).reshape((-1,1))
        
        self.probForSampling = self.ratios + 1 / (self.ratios.shape[1] * selectionResamplingFactor)
        self.probForSampling = self.probForSampling / self.probForSampling.sum(axis = 1).reshape((-1,1))
        
        # Initialer Status
        self.reset()

        
    ###
    # Reset
    def reset(self):
        ###
        # Create data matrix
        
        # Determine indices for accounts
        if self.saveAccounts:
            maxNAccounts = self.nAccountsPerEpoch
        else:
            maxNAccounts = min(self.nAccountsPerEpoch, self.ratingValues.shape[0])
        
        if maxNAccounts % self.batchSize:
            accountOffset = self.batchSize - maxNAccounts % self.batchSize
        else:
            accountOffset = 0
        
        indicesAccounts = np.random.choice(self.ratingValues.shape[0], maxNAccounts, replace = self.saveAccounts)
        if accountOffset:
            indicesAccounts = np.concatenate((indicesAccounts, np.random.choice(self.ratingValues.shape[0], accountOffset, replace = True)))
        
        # Process indices
        dataMatrixList = list()
        for indexAccount in indicesAccounts:
            # Read anime
            indicesAnime = np.random.choice(self.ratingValues.shape[1], self.nAnimePerAccount, replace = False, p = self.probForSampling[indexAccount,:])
            
            for i in range(0, self.nAnimePerAccount):
                for j in range(i+1, self.nAnimePerAccount):
                    if self.meanType == 'harm':
                        dataMatrixList.append([indicesAnime[i], indicesAnime[j], 2 * self.ratios[indexAccount, indicesAnime[i]] * self.ratios[indexAccount, indicesAnime[j]] / (self.ratios[indexAccount, indicesAnime[i]] + self.ratios[indexAccount, indicesAnime[j]]) if self.ratios[indexAccount, indicesAnime[i]] > self.EPS_HARM and self.ratios[indexAccount, indicesAnime[j]] > self.EPS_HARM else 0])
                    elif self.meanType == 'geom':
                        dataMatrixList.append([indicesAnime[i], indicesAnime[j], np.sqrt(self.ratios[indexAccount, indicesAnime[i]] * self.ratios[indexAccount, indicesAnime[j]]) if self.ratios[indexAccount, indicesAnime[i]] > self.EPS_GEOM and self.ratios[indexAccount, indicesAnime[j]] > self.EPS_GEOM else 0])
                    else:
                        raise NotImplementedError('Mean type not implemented')
        
        self.dataMatrix = np.array(dataMatrixList)
        
        self.currentBatch = 0
        self.totalBatches = (self.dataMatrix.shape[0] - 1) // self.batchSize + 1
    
    
    ###
    # Next
    def next(self):
        if self.currentBatch < self.totalBatches:
            # Data
            adata = mx.nd.array(self.dataMatrix[(self.currentBatch * self.batchSize):((self.currentBatch + 1) * self.batchSize), 0:2])
            alabel = mx.nd.array(self.dataMatrix[(self.currentBatch * self.batchSize):((self.currentBatch + 1) * self.batchSize), 2])
        
            # Increment batch count
            self.currentBatch += 1
            
            # Output values
            return(mx.io.DataBatch(data = [adata], label = [alabel], pad = 0))
        else:
            raise(StopIteration)   
    
    
    ###
    # Data structure for prediction matrices
    @property
    def provide_data(self):
        return([('data', (self.batchSize, 2))])


    ###
    # Data structure for labels
    @property
    def provide_label(self):
        return([('label', (self.batchSize,))])    


###
# Main
if __name__ == '__main__':
    # Retrieve anime list and metadata
    animeList = []
    with open(ANIME_LIST, 'r') as animeListFile:
        for line in animeListFile:
            animeList.append(line.strip())
        
    with open(ANIME_DATA, 'r') as animeDictFile:
        animeDict = json.load(animeDictFile)
        
    ###
    # Set logging level
    loggingDict = {'DEBUG': 10, 'INFO': 20}
    logging.basicConfig(level = loggingDict[LOGGING_LEVEL])


    # Supress SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    ###
    # Fetch anime rating data

    # Read in data
    ratingData = pd.read_pickle(ANIME_DATA_FILE)
    
    # Filter data
    ratingDataFiltered = ratingData.loc[ratingData.count(axis = 1) < MIN_ANIME_REQUIRED].reset_index(drop = True)
    ratingDataFiltered = ratingData.loc[ratingData.mean(axis = 1) > BOTTER_LIMIT].reset_index(drop = True)
    ratingDataFiltered = ratingDataFiltered.fillna(0)
    
    ###
    # Define neural net
    netzSymb = neuralNet(ratingDataFiltered.shape[1], N_EMBEDDING)

    # Iterator
    np.random.seed(SEED)
    mx.random.seed(SEED)
    
    iterator = embeddingIterator(ratingDataFiltered)
    
    # Start neural net
    if MX_CONTEXT == 'GPU':
        mxContext = mx.gpu()
    elif MX_CONTEXT == 'CPU':
        mxContext = mx.cpu()
    else:
        raise AssertionError("Invalid mxNet context")
    
    nnModel = mx.model.Module(netzSymb, data_names = [x[0] for x in iterator.provide_data], label_names = [x[0] for x in iterator.provide_label], context = mxContext)

    nnModel.bind(data_shapes = iterator.provide_data, label_shapes = iterator.provide_label)
    nnModel.init_params(mx.initializer.Xavier(rnd_type = 'gaussian', magnitude = 1))
    nnModel.init_optimizer(optimizer = 'adam', optimizer_params = {'learning_rate': LEARN_RATE})


    ##
    # Train neural net
#    progressBar = mx.callback.ProgressBar(total = iterator.totalBatches)
#    nnModel.fit(iterator, eval_metric = 'mse', num_epoch = N_EPOCH, batch_end_callback = progressBar)
    nnModel.fit(iterator, eval_metric = 'mse', num_epoch = N_EPOCH)


    ###
    # Read weight matrix
    weightMatrix = nnModel.get_params()[0][list(nnModel.get_params()[0].keys())[0]].asnumpy()
    
    # Normalize
    weightMatrix = weightMatrix / np.linalg.norm(weightMatrix, axis = 1).reshape((-1,1))

    ###
    # MDS (2D transform)
    mdsTransformation = manifold.MDS(n_components = 2)
    mdsMatrix = mdsTransformation.fit_transform(weightMatrix)
    
    mds2dDf = pd.DataFrame(mdsMatrix, columns = ['X1', 'X2'])
    mds2dDf['Anime'] = animeList
    
    ###
    # Get anime images
    animeCovers = list()
    for anime in tqdm(animeList, total = len(animeList)):
        while True:
            pictureGet = requests.get(animeDict[anime]['main_picture']['medium'])
            if pictureGet.status_code == 200:
                break
            else:
                time.sleep(1)
        
        animeCovers.append(Image.open(BytesIO(pictureGet.content)))
    
    # Matplotlib
    fig = plt.figure(figsize=(3, 2), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.scatter(mds2dDf['X1'], mds2dDf['X2'], color = 'white')
    ax.axis('off')
    for i in range(0, len(animeList)):
        imageBox = OffsetImage(animeCovers[i], zoom = ICON_ZOOM)
        ax.add_artist(AnnotationBbox(imageBox, mds2dDf.iloc[i][['X1', 'X2']].values, frameon = False))
    
    fig.savefig("anime_clustering_mda_{topanime}.png".format(topanime = len(animeList)), transparent = False, bbox_inches = 'tight', pad_inches = 0, dpi = 1000)
    plt.close()
    
    
    ###
    # TSNE
    tsneTransformation = manifold.TSNE(n_components = 2)
    tsneMatrix = tsneTransformation.fit_transform(weightMatrix)

    tsne2dDf = pd.DataFrame(tsneMatrix, columns = ['X1', 'X2'])
    tsne2dDf['Anime'] = animeList    

    # Matplotlib
    fig = plt.figure(figsize=(3, 2), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.scatter(tsne2dDf['X1'], tsne2dDf['X2'], color = 'white')
    ax.axis('off')
    for i in range(0, len(animeList)):
        imageBox = OffsetImage(animeCovers[i], zoom = ICON_ZOOM)
        ax.add_artist(AnnotationBbox(imageBox, tsne2dDf.iloc[i][['X1', 'X2']].values, frameon = False))
    
    fig.savefig("anime_clustering_tsne_{topanime}.png".format(topanime = len(animeList)), transparent = False, bbox_inches = 'tight', pad_inches = 0, dpi = 1000)
    plt.close()

    
    ###
    # Plotly
    if (False):
        fig = px.scatter(mds2dDf, x = 'X1', y = 'X2', hover_data = {'X1': False, 'X2': False, 'Anime': True})
        
        app = dash.Dash()
        app.layout = html.Div([
            dcc.Graph(figure=fig)
        ])
        
        app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter