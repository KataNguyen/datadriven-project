import numpy as np
import pandas as pd
import matplotlib as mpl ; mpl.use('Agg')
import matplotlib.pyplot as plt ; plt.ioff()
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter
import matplotlib.ticker
import openpyxl
import os
import sys
from os import listdir
from os.path import isfile, isdir, join, realpath, dirname
from win32com.client import Dispatch
import time
from datetime import datetime, timedelta
import requests
import json
import holidays
from typing import Union
import itertools

###############################################################################

from scipy import stats

###############################################################################

from sklearn.cluster import KMeans
from scipy.stats import rankdata
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

###############################################################################

from bs4 import BeautifulSoup
from urllib.request import urlopen
