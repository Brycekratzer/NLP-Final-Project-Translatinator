import pandas as pd 
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm as tqdm
from evaluate import load
from transformers import MarianMTModel, MarianTokenizer

sentences = pd.read_csv('../data/en-es_Full_Dataset.csv')


