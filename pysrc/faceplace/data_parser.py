import matplotlib

matplotlib.use("Agg")
import scipy.stats as st
import scipy.linalg as la
import pylab as pl
import os
import copy

pl.ion()
import pdb
from utils import smartDumpDictHdf5, smartAppend
import h5py
import dask.array as da
import pandas as pd
import scipy as sp
from PIL import Image
import scipy.ndimage as img
from torch.utils.data import Dataset
import torch


class FaceDataset(Dataset):
    def __init__(self, Y, D, W):
        self.len = Y.shape[0]
        self.Y, self.D, self.W = Y, D, W

    def __getitem__(self, index):
        return (self.Y[index], self.D[index], self.W[index], index)

    def __len__(self):
        return self.len


def read_face_data(h5fn):

    f = h5py.File(h5fn, "r")
    keys = ["test", "train", "val"]
    Y = {}
    Rid = {}
    Did = {}
    for key in keys:
        Y[key] = f["Y_" + key][:]
    for key in keys:
        Rid[key] = f["Rid_" + key][:]
    for key in keys:
        Did[key] = f["Did_" + key][:]
    f.close()

    # exclude test and validation not in trian
    uDid = sp.unique(Did["train"])
    for key in ["test", "val"]:
        Iok = sp.in1d(Did[key], uDid)
        Y[key] = Y[key][Iok]
        Rid[key] = Rid[key][Iok]
        Did[key] = Did[key][Iok]

    # one hot encode donors
    table = {}
    for _i, _id in enumerate(uDid):
        table[_id] = _i
    D = {}
    for key in keys:
        D[key] = sp.array([table[_id] for _id in Did[key]])[:, sp.newaxis]

    # one hot encode views
    uRid = sp.unique(sp.concatenate([Rid[key] for key in keys]))
    table_w = {}
    for _i, _id in enumerate(uRid):
        table_w[_id] = _i
    W = {}
    for key in keys:
        W[key] = sp.array([table_w[_id] for _id in Rid[key]])[:, sp.newaxis]

    for key in keys:
        Y[key] = Y[key].astype(float) / 255.0
        Y[key] = torch.tensor(Y[key].transpose((0, 3, 1, 2)).astype(sp.float32))
        D[key] = torch.tensor(D[key].astype(sp.float32))
        W[key] = torch.tensor(W[key].astype(sp.float32))

    return Y, D, W
