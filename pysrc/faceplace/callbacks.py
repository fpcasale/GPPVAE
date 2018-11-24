import sys

sys.path.append("./..")
import pdb
import os
import scipy as sp
import scipy.stats as st
import scipy.linalg as la
import numpy as np
import pylab as pl
import torch
from torch.autograd import Variable


def _compose(orig, recon):
    _imgo = []
    _imgr = []
    for i in range(orig.shape[0]):
        _imgo.append(orig[i])
    for i in range(orig.shape[0]):
        _imgr.append(recon[i])
    _imgo = sp.concatenate(_imgo, 1)
    _imgr = sp.concatenate(_imgr, 1)
    _rv = sp.concatenate([_imgo, _imgr], 0)
    _rv = sp.clip(_rv, 0, 1)
    return _rv


def _compose_multi(imgs):
    _imgs = []
    for i in range(len(imgs)):
        _imgs.append([])
        for j in range(imgs[i].shape[0]):
            _imgs[i].append(imgs[i][j])
        _imgs[i] = sp.concatenate(_imgs[i], 1)
    _rv = sp.concatenate(_imgs, 0)
    _rv = sp.clip(_rv, 0, 1)
    return _rv


def callback(epoch, val_queue, vae, history, figname, device):

    with torch.no_grad():

        # compute z
        zm = []
        zs = []
        for batch_i, data in enumerate(val_queue):
            y = data[0].to(device)
            _zm, _zs = vae.encode(y)
            zm.append(_zm.data.cpu().numpy())
            zs.append(_zs.data.cpu().numpy())
        zm, zs = sp.concatenate(zm, 0), sp.concatenate(zs, 0)

        # init fig
        pl.figure(1, figsize=(8, 8))

        # plot history
        xs = sp.arange(1, epoch + 2)
        keys = ["loss", "nll", "kld", "mse"]
        plots = [1, 2, 5, 6]
        for ik, key in enumerate(keys):
            pl.subplot(4, 4, plots[ik])
            pl.title(key)
            pl.plot(xs, history[key], "k")
            if key not in ["lr", "vy"]:
                pl.plot(xs, history[key + "_val"], "r")
            if key == "mse":
                pl.ylim(0.0, 0.01)

        # plot hist of zm and zs
        pl.subplot(4, 4, 13)
        pl.title("Zm")
        _y, _x = np.histogram(zm.ravel(), 30)
        _x = 0.5 * (_x[:-1] + _x[1:])
        pl.plot(_x, _y, "k")
        pl.subplot(4, 4, 14)
        pl.title("log$_{10}$Zs")
        _y, _x = np.histogram(sp.log10(zs.ravel()), 30)
        _x = 0.5 * (_x[:-1] + _x[1:])
        pl.plot(_x, _y, "k")

        # val reconstructions
        _zm = Variable(torch.tensor(zm[:24]), requires_grad=False).to(device)
        Rv = vae.decode(_zm[:24]).data.cpu().numpy().transpose((0, 2, 3, 1))
        Yv = val_queue.dataset.Y[:24].numpy().transpose((0, 2, 3, 1))

        # make plot
        pl.subplot(4, 2, 2)
        _img = _compose(Yv[0:6], Rv[0:6])
        pl.imshow(_img)

        pl.subplot(4, 2, 4)
        _img = _compose(Yv[6:12], Rv[6:12])
        pl.imshow(_img)

        pl.subplot(4, 2, 6)
        _img = _compose(Yv[12:18], Rv[12:18])
        pl.imshow(_img)

        pl.subplot(4, 2, 8)
        _img = _compose(Yv[18:24], Rv[18:24])
        pl.imshow(_img)

        pl.savefig(figname)
        pl.close()


def callback_gppvae0(epoch, history, covs, imgs, ffile):

    # init fig
    pl.figure(1, figsize=(8, 8))
    pl.subplot(4, 4, 1)
    pl.title("loss")
    pl.plot(history["loss"])
    pl.subplot(4, 4, 2)
    pl.title("vars")
    pl.plot(sp.array(history["vs"])[:, 0], "r")
    pl.plot(sp.array(history["vs"])[:, 1], "k")
    pl.subplot(4, 4, 5)
    pl.title("mse_out")
    pl.plot(history["mse_out"])

    pl.subplot(4, 4, 9)
    pl.title("XX")
    pl.imshow(covs["XX"], vmin=-0.4, vmax=1)
    pl.colorbar()
    pl.subplot(4, 4, 10)
    pl.title("WW")
    pl.imshow(covs["WW"], vmin=-0.4, vmax=1)
    pl.colorbar()

    Yv, Rv = imgs["Yv"], imgs["Yo"]

    # make plot
    pl.subplot(4, 2, 2)
    _img = _compose(Yv[0:6], Rv[0:6])
    pl.imshow(_img)

    pl.subplot(4, 2, 4)
    _img = _compose(Yv[6:12], Rv[6:12])
    pl.imshow(_img)

    pl.subplot(4, 2, 6)
    _img = _compose(Yv[12:18], Rv[12:18])
    pl.imshow(_img)

    pl.subplot(4, 2, 8)
    _img = _compose(Yv[18:24], Rv[18:24])
    pl.imshow(_img)

    pl.savefig(ffile)
    pl.close()


def callback_gppvae(epoch, history, covs, imgs, ffile):

    # init fig
    pl.figure(1, figsize=(8, 8))
    pl.subplot(4, 4, 1)
    pl.title("loss")
    pl.plot(history["loss"], "k")
    pl.subplot(4, 4, 2)
    pl.title("vars")
    pl.plot(sp.array(history["vs"])[:, 0], "r")
    pl.plot(sp.array(history["vs"])[:, 1], "k")
    pl.ylim(0, 1)
    pl.subplot(4, 4, 5)
    pl.title("recon_term")
    pl.plot(history["recon_term"], "k")
    pl.subplot(4, 4, 6)
    pl.title("gp_nll")
    pl.plot(history["gp_nll"], "k")
    pl.subplot(4, 4, 9)
    pl.title("mse_out")
    pl.plot(history["mse_out"], "k")
    pl.ylim(0, 0.1)
    pl.subplot(4, 4, 10)
    pl.title("mse")
    pl.plot(history["mse"], "k")
    pl.plot(history["mse_val"], "r")
    pl.ylim(0, 0.01)

    pl.subplot(4, 4, 13)
    pl.title("XX")
    pl.imshow(covs["XX"], vmin=-0.4, vmax=1)
    pl.colorbar()
    pl.subplot(4, 4, 14)
    pl.title("WW")
    pl.imshow(covs["WW"], vmin=-0.4, vmax=1)
    pl.colorbar()

    Yv, Yr, Rv = imgs["Yv"], imgs["Yr"], imgs["Yo"]

    # make plot
    pl.subplot(4, 2, 2)
    _img = _compose_multi([Yv[0:6], Yr[0:6], Rv[0:6]])
    pl.imshow(_img)

    pl.subplot(4, 2, 4)
    _img = _compose_multi([Yv[6:12], Yr[6:12], Rv[6:12]])
    pl.imshow(_img)

    pl.subplot(4, 2, 6)
    _img = _compose_multi([Yv[12:18], Yr[12:18], Rv[12:18]])
    pl.imshow(_img)

    pl.subplot(4, 2, 8)
    _img = _compose_multi([Yv[18:24], Yr[18:24], Rv[18:24]])
    pl.imshow(_img)

    pl.savefig(ffile)
    pl.close()
