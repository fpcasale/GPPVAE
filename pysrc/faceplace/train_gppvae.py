import matplotlib
import sys

matplotlib.use("Agg")
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vae import FaceVAE
from vmod import Vmodel
from gp import GP
import h5py
import scipy as sp
import os
import pdb
import logging
import pylab as pl
from utils import smartSum, smartAppendDict, smartAppend, export_scripts
from callbacks import callback_gppvae
from data_parser import read_face_data, FaceDataset
from optparse import OptionParser
import logging
import pickle
import time


parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default="./../data/faceplace/data_faces.h5",
    help="dataset path",
)
parser.add_option(
    "--outdir", dest="outdir", type=str, default="./../out/gppvae", help="output dir"
)
parser.add_option("--vae_cfg", dest="vae_cfg", type=str, default=None)
parser.add_option("--vae_weights", dest="vae_weights", type=str, default=None)
parser.add_option("--seed", dest="seed", type=int, default=0, help="seed")
parser.add_option(
    "--vae_lr",
    dest="vae_lr",
    type=float,
    default=2e-4,
    help="learning rate of vae params",
)
parser.add_option(
    "--gp_lr", dest="gp_lr", type=float, default=1e-3, help="learning rate of gp params"
)
parser.add_option(
    "--xdim", dest="xdim", type=int, default=64, help="rank of object linear covariance"
)
parser.add_option("--bs", dest="bs", type=int, default=64, help="batch size")
parser.add_option(
    "--epoch_cb",
    dest="epoch_cb",
    type=int,
    default=100,
    help="number of epoch by which a callback (plot + dump weights) is executed",
)
parser.add_option(
    "--epochs", dest="epochs", type=int, default=10000, help="total number of epochs"
)
parser.add_option("--debug", action="store_true", dest="debug", default=False)
(opt, args) = parser.parse_args()
opt_dict = vars(opt)


if opt.vae_cfg is None:
    opt.vae_cfg = "./../out/vae/vae.cfg.p"
vae_cfg = pickle.load(open(opt.vae_cfg, "rb"))

if opt.vae_weights is None:
    opt.vae_weights = "./../out/vae/weights/weights.00000.pt"

if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# output dir
wdir = os.path.join(opt.outdir, "weights")
fdir = os.path.join(opt.outdir, "plots")
if not os.path.exists(wdir):
    os.makedirs(wdir)
if not os.path.exists(fdir):
    os.makedirs(fdir)

# copy code to output folder
export_scripts(os.path.join(opt.outdir, "scripts"))

# create logfile
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(opt.outdir, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("opt = %s", opt)


def main():

    torch.manual_seed(opt.seed)

    if opt.debug:
        pdb.set_trace()

    # load data
    img, obj, view = read_face_data(opt.data)  # image, object, and view
    train_data = FaceDataset(img["train"], obj["train"], view["train"])
    val_data = FaceDataset(img["val"], obj["val"], view["val"])
    train_queue = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    val_queue = DataLoader(val_data, batch_size=opt.bs, shuffle=False)

    # longint view and object repr
    Dt = Variable(obj["train"][:, 0].long(), requires_grad=False).cuda()
    Wt = Variable(view["train"][:, 0].long(), requires_grad=False).cuda()
    Dv = Variable(obj["val"][:, 0].long(), requires_grad=False).cuda()
    Wv = Variable(view["val"][:, 0].long(), requires_grad=False).cuda()

    # define VAE and optimizer
    vae = FaceVAE(**vae_cfg).to(device)
    RV = torch.load(opt.vae_weights)
    vae.load_state_dict(RV)
    vae.to(device)

    # define gp
    P = sp.unique(obj["train"]).shape[0]
    Q = sp.unique(view["train"]).shape[0]
    vm = Vmodel(P, Q, opt.xdim, Q).cuda()
    gp = GP(n_rand_effs=1).to(device)
    gp_params = nn.ParameterList()
    gp_params.extend(vm.parameters())
    gp_params.extend(gp.parameters())

    # define optimizers
    vae_optimizer = optim.Adam(vae.parameters(), lr=opt.vae_lr)
    gp_optimizer = optim.Adam(gp_params, lr=opt.gp_lr)

    if opt.debug:
        pdb.set_trace()

    history = {}
    for epoch in range(opt.epochs):

        # 1. encode Y in mini-batches
        Zm, Zs = encode_Y(vae, train_queue)

        # 2. sample Z
        Eps = Variable(torch.randn(*Zs.shape), requires_grad=False).cuda()
        Z = Zm + Eps * Zs

        # 3. evaluation step (not needed for training)
        Vt = vm(Dt, Wt).detach()
        Vv = vm(Dv, Wv).detach()
        rv_eval, imgs, covs = eval_step(vae, gp, vm, val_queue, Zm, Vt, Vv)

        # 4. compute first-order Taylor expansion coefficient
        Zb, Vbs, vbs, gp_nll = gp.taylor_coeff(Z, [Vt])
        rv_eval["gp_nll"] = float(gp_nll.data.mean().cpu()) / vae.K

        # 5. accumulate gradients over mini-batches and update params
        rv_back = backprop_and_update(
            vae,
            gp,
            vm,
            train_queue,
            Dt,
            Wt,
            Eps,
            Zb,
            Vbs,
            vbs,
            vae_optimizer,
            gp_optimizer,
        )
        rv_back["loss"] = (
            rv_back["recon_term"] + rv_eval["gp_nll"] + rv_back["pen_term"]
        )

        smartAppendDict(history, rv_eval)
        smartAppendDict(history, rv_back)
        smartAppend(history, "vs", gp.get_vs().data.cpu().numpy())

        logging.info(
            "epoch %d - tra_mse_val: %f - train_mse_out: %f"
            % (epoch, rv_eval["mse_val"], rv_eval["mse_out"])
        )

        # callback?
        if epoch % opt.epoch_cb == 0:
            logging.info("epoch %d - executing callback" % epoch)
            ffile = os.path.join(opt.outdir, "plot.%.5d.png" % epoch)
            callback_gppvae(epoch, history, covs, imgs, ffile)


def encode_Y(vae, train_queue):

    vae.eval()

    with torch.no_grad():

        n = train_queue.dataset.Y.shape[0]
        Zm = Variable(torch.zeros(n, vae_cfg["zdim"]), requires_grad=False).cuda()
        Zs = Variable(torch.zeros(n, vae_cfg["zdim"]), requires_grad=False).cuda()

        for batch_i, data in enumerate(train_queue):
            y = data[0].cuda()
            idxs = data[-1].cuda()
            zm, zs = vae.encode(y)
            Zm[idxs], Zs[idxs] = zm.detach(), zs.detach()

    return Zm, Zs


def eval_step(vae, gp, vm, val_queue, Zm, Vt, Vv):

    rv = {}

    with torch.no_grad():

        _X = vm.x().data.cpu().numpy()
        _W = vm.v().data.cpu().numpy()
        covs = {"XX": sp.dot(_X, _X.T), "WW": sp.dot(_W, _W.T)}
        rv["vars"] = gp.get_vs().data.cpu().numpy()
        # out of sample
        vs = gp.get_vs()
        U, UBi, _ = gp.U_UBi_Shb([Vt], vs)
        Kiz = gp.solve(Zm, U, UBi, vs)
        Zo = vs[0] * Vv.mm(Vt.transpose(0, 1).mm(Kiz))
        mse_out = Variable(torch.zeros(Vv.shape[0], 1), requires_grad=False).cuda()
        mse_val = Variable(torch.zeros(Vv.shape[0], 1), requires_grad=False).cuda()
        for batch_i, data in enumerate(val_queue):
            idxs = data[-1].cuda()
            Yv = data[0].cuda()
            Zv = vae.encode(Yv)[0].detach()
            Yr = vae.decode(Zv)
            Yo = vae.decode(Zo[idxs])
            mse_out[idxs] = (
                ((Yv - Yo) ** 2).view(Yv.shape[0], -1).mean(1)[:, None].detach()
            )
            mse_val[idxs] = (
                ((Yv - Yr) ** 2).view(Yv.shape[0], -1).mean(1)[:, None].detach()
            )
            # store a few examples
            if batch_i == 0:
                imgs = {}
                imgs["Yv"] = Yv[:24].data.cpu().numpy().transpose(0, 2, 3, 1)
                imgs["Yr"] = Yr[:24].data.cpu().numpy().transpose(0, 2, 3, 1)
                imgs["Yo"] = Yo[:24].data.cpu().numpy().transpose(0, 2, 3, 1)
        rv["mse_out"] = float(mse_out.data.mean().cpu())
        rv["mse_val"] = float(mse_val.data.mean().cpu())

    return rv, imgs, covs


def backprop_and_update(
    vae, gp, vm, train_queue, Dt, Wt, Eps, Zb, Vbs, vbs, vae_optimizer, gp_optimizer
):

    rv = {}

    vae_optimizer.zero_grad()
    gp_optimizer.zero_grad()
    vae.train()
    gp.train()
    vm.train()
    for batch_i, data in enumerate(train_queue):

        # subset data
        y = data[0].cuda()
        eps = Eps[data[-1]]
        _d = Dt[data[-1]]
        _w = Wt[data[-1]]
        _Zb = Zb[data[-1]]
        _Vbs = [Vbs[0][data[-1]]]

        # forward vae
        zm, zs = vae.encode(y)
        z = zm + zs * eps
        yr = vae.decode(z)
        recon_term, mse = vae.nll(y, yr)

        # forward gp
        _Vs = [vm(_d, _w)]
        gp_nll_fo = gp.taylor_expansion(z, _Vs, _Zb, _Vbs, vbs) / vae.K

        # penalization
        pen_term = -0.5 * zs.sum(1)[:, None] / vae.K

        # loss and backward
        loss = (recon_term + gp_nll_fo + pen_term).sum()
        loss.backward()

        # store stuff
        _n = train_queue.dataset.Y.shape[0]
        smartSum(rv, "mse", float(mse.data.sum().cpu()) / _n)
        smartSum(rv, "recon_term", float(recon_term.data.sum().cpu()) / _n)
        smartSum(rv, "pen_term", float(pen_term.data.sum().cpu()) / _n)

    vae_optimizer.step()
    gp_optimizer.step()

    return rv


if __name__ == "__main__":
    main()
