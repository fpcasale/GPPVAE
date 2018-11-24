import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import h5py
import scipy as sp
import os
import pdb


class GP(nn.Module):
    def __init__(self, n_rand_effs=1, vsum2one=True):

        super(GP, self).__init__()

        # store stuff
        self.n_rand_effs = n_rand_effs
        self.vsum2one = vsum2one

        # define variables
        n_vcs = n_rand_effs + 1
        self.lvs = nn.Parameter(torch.zeros([n_vcs]))

    def U_UBi_Shb(self, Vs, vs):

        # compute U and V
        V = torch.cat([torch.sqrt(vs[i]) * V for i, V in enumerate(Vs)], 1)
        U = V / torch.sqrt(vs[-1])
        eye = torch.eye(U.shape[1]).cuda()
        B = torch.mm(torch.transpose(U, 0, 1), U) + eye
        # cholB = torch.potrf(B, upper=False)
        # Bi = torch.potri(cholB, upper=False)
        Ub, Shb, Vb = torch.svd(B)
        # Bi = (Vb / Shb).mm(torch.transpose(Vb, 0, 1))
        Bi = torch.inverse(B)
        UBi = torch.mm(U, Bi)

        return U, UBi, Shb

    def solve(self, X, U, UBi, vs):

        UX = U.transpose(0, 1).mm(X)
        UBiUX = UBi.mm(UX)
        RV = (X - UBiUX) / vs[-1]

        return RV

    def get_vs(self):
        if self.vsum2one:
            rv = F.softmax(self.lvs, 0)
        else:
            rv = torch.exp(self.lvs) / float(n_vcs)
        return rv

    def taylor_coeff(self, X, Vs):

        # solve
        vs = self.get_vs()
        U, UBi, Shb = self.U_UBi_Shb(Vs, vs)
        Xb = self.solve(X, U, UBi, vs)

        # variables to fill
        Vbs = []
        vbs = Variable(torch.zeros([self.n_rand_effs + 1]), requires_grad=False).cuda()

        # compute Vbs and vbs
        for iv, V in enumerate(Vs):
            XbV = Xb.transpose(0, 1).mm(V)
            XbXbV = Xb.mm(XbV)
            KiV = self.solve(V, U, UBi, vs)
            Vb = vs[iv] * (X.shape[1] * KiV - XbXbV)
            Vbs.append(Vb)

            # compute vgbar
            vbs[iv] = -0.5 * torch.einsum("ij,ij->", [XbV, XbV])
            vbs[iv] += 0.5 * X.shape[1] * torch.einsum("ij,ij->", [V, KiV])

        # compute vnbar
        trKi = (X.shape[0] - torch.einsum("ni,ni->", [UBi, U])) / vs[-1]
        vbs[-1] = -0.5 * torch.einsum("ij,ij->", [Xb, Xb])
        vbs[-1] += 0.5 * X.shape[1] * trKi

        # compute negative log likelihood (nll)
        quad_term = torch.einsum("ij,ij->i", [X, Xb])[:, None]
        logdetK = Xb.shape[0] * Xb.shape[1] * torch.log(vs[-1])
        logdetK += Xb.shape[1] * torch.sum(torch.log(Shb))
        nll = 0.5 * quad_term + 0.5 * logdetK / X.shape[0]

        # detach all
        Xb = Xb.detach()
        Vbs = [Vb.detach() for Vb in Vbs]
        vbs = vbs.detach()
        nll = nll.detach()

        return Xb, Vbs, vbs, nll

    def nll(self, X, Vs):

        # solve
        vs = self.get_vs()
        U, UBi, Shb = self.U_UBi_Shb(Vs, vs)
        Xb = self.solve(X, U, UBi, vs)

        # compute negative log likelihood (nll)
        quad_term = torch.einsum("ij,ij->i", [X, Xb])[:, None]
        logdetK = Xb.shape[0] * Xb.shape[1] * torch.log(vs[-1])
        logdetK += Xb.shape[1] * torch.sum(torch.log(Shb))
        nll = 0.5 * quad_term + 0.5 * logdetK / X.shape[0]

        return nll

    def nll_ineff(self, X, Vs):
        vs = self.get_vs()
        V = torch.cat([torch.sqrt(vs[i]) * V for i, V in enumerate(Vs)], 1)
        K = V.mm(V.transpose(0, 1)) + vs[-1] * torch.eye(X.shape[0]).cuda()
        Uk, Shk, Vk = torch.svd(K)
        Ki = torch.inverse(K)
        # Ki = (Vk / Shk).mm(torch.transpose(Vk, 0, 1))
        Xb = Ki.mm(X)

        quad_term = torch.einsum("ij,ij->i", [X, Xb])[:, None]
        logdetK = X.shape[1] * torch.log(Shk).sum()
        nll = 0.5 * quad_term + 0.5 * logdetK / X.shape[0]

        return nll

    def taylor_expansion(self, X, Vs, Xb, Vbs, vbs):
        rv = torch.einsum("ij,ij->i", [Xb, X])[:, None]
        for V, Vb in zip(Vs, Vbs):
            rv += torch.einsum("ij,ij->i", [Vb, V])[:, None]
        vs = self.get_vs()
        rv += torch.einsum("i,i->", [vbs, vs]) / float(X.shape[0])
        return rv


if __name__ == "__main__":

    def generate_data(N, S, L):

        # generate genetics
        G = 1.0 * (sp.rand(N, S) < 0.2)
        G -= G.mean(0)
        G /= G.std(0) * sp.sqrt(G.shape[1])

        # generate latent phenotypes
        Zg = sp.dot(G, sp.randn(G.shape[1], L))
        Zn = sp.randn(N, L)

        # generate variance exapleind
        vg = sp.linspace(0.8, 0, L)

        # rescale and sum
        Zg *= sp.sqrt(vg / Zg.var(0))
        Zn *= sp.sqrt((1 - vg) / Zn.var(0))
        Z = Zg + Zn

        return Z, G

    torch.manual_seed(0)

    N = 1000
    S = 100
    L = 256

    Z, G = generate_data(N, S, L)
    Z = nn.Parameter(torch.tensor(Z.astype("float32")).cuda())
    G = nn.Parameter(torch.tensor(G.astype("float32")).cuda())

    pdb.set_trace()

    # define VAE and optimizer
    gp = GP(n_rand_effs=1).cuda()
    optimizer = optim.Adam(gp.parameters(), lr=1e-2)

    if 1:
        """ Chack computation of nll """
        # zero grad
        Xb, Vbs, vbs, nll = gp.taylor_coeff(Z, [G])
        nll0 = gp.nll_ineff(Z, [G])
        nll1 = gp.nll(Z, [G])
        print(((nll - nll0) ** 2).mean())
        print(((nll - nll1) ** 2).mean())
        pdb.set_trace()

    if 1:
        """ Check taylor expansion """
        Xb, Vbs, vbs, nll = gp.taylor_coeff(Z, [G])
        nll_fo = gp.taylor_expansion(Z, [G], Xb, Vbs, vbs)
        nll_fo.sum().backward()
        Zgrad1 = Z.grad.data.cpu().numpy()
        Ggrad1 = G.grad.data.cpu().numpy()
        for param in gp.parameters():
            vgrad1 = param.grad.cpu().numpy()

        gp.lvs.grad.data.zero_()
        Z.grad.data.zero_()
        G.grad.data.zero_()
        nll0 = gp.nll_ineff(Z, [G])
        nll0.sum().backward()
        Zgrad2 = Z.grad.data.cpu().numpy()
        Ggrad2 = G.grad.data.cpu().numpy()
        for param in gp.parameters():
            vgrad2 = param.grad.cpu().numpy()

        gp.lvs.grad.data.zero_()
        Z.grad.data.zero_()
        G.grad.data.zero_()
        nll1 = gp.nll(Z, [G])
        nll1.sum().backward()
        Zgrad3 = Z.grad.data.cpu().numpy()
        Ggrad3 = G.grad.data.cpu().numpy()
        for param in gp.parameters():
            vgrad3 = param.grad.cpu().numpy()

        print(((Zgrad2 - Zgrad1) ** 2).mean())
        print(((Ggrad2 - Ggrad1) ** 2).mean())
        print(((vgrad2 - vgrad1) ** 2).mean())

        print(((Zgrad3 - Zgrad1) ** 2).mean())
        print(((Ggrad3 - Ggrad1) ** 2).mean())
        print(((vgrad3 - vgrad1) ** 2).mean())

        pdb.set_trace()

    if 1:
        """ goes for training """
        history = {}
        epochs = 200
        for epoch in range(epochs):
            print("Epoch %.4d" % epoch)

            optimizer.zero_grad()
            Xb, Vbs, vbs, nll = gp.taylor_coeff(Z, [G])
            nll_fo = gp.taylor_expansion(Z, [G], Xb, Vbs, vbs)
            nll_fo.backward()
            optimizer.step()

            # append stuff
            smartAppend(history, "nll", float(nll.data.cpu()))
            smartAppend(history, "vs", gp.get_vs().data.cpu().numpy())
        history["vs"] = sp.array(history["vs"])

        pl.subplot(221)
        pl.plot(history["nll"])
        pl.subplot(222)
        pl.plot(history["vs"][:, 0], "r")
        pl.plot(history["vs"][:, 1], "k")
        pl.tight_layout()
        pl.savefig("conv_gp.png")
        pdb.set_trace()
        pl.close()
