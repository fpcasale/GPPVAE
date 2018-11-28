import sys
from utils import download, unzip
import sys
import glob
import scipy as sp
import scipy.linalg as la
from scipy.ndimage import imread
from scipy.misc import imresize
import os
import h5py

# where have been downloaded
data_dir = sys.argv[1]

def main():

    # 1. download and unzip data
    download_data(data_dir)

    # 2. load data
    RV = import_data()

    # 3. split train, validation and test
    RV = split_data(RV)

    # 4. export
    out_file = os.path.join(data_dir, "data_faces.h5")
    fout = h5py.File(out_file, "w")
    for key in RV.keys():
        fout.create_dataset(key, data=RV[key])
    fout.close()


def unzip_data():

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    fnames = [
        "asian.zip",
        "africanamerican.zip",
        "caucasian.zip",
        "hispanic.zip",
        "multiracial.zip",
    ]

    for fname in fnames:
        print(".. unzipping")
        unzip(os.path.join(data_dir, fname), data_dir)


def import_data(size=128):

    files = []
    orients = ["00F", "30L", "30R", "45L", "45R", "60L", "60R", "90L", "90R"]
    for orient in orients:
        _files = glob.glob(os.path.join(data_dir, "*/*_%s.jpg" % orient))
        files = files + _files
    files = sp.sort(files)

    D1id = []
    D2id = []
    Did = []
    Rid = []
    Y = sp.zeros([len(files), size, size, 3], dtype=sp.uint8)
    for _i, _file in enumerate(files):
        y = imread(_file)
        y = imresize(y, size=[size, size], interp="bilinear")
        Y[_i] = y
        fn = _file.split(".jpg")[0]
        fn = fn.split("/")[-1]
        did1, did2, rid = fn.split("_")
        Did.append(did1 + "_" + did2)
        Rid.append(rid)
    Did = sp.array(Did, dtype="|S100")
    Rid = sp.array(Rid, dtype="|S100")

    RV = {"Y": Y, "Did": Did, "Rid": Rid}
    return RV


def split_data(RV):

    sp.random.seed(0)
    n_train = int(4 * RV["Y"].shape[0] / 5.0)
    n_test = int(1 * RV["Y"].shape[0] / 10.0)
    idxs = sp.random.permutation(RV["Y"].shape[0])
    idxs_train = idxs[:n_train]
    idxs_test = idxs[n_train : (n_train + n_test)]
    idxs_val = idxs[(n_train + n_test) :]

    Itrain = sp.in1d(sp.arange(RV["Y"].shape[0]), idxs_train)
    Itest = sp.in1d(sp.arange(RV["Y"].shape[0]), idxs_test)
    Ival = sp.in1d(sp.arange(RV["Y"].shape[0]), idxs_val)

    out = {}
    for key in RV.keys():
        out["%s_train" % key] = RV[key][Itrain]
        out["%s_val" % key] = RV[key][Ival]
        out["%s_test" % key] = RV[key][Itest]

    return out


if __name__ == "__main__":

    main()
