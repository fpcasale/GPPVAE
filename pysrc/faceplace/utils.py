import scipy as sp
import sys
import warnings
import shutil
import os
import glob

PY2 = sys.version_info < (3,)


def smartAppend(table, name, value):
    if name not in table.keys():
        table[name] = []
    table[name].append(value)


def smartAppendDict(table, table_):
    for key in table_.keys():
        smartAppend(table, key, table_[key])


def smartSum(table, name, value):
    if name not in table.keys():
        table[name] = value
    else:
        table[name] += value


def smartDumpDictHdf5(RV, o):
    for key in RV.keys():
        if type(RV[key]) == dict:
            g = o.create_group(key)
            smartDumpDictHdf5(RV[key], g)
        else:
            o.create_dataset(
                name=key, data=sp.array(RV[key]), chunks=True, compression="gzip"
            )


def download(url, dest=None):
    import os

    if PY2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    if dest is None:
        dest = os.getcwd()

    filepath = os.path.join(dest, _filename(url))
    urlretrieve(url, filepath)


def _filename(url):
    import os

    if PY2:
        from urlparse import urlparse
    else:
        from urllib.parse import urlparse

    a = urlparse(url)
    return os.path.basename(a.path)


def unzip(filepath, outdir):
    import zipfile

    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(outdir)


def export_scripts(path):

    if not os.path.exists(path):
        os.mkdir(path)

    scripts = glob.glob("*.py")
    for script in scripts:
        shutil.copyfile(script, os.path.join(path, script))
