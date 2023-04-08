import zipfile
import pandas as pd


with zipfile.ZipFile('/media/kuga/瓜果山/results/mlpf.zip', 'r').open('mlpf/D-END/Architecture/Architecture-ND00-2.pkl') as f:
    a = pd.read_pickle(f)

breakpoint()
