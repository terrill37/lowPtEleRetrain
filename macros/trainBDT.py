import uproot
import glob
import pandas as pd
from tqdm import tqdm

import numpy as np

def get_df(root_file_name, unnecessary_columns):
    rootFile = uproot.open(root_file_name)
    #if len(rootFile.allkeys())==0: return pd.DataFrame()
    df = rootFile["ntuplizer/tree"].arrays(library="pd")
    return df.drop(unnecessary_columns, axis=1)


def get_label(name):
    if name == 0: return "background"
    else: return "signal"

def plot_electrons(df, column, bins, logscale=False, ax=None, title=None):
    if ax is None: ax = plt.gca()
    for name, group in df.groupby("matchedToGenEle"): 
        group[column].hist(bins=bins, histtype="step", label=get_label(name), ax=ax, density=True)
    ax.set_ylabel("density")
    ax.set_xlabel(column)
    ax.legend()
    ax.set_title(title)
    if logscale: ax.set_yscale("log", nonposy='clip')

def plotting(df, branch, output):
    import matplotlib.pyplot as plt
    import mplhep

    fig, axes = plt.subplots(1,1, figsize=(5,5))
    plot_electrons(df, branch, np.linspace(0, 30, 60), ax=axes)
    #plot_electrons(df, "scl_eta", np.linspace(-2.5,2.5,50), ax=axes[1])
    plt.savefig(output+'/'+branch+'.png')


def main():
    import argparse
    p=argparse.ArgumentParser(description="Train low pt electron BDT")
    p.add_argument('ntupleFiles', nargs='*', help="<REQUIRED> ntuple location")
    p.add_argument('--output', required=True, help="<REQUIRED> output location for plots and stuff")
    p.add_argument('--debug', default=False, action="store_true", help="Turn on debug prints")

    args=p.parse_args()
    
    unnecessary = ['ele_ID', 'nEvent', 'nRun', 'nLumi']

    df = pd.concat((get_df(f,unnecessary) for f in tqdm(args.ntupleFiles)), ignore_index=True)

    if args.debug: print(df.columns)

    #Categorizing electrons as signal or background
    #use matchToGenEle branch
    #UNMATCHED = 0
    #electrons from taus = 2 (drop from df)
    df = df.query("matchedToGenEle != 2")

    #can combine unmatched (0) and non-prompt (3) (maybe want to be agnostic here?)
    #df.loc[df["matchedToGenEle"] != 1, "matchedToGenEle"] = 0

    #drop electrons outside detector acceptance
    df = df.query("abs(scl_eta) < 2.5")

    #keep electrons with pt >= 1GeV
    df = df.query("ele_pt >= 1")

    #plot histograms with distributions for signal and background electrons
    if args.debug:
        for col in df.columns:
            plotting(df, col, args.output)



if __name__=="__main__": main()

