import uproot
import glob
import pandas as pd
from tqdm import tqdm

import os

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import xgboost2tmva

import numpy as np

import ROC

features = ["rho", "scl_eta", "ele_oldr9", "ele_scletawidth", "ele_sclphiwidth", "ele_oldhe",
            "ele_fbrem", "ele_ep", "ele_deltaetain", "ele_deltaphiin",
            "ele_deltaetaseed", "scl_E", "ele_sclNclus",
            "trk_p", "trk_chi2red", "trk_dr", "trk_nhits",
            "gsf_mode_p", "gsf_dr", "gsf_nhits", "gsf_chi2red",
            "sc_clus1_nxtal", "sc_clus1_dphi", "sc_clus1_deta", "sc_clus1_E", 
            "sc_clus1_EoverP", "sc_clus2_dphi", "sc_clus2_deta", "sc_clus2_E", 
            "sc_clus2_EoverP",
            "ele_shFracHits", "ele_eclu_EoverP", "ele_seedBDT",
            "ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi", "ele_oldcircularity"]

unnecessary = ['nEvent', 'nRun', 'nLumi']

binning={
    "ele_pt"   : np.linspace(0,15,30),
    "scl_eta"  : np.linspace(-2.5, 2.5, 50),
    "ele_isEB" : 2,
    "ele_isEE" : 2,
    "scl_E"    : np.linspace(0, 60, 60),
    "trk_chi2red" : np.linspace(-1, 20, 42),
    "trk_dr"   : np.linspace(-0.05, 0.5, 35),
    "trk_nhits": np.linspace(-1, 10, 11),
    "trk_p"    : np.linspace(-10, 10, 20),
    "ele_deltaetain" : np.linspace(0, 0.06, 30),
    "ele_deltaetaseed" : np.linspace(0, 0.2, 40),
    "ele_deltaphiin" : np.linspace(0, 0.6, 30),
    "ele_ep" : np.linspace(0, 6, 30),
    "ele_fbrem" : np.linspace(-1, 1, 50),
    "ele_gsfchi2" : np.linspace(0, 30, 60),
    "ele_oldhe" : np.linspace(0, 20, 40),
    "ele_oldr9" : np.linspace(0, 2, 20),
    "ele_sclNclus" : np.linspace(0, 15, 15),
    "ele_scletawidth": np.linspace(0, 1.2, 48),
    "ele_sclphiwidth": np.linspace(0, 1.2, 48),
    "gsf_chi2red": np.linspace(0,30,60),
    "gsf_dr" : np.linspace(-0.05,0.5,35),
    "gsf_mode_p" : np.linspace(0,50,100),
    "gsf_nhits":np.linspace(0,30,30),
    "rho" : np.linspace(0,60,60),
    "sc_clus1_E" : np.linspace(0,15,30),
    "sc_clus2_E" : np.linspace(0,15,30),
    "sc_clus1_EoverP" : np.linspace(0,3,25),
    "sc_clus2_EoverP" : np.linspace(0,3,25),
    "sc_clus1_deta" : np.linspace(-1.5,1.5,20),
    "sc_clus2_deta" : np.linspace(-1.5,1.5,20),
    "sc_clus1_dphi" : np.linspace(-1.5,1.5,20),
    "sc_clus2_dphi" : np.linspace(-1.5,1.5,20),
    "sc_clus1_nxtal": np.linspace(0,10,10),
    "ele_eclu_EoverP": np.linspace(-0.05, 0.05, 25)
}

def get_tree(root_file_name, unnecessary_columns):
    rootFile = uproot.open(root_file_name)
    #if len(rootFile.allkeys())==0: return pd.DataFrame()
    tree = rootFile["ntuplizer/tree"] #.arrays(library="pd")
    return tree #df.drop(unnecessary_columns, axis=1)

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

def plotting(df, branch, output, debug=False):
    import matplotlib.pyplot as plt
    import mplhep
    
    if debug: print(branch)
    if branch in binning.keys(): bins=binning[branch]
    else: bins=100
    fig, axes = plt.subplots(1,1, figsize=(5,5))
    plot_electrons(df, branch, bins, ax=axes)
    plt.savefig(output+'/'+branch+'.png')
    plt.close()

def main():
    import argparse
    p=argparse.ArgumentParser(description="make ROC curves from input ntuple")
    p.add_argument('ntupleFile', help="<REQUIRED> ntuple location")
    p.add_argument('--output', required=True, help="<REQUIRED> output location for plot")
    p.add_argument('--debug', default=False, action="store_true", help="Turn on debug prints")
    p.add_argument('--newModels', nargs='*', default=None, help="New Model")

    args=p.parse_args()

    ntupleFile = uproot.open(args.ntupleFile)

    if args.debug: print(ntupleFile.keys())

    tree = ntupleFile["ntuplizer/tree"]
    bdtScores_run2 = pd.DataFrame()
    seedBDT        = pd.DataFrame()

    for chunk in tree.iterate(features+['matchedToGenEle', 'ele_pt', 'ele_ID'], library='pd', step_size=100_000):
        df = chunk.query("matchedToGenEle != 2")
        df.loc[df["matchedToGenEle"] > 0, "matchedToGenEle"] = 1
        df = df.query("ele_pt >= 1 & ele_pt < 10 & abs(scl_eta) < 2.5")

        isE = df["matchedToGenEle"]

        bdtScores_run2 = pd.concat([bdtScores_run2, df[["matchedToGenEle", "ele_ID"]]])
        seedBDT        = pd.concat([seedBDT,        df[["matchedToGenEle", "ele_seedBDT"]]])

        if not args.newModel is None:
            booster=xgb.Booster()
            booster.load_model('electron_id_0.bin')
            
            df = df[features]
            df=df.astype(np.float32)
            df=pd.concat([isE, df], axis=1)
            df.to_csv('data.csv', mode="a", index=False, header=False)
    
    if not args.newModel is None:
        dmat = xgb.DMatrix('data.csv?format=csv&label_column=0#dtrain.cache')
        preds = booster.predict(dmat)
        df_preds = pd.DataFrame({"matchedToGenEle": dmat.get_label(),
                                 "bdtScore"       : preds})

        os.remove('data.csv')
    
    fprs, tprs, aucs, labels = [], [], [], []

    fpr, tpr, roc_auc = ROC.getROCs(bdtScores_run2["matchedToGenEle"], bdtScores_run2["ele_ID"])
    fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc); labels.append("2020Nov28")
    ROC.plotROCs([fpr], [tpr], [roc_auc], args.output, labels=["2020Nov28"])

    fpr, tpr, roc_auc = ROC.getROCs(seedBDT["matchedToGenEle"], seedBDT["ele_seedBDT"])
    fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc); labels.append("electronID(\"unbiased\")")
    ROC.plotROCs([fpr], [tpr], [roc_auc], args.output, labels=["electronID(\"unbiased\")"], figName="SEED_ROC")

    fpr, tpr, roc_auc = ROC.getROCs(df_preds["matchedToGenEle"], df_preds["bdtScore"])
    fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc); labels.append("Run3")
    ROC.plotROCs([fpr], [tpr], [roc_auc], args.output, labels=["Run3"], figName="Run3")

    ROC.plotROCs(fprs, tprs, aucs, args.output, labels=labels, figName="rocComp")
    
    #for feat in features:
    #    if tree[feat].dtype is bool: continue
    #    plotting(tree, feat, args.output, args.debug)


if __name__=="__main__": main()
 
