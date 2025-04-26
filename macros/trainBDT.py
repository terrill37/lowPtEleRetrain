import uproot
import glob
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import xgboost2tmva

import numpy as np

import os

import ROC

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
    p=argparse.ArgumentParser(description="Train low pt electron BDT")
    p.add_argument('ntupleFiles', nargs='*', help="<REQUIRED> ntuple location")
    p.add_argument('--output', required=True, help="<REQUIRED> output location for plots and stuff")
    p.add_argument('--modelName', required=True, help="<REQUIRED> name of model")
    p.add_argument('--debug', default=False, action="store_true", help="Turn on debug prints")

    args=p.parse_args()
    
    unnecessary = ['nEvent', 'nRun', 'nLumi']
    
    #list of features to be used in the training FIXME move to separate file to be used elsewhere
    features = ["rho", "scl_eta", "ele_oldr9", "ele_scletawidth", "ele_sclphiwidth", "ele_oldhe",
                "ele_fbrem", "ele_ep", "ele_deltaetain", "ele_deltaphiin",
                "ele_deltaetaseed", "scl_E", "ele_sclNclus",
                "trk_p", "trk_chi2red", "trk_dr", "trk_nhits",
                "gsf_mode_p", "gsf_dr", "gsf_nhits", "gsf_chi2red",
                "sc_clus1_nxtal", "sc_clus1_dphi", "sc_clus1_deta", "sc_clus1_E", "sc_clus1_EoverP",
                "sc_clus2_dphi", "sc_clus2_deta", "sc_clus2_E", "sc_clus2_EoverP",
                "ele_shFracHits", "ele_eclu_EoverP", "ele_seedBDT",
                "ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi", "ele_oldcircularity"]

    #only need to grab features plus flag
    necessary_cols = features + ['matchedToGenEle', 'ele_pt', 'ele_ID']
    
    dmatrix_train=None
    dtest_list,bdt_results_train, bdt_results_test=[],[],[]
    old_results=[]
    batch_size=100_000
    header=True
    for chunk in get_tree(args.ntupleFiles[0], unnecessary).iterate(necessary_cols, step_size=batch_size, library='pd'):
        #Categorizing electrons as signal or background
        #use matchToGenEle branch
        #UNMATCHED = 0
        #electrons from taus = 2 (drop from df)
        df = chunk.query("matchedToGenEle != 2")

        #can combine unmatched (0) and non-prompt (3) (maybe want to be agnostic here?)
        df.loc[df["matchedToGenEle"] > 0, "matchedToGenEle"] = 1

        #drop electrons outside detector acceptance
        df = df.query("ele_pt >= 1 & ele_pt < 10 & abs(scl_eta) < 2.5")
        
        if df.empty: continue
        
        isE     = df["matchedToGenEle"]
        df_feat = df[features]
        run2IDs  = df[["matchedToGenEle", "ele_ID", "ele_seedBDT"]]
    
        #train test split
        df_train,df_test,isE_train,isE_test = train_test_split(df_feat, isE, test_size=0.2, random_state=45)
        
        df_train = pd.concat([isE_train, df_train], axis=1)
        df_test  = pd.concat([isE_test,  df_test],  axis=1)

        df_train.to_csv('data_train'+args.modelName+'.csv', mode="a", index=False, header=False)
        df_test.to_csv('data_test'+args.modelName+'.csv', mode="a", index=False, header=False) 
        run2IDs.to_csv('data_run2'+args.modelName+'.csv', mode="a", index=False, header=header)
        header=False
    
    #create dmatrices
    dmat_train = xgb.DMatrix('data_train'+args.modelName+'.csv?format=csv&label_column=0#dtrain.cache')
    dmat_test  = xgb.DMatrix('data_test'+args.modelName+'.csv?format=csv&label_column=0#dtest.cache')

    ###XGBoost settings
    n_boost_rounds = 100
    xgboost_params = {'eval_metric' : 'auc',
                      'objective'   : 'binary:logitraw'}
    #Get the number of positive and negative training examples in this category
    Y_train = dmat_train.get_label()
    n_pos = np.sum(Y_train==1)
    n_neg = np.sum(Y_train==0)

    print(rf"training on {n_pos} signal and {n_neg} background electrons.")
    del Y_train

    #set hyperparameter: scale_pos_weight
    #corresponds to a weight given to every positive sample
    #set to n_neg/n_pos for imbalanced datasets to balance total contributions
    #of the positive and negative classes in the loss function
    xgboost_params["scale_pos_weight"] = 1. * n_neg/n_pos

    #train the model
    model = xgb.train(xgboost_params, dmat_train,
                      num_boost_round = n_boost_rounds,
                      evals=[(dmat_train, 'train'),
                             (dmat_test,  'test')],
                      early_stopping_rounds = 100,
                      verbose_eval=False)
    
    best_iteration = model.best_iteration + 1
    if best_iteration<n_boost_rounds: print(f"early stopping after {best_iteration} boosting rounds")

    print("")

    #xgboost2tmva.convert_model(model.get_dump(), input_variables=[(f,'F') for f in features],
    #                           output_xml='electron_id.xml')

    model.save_model(f"electron_id_{args.modelName}.bin")

    bdt_results_train = pd.DataFrame({"matchedToGenEle" : dmat_train.get_label(), 
                                           "score" : model.predict(dmat_train)})
    bdt_results_test  = pd.DataFrame({"matchedToGenEle"  : dmat_test.get_label(),
                                          "score"  : model.predict(dmat_test)})
    run2_results      = pd.read_csv("data_run2"+args.modelName+".csv")

    fprs, tprs, aucs, labels = [], [], [], []
    
    fpr,tpr,roc_auc = ROC.getROCs(bdt_results_train["matchedToGenEle"],bdt_results_train["score"])
    fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc); labels.append("Run3 Train")

    fpr,tpr,roc_auc = ROC.getROCs(bdt_results_test["matchedToGenEle"],bdt_results_test["score"])
    fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc); labels.append("Run3 Test")
        
    fpr,tpr,roc_auc = ROC.getROCs(run2_results["matchedToGenEle"],run2_results["ele_ID"])
    fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc); labels.append("2020Nov28")

    fpr,tpr,roc_auc = ROC.getROCs(run2_results["matchedToGenEle"],run2_results["ele_seedBDT"])
    fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc); labels.append("electronID(\"unbiased\")")

    ROC.plotROCs(fprs, tprs, aucs, args.output, labels=labels, figName="rocComp")

    os.remove('data_train.csv')
    os.remove('data_test.csv')
    os.remove('data_run2.csv')
         
    

if __name__=="__main__": main()

