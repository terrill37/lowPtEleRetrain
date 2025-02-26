import uproot
import glob
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import xgboost as xgb
import xgboost2tmva

import numpy as np

import ROC

binning={
    "ele_pt"   : np.linspace(0,30,60),
    "scl_eta"  : np.linspace(-2.5, 2.5, 50),
    "ele_isEB" : 2,
    "ele_isEE" : 2
}

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

def plotting(df, branch, output, debug=False):
    import matplotlib.pyplot as plt
    import mplhep
    
    if debug: print(branch)
    if branch in binning.keys(): bins=binning[branch]
    else: bins=100
    fig, axes = plt.subplots(1,1, figsize=(5,5))
    plot_electrons(df, branch, bins, ax=axes)
    plt.savefig(output+'/'+branch+'.png')

def main():
    import argparse
    p=argparse.ArgumentParser(description="Train low pt electron BDT")
    p.add_argument('ntupleFiles', nargs='*', help="<REQUIRED> ntuple location")
    p.add_argument('--output', required=True, help="<REQUIRED> output location for plots and stuff")
    p.add_argument('--debug', default=False, action="store_true", help="Turn on debug prints")

    args=p.parse_args()
    
    unnecessary = ['nEvent', 'nRun', 'nLumi']

    df = pd.concat((get_df(f,unnecessary) for f in tqdm(args.ntupleFiles)), ignore_index=True)

    if args.debug: print(df.columns)

    #Categorizing electrons as signal or background
    #use matchToGenEle branch
    #UNMATCHED = 0
    #electrons from taus = 2 (drop from df)
    df = df.query("matchedToGenEle != 2")

    #can combine unmatched (0) and non-prompt (3) (maybe want to be agnostic here?)
    df.loc[df["matchedToGenEle"] > 0, "matchedToGenEle"] = 1

    #drop electrons outside detector acceptance
    df = df.query("abs(scl_eta) < 2.5")

    #keep electrons with pt >= 1GeV
    df = df.query("ele_pt >= 1")

    #list of features to be used
    #FIXME need to get: 
    # trk_nhits
    # trk_chi2red 
    # gsf_nhits (possibly 'ele_gsfhits'?)
    # match eclu_EoverP (possibly 'ele_IoEmIop'?)
    # trk_p
    # gsf_mode_p
    # core_shFracHits
    # gsf_bdtout1
    # gsf_dr
    # trk_dr
    # sc_clus1_nxtal
    # sc_clus1_dphi
    # sc_clus2_dphi
    # sc_clus1_deta
    # sc_clus2_deta
    # sc_clus1_E
    # sc_clus2_E
    # sc_clus1_E_ov_p
    # sc_clus2_E_ov_p
    features = ["rho", "scl_eta", "ele_oldr9", "ele_scletawidth", "ele_sclphiwidth", "ele_oldhe",
                "ele_gsfchi2", "ele_fbrem", "ele_ep", "ele_deltaetain", "ele_deltaphiin",
                "ele_deltaetaseed", "scl_E", "ele_sclNclus"]
    
    #plot histograms with distributions for signal and background electrons
    if args.debug: 
        for feature in features:
            if df[feature].dtype is bool: continue
            plotting(df, feature, args.output, args.debug)
    
    n_boost_rounds = 50
    xgboost_params = {'eval_metric' : 'auc',
                      'objective'   : 'binary:logitraw'}

    category_titles = ["EB"]

    #This is where the retraining happens
    for i, cat in enumerate(category_titles):
        #get the features (either endcap or barrel)
        #features = features_EE if 'EE' in category else features_EB
        #Currently using inclusive training of barrel and endcap
        
        #get features from and the target from the data frame
        X = df[features]
        Y = df["matchedToGenEle"]

        #split X and Y up into train and test samples
        X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

        idx_train = X_train.index
        idx_test  = X_test.index

        #XGBoost has its own format, need to create these structures
        dmatrix_train = xgb.DMatrix(X_train.copy(), label=np.copy(Y_train))
        dmatrix_test  = xgb.DMatrix(X_test.copy(),  label=np.copy(Y_test))

        #Get the number of positive and negative training examples in this category
        n_pos = np.sum(Y_train==1)
        n_neg = np.sum(Y_train==0)

        print(cat+":")
        print(rf"training on {n_pos} signal and {n_neg} background electrons.")

        #set hyperparameter: scale_pos_weight
        #corresponds to a weight given to every positive sample
        #set to n_neg/n_pos for imbalanced datasets to balance total contributions
        #of the positive and negative classes in the loss function
        xgboost_params["scale_pos_weight"] = 1. * n_neg/n_pos

        #train the model
        model = xgb.train(xgboost_params, dmatrix_train,
                          num_boost_round = n_boost_rounds,
                          evals=[(dmatrix_train, 'train'),
                                 (dmatrix_test,  'test')],
                          early_stopping_rounds=10,
                          verbose_eval=False)
        
        best_iteration = model.best_iteration + 1
        if best_iteration<n_boost_rounds: 
            print(f"early stopping after {best_iteration} boosting rounds")

        print("")

        xgboost2tmva.convert_model(model.get_dump(), input_variables=[(f,'F') for f in features],
                                   output_xml=f'electron_id_{i}.xml')
        
        model.save_model(f"electron_id_{i}.bin")

        df.loc[idx_train, "score"] = model.predict(dmatrix_train)
        df.loc[idx_test,  "score"] = model.predict(dmatrix_test)
    
        df["test"] = False
        df.loc[idx_train, "test"] = False
        df.loc[idx_test,  "test"] = True
        
    #Now make a ROC curve
    print("now making roc curve")
    df_train = df.query("not test")
    df_test  = df.query("test")
    
    labels=["train", "test", "Run2"]
    fprs,tprs,aucs=[],[],[]

    fpr,tpr,roc_auc = ROC.getROCs(isObj=df_train["matchedToGenEle"], scores=df_train["score"])
    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(roc_auc)
    
    fpr,tpr,roc_auc = ROC.getROCs(isObj=df_test["matchedToGenEle"], scores=df_test["score"])
    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(roc_auc)
    
    fpr,tpr,roc_auc = ROC.getROCs(isObj=df["matchedToGenEle"], scores=df["ele_ID"])
    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(roc_auc)
    
    ROC.plotROCs(fprs,tprs,aucs,args.output,labels)


if __name__=="__main__": main()

