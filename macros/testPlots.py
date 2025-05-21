import uproot
import glob
import pandas as pd
from tqdm import tqdm

import os
import subprocess as sp

#from sklearn.model_selection import train_test_split
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
    "ele_seedBDT"          : (80, -20, 20),
    "ele_seedBiasedBDT"    : (80, -20, 20),
    "ele_pt"               : (30, 0, 15),
    "ele_eta"              : (40, -2.5, 2.5),
    "scl_eta"              : (40, -2.5, 2.5),
    "ele_isEB"             : (2,0,2),
    "ele_isEE"             : (2,0,2),
    "scl_E"                : (30, 0, 30),
    "trk_chi2red"          : (42, -1, 5),
    "trk_dr"               : (35, -0.05, 1.0),
    "trk_nhits"            : (20, 0, 20),
    "trk_p"                : (40, 0, 20),
    "ele_deltaetain"       : (30, -0.01, 0.1),
    "ele_deltaetaseed"     : (40, 0, 0.4),
    "ele_deltaphiin"       : (30, 0, 0.3),
    "ele_oldsigmaietaieta" : (100, 0, 0.08),
    "ele_oldsigmaiphiiphi" : (100, 0, 0.08),
    "ele_shFracHits"       : (20, 0, 4),
    "ele_ep"               : (40, 0, 4),
    "ele_fbrem"            : (50, -1, 1),
    "ele_gsfchi2"          : (60, 0, 30),
    "ele_oldhe"            : (40, 0, 10),
    "ele_oldr9"            : (20, 0, 2),
    "ele_sclNclus"         : (15, 0, 15),
    "ele_scletawidth"      : (24, 0, 0.6),
    "ele_sclphiwidth"      : (24, 0, 0.6),
    "gsf_chi2red"          : (20, 0, 10),
    "gsf_dr"               : (35, -0.05, 0.05),
    "gsf_mode_p"           : (60, 0, 30),
    "gsf_nhits"            : (25, 0, 25),
    "rho"                  : (60, 0, 60),
    "sc_clus1_E"           : (30, 0, 15),
    "sc_clus2_E"           : (20, 0, 10),
    "sc_clus1_EoverP"      : (25, 0, 1.5),
    "sc_clus2_EoverP"      : (25, 0, 1.5),
    "sc_clus1_deta"        : (30, -1.0, 1.0),
    "sc_clus2_deta"        : (30, -1.0, 1.0),
    "sc_clus1_dphi"        : (30, -1.0, 1.0),
    "sc_clus2_dphi"        : (30, -1.0, 1.0),
    "sc_clus1_nxtal"       : (8, 0, 8),
    "ele_eclu_EoverP"      : (25, -0.05, 0.05),
    "ele_oldcircularity"   : (20, 0, 1),
}

def cern_transfer(cernDir, output, skipFeatures=True):
    #check if cern webpage already available
    cmd = "xrdfs root://eosuser.cern.ch/ stat "+cernDir+"/"+output+"/index.php"
    result = sp.run(cmd, executable="/bin/bash", shell=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    if result.returncode!=0:
        #create webpage
        cmd = "xrdfs root://eosuser.cern.ch/ mkdir "+cernDir+"/"+output
        result = sp.run(cmd, executable="/bin/bash", shell=True)
        cmd = "xrdcp examplePage/index.php root://eosuser.cern.ch/"+cernDir+"/"+output
        sp.check_call(cmd, executable="/bin/bash", shell=True)
    
    import glob
    import concurrent.futures

    def copy_file(filepath):
        filename = os.path.basename(filepath)
        cmd = f"xrdcp -f {filepath} root://eosuser.cern.ch/{cernDir}/{output}"
        return sp.check_call(cmd, executable="/bin/bash", shell=True)
    
    files_to_copy = []
    if skipFeatures: fileTypes = ["rocComp*"]
    else:            fileTypes = ["*.png", "*.pdf"]
    for fileType in ["*.png", "*.pdf"]:
        files_to_copy.extend(glob.glob(os.path.join(output, fileType)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(copy_file, f) for f in files_to_copy]
        for future in concurrent.futures.as_completed(futures):
            try: future.result()
            except Exception as e: print(f"Error copying file: {e}")
        
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

def plotHists(sigHists, bkgHists, bin_edges_dict, output, reg):
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    
    for feat, _ in sigHists.items():
        #if not feat=="scl_eta": continue
        bin_centers = (bin_edges_dict[feat][:-1] + bin_edges_dict[feat][1:]) / 2
        
        plt.figure()
        hep.cms.label('', data=False, llabel='Preliminary', rlabel='', fontsize=20)

        #plt.step(bin_centers, sigHists[feat], where='mid', label="Signal",     color='blue')
        #plt.step(bin_centers, bkgHists[feat], where='mid', label="Background", color='red')
        bin_edges = bin_edges_dict[feat]
        plt.hist(bin_edges[:-1], bins=bin_edges, weights=sigHists[feat], density=True, 
                 label="Signal",     color='blue', histtype='step')
        plt.hist(bin_edges[:-1], bins=bin_edges, weights=bkgHists[feat], density=True,
                 label="Background", color='red',  histtype='step')

        plt.xlabel(feat)
        plt.ylabel('Density')
        plt.title('')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f'{output}/{feat}_{reg}_hist.png')
        plt.savefig(f'{output}/{feat}_{reg}_hist.pdf')

        plt.close()

def plotFScores(fscores, output):
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    
    #unpack names and scores
    names  = [x[0] for x in fscores]
    scores = [x[1] for x in fscores]

    plt.figure(figsize=(8,6))
    hep.cms.label('', data=False, llabel='CMS Preliminary', rlabel='', fontsize=45)
    plt.barh(names, scores, color='skyblue')
    plt.xlabel('F-Score')
    plt.title('Feature importance (F-Score)')
    plt.gca().invert_yaxis() #highest scores on top
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'{output}/features_f_score.png')
    plt.savefig(f'{output}/features_f_score.pdf')

    plt.show()

def main():
    import argparse
    p=argparse.ArgumentParser(description="make ROC curves from input ntuple")
    p.add_argument('ntupleFile', help="<REQUIRED> ntuple location")
    p.add_argument('--output', required=True, help="<REQUIRED> output location for plot")
    p.add_argument('--debug', default=False, action="store_true", help="Turn on debug prints")
    p.add_argument('--newModels', nargs='*', default=None, help="New Model")
    p.add_argument('--skipFeatures', action="store_true", default=False, help="skip generating histograms for features")
    p.add_argument('--reweight',  default=False, action="store_true", help="Calculate and apply pt-eta weights")

    args=p.parse_args()

    ntupleFile = uproot.open(args.ntupleFile)      

    if args.debug: print(ntupleFile.keys())

    tree = ntupleFile["ntuplizer/tree"]
    
    if args.reweight:
        pt_bins  = np.linspace(1, 10,  21)
        eta_bins = np.linspace(0, 2.5, 26)

        hist_sig_rew = np.zeros((len(pt_bins) - 1, len(eta_bins) - 1))
        hist_bkg_rew = np.zeros((len(pt_bins) - 1, len(eta_bins) - 1))
        
        for chunk in tree.iterate(['ele_pt', 'ele_eta', 'scl_eta', 'matchedToGenEle', 'ele_seedBiasedBDT'], library='pd', step_size=100_000):
            df = chunk.query("matchedToGenEle != 2")
            df.loc[df["matchedToGenEle"] > 0, "matchedToGenEle"] = 1
            df.query("ele_pt >= 1 & ele_pt < 10 & abs(scl_eta) < 2.5")
            
            pt  = df['ele_pt']
            eta = np.abs(df['scl_eta'])
            sig_mask = df['matchedToGenEle'] == 1
            bkg_mask = df['matchedToGenEle'] == 0

            hist_sig_rew += np.histogram2d(pt[sig_mask], eta[sig_mask], bins=[pt_bins, eta_bins])[0]
            hist_bkg_rew += np.histogram2d(pt[bkg_mask], eta[bkg_mask], bins=[pt_bins, eta_bins])[0]

        #compute weights
        weights_2d = hist_bkg_rew / hist_sig_rew 

    for reg in ['EB1', 'EB2', 'EE', 'all', 'weighted']:
        if not args.reweight and reg=='weighted': continue
        bdtScores_run2 = pd.DataFrame()
        seedBDT        = pd.DataFrame()
        #biasSeedBDT    = pd.DataFrame()

        stats = {}
        hists_sig, hists_bkg = {}, {}
        bin_edges_dict = {}
        for feat in features+['ele_pt', 'ele_eta', 'ele_seedBiasedBDT']:
            stats[feat] = {
                'sum_sig'   : 0.0,
                'sum_bkg'   : 0.0,
                'sum2_sig'  : 0.0,
                'sum2_bkg'  : 0.0,
                'count_sig' : 0,
                'count_bkg' : 0
            }
            if feat in binning: n_bins, x_min, x_max = binning[feat]
            else: n_bins, x_min, x_max = (50, 0, 500)
            hists_sig[feat] = np.zeros(n_bins)
            hists_bkg[feat] = np.zeros(n_bins)

            bin_edges = np.linspace(x_min, x_max, n_bins+1)
            bin_edges_dict[feat] = bin_edges

        for chunk in tree.iterate(features+['matchedToGenEle', 'ele_pt', 'ele_eta', 'ele_ID', 'ele_seedBiasedBDT', 'ele_isEB', 'ele_isEE'], library='pd', step_size=100_000): #FIXME add back "ele_pt"
            df = chunk.query("matchedToGenEle != 2")
            df.loc[df["matchedToGenEle"] > 0, "matchedToGenEle"] = 1
            df = df.query("ele_pt >= 1 & ele_pt < 10 & abs(scl_eta) < 2.5")
            #if   reg=="EB": df = df.query("ele_isEB == 1")
            #elif reg=="EE": df = df.query("ele_isEE == 1")
            if   reg=="EB1" : df = df.query("abs(scl_eta) <  0.8")
            elif reg=="EB2" : df = df.query("abs(scl_eta) <  1.479 & abs(scl_eta) >= 0.8")
            elif reg=="EE"  : df = df.query("abs(scl_eta) >= 1.479")
            elif reg=="all" or reg=="weighted": pass
            else: print("not a valid region")
            
            feats = df[features+['ele_pt', 'ele_eta', 'ele_seedBiasedBDT']].dropna()
            isE   = df["matchedToGenEle"].dropna()
                
            
            for feat in features+['ele_pt', 'ele_eta', 'ele_seedBiasedBDT']:
                if args.skipFeatures: break
                x_sig = feats[feat][isE == 1]
                x_bkg = feats[feat][isE == 0]
                
                if reg=='weighted':
                    pt_sig  = feats['ele_pt'][isE == 1]
                    eta_sig = np.abs(feats['scl_eta'][isE == 1])
                    pt_idx  = np.digitize(pt_sig,  bins=pt_bins)  - 1
                    eta_idx = np.digitize(eta_sig, bins=eta_bins) - 1
                    
                    valid = (pt_idx >= 0)  & (pt_idx < weights_2d.shape[0]) & \
                            (eta_idx >= 0) & (eta_idx < weights_2d.shape[1])

                    weights = np.zeros_like(x_sig)
                    weights[valid] = weights_2d[pt_idx[valid], eta_idx[valid]]

                stats[feat]['sum_sig']   += x_sig.sum()
                stats[feat]['sum_bkg']   += x_bkg.sum()
                stats[feat]['sum2_sig']  += (x_sig**2).sum()
                stats[feat]['sum2_bkg']  += (x_bkg**2).sum()
                stats[feat]['count_sig'] += x_sig.shape[0]
                stats[feat]['count_bkg'] += x_bkg.shape[0]

                if reg=='weighted': 
                    hist_s, _ = np.histogram(x_sig, bins=bin_edges_dict[feat], weights=weights)
                else: hist_s, _ = np.histogram(x_sig, bins=bin_edges_dict[feat])
                hist_b, _ = np.histogram(x_bkg, bins=bin_edges_dict[feat])
                
                hists_sig[feat] += hist_s
                hists_bkg[feat] += hist_b

            bdtScores_run2 = pd.concat([bdtScores_run2, df[["matchedToGenEle", "ele_ID"]]])
            seedBDT        = pd.concat([seedBDT,        df[["matchedToGenEle", "ele_seedBDT", "ele_seedBiasedBDT"]]])
            #biasSeedBDT    = pd.concat([biasSeedBDT,    df[["matchedToGenEle", "ele_seedBiasedBDT"]]])

            if not args.newModels is None:
                df=feats.astype(np.float32)
                df.drop(['ele_pt', 'ele_eta', 'ele_seedBiasedBDT'], axis=1, inplace=True)
                df=pd.concat([isE, df], axis=1)
                df.to_csv(args.output+'/data_'+reg+'.csv', mode="a", index=False, header=False)
    
        if not args.newModels is None:
            #loop through models
            print(f"reg: {reg}")
            dmat = xgb.DMatrix(args.output+'/data_'+reg+'.csv?format=csv&label_column=0#dtrain.cache')
            modelPreds = {}
            #switch to different models
            for modelName in args.newModels:
                if (reg=='EB1' or reg=='EB2') and '_EE' in modelName: continue
                if reg=='EE' and 'EB' in modelName: continue
                booster = xgb.Booster()
                booster.load_model(modelName)
                preds = booster.predict(dmat)

                modelPreds[modelName.replace('.bin','').replace('electron_id_','')] = pd.DataFrame(
                        {"matchedToGenEle": dmat.get_label(),
                         "bdtScore"       : preds})

            os.remove(args.output+'/data_'+reg+'.csv')
    
        fprs, tprs, aucs, labels = [], [], [], []

        fpr, tpr, roc_auc = ROC.getROCs(bdtScores_run2["matchedToGenEle"], bdtScores_run2["ele_ID"])
        fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc); labels.append("2020Nov28")
        ROC.plotROCs([fpr], [tpr], [roc_auc], args.output, labels=["2020Nov28"], figName=f"roc_{reg}")

        fpr, tpr, roc_auc = ROC.getROCs(seedBDT["matchedToGenEle"], seedBDT["ele_seedBDT"])
        fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc); labels.append("electronID(\"unbiased\")")

        fpr, tpr, roc_auc = ROC.getROCs(seedBDT["matchedToGenEle"], seedBDT["ele_seedBiasedBDT"])
        fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc); labels.append("electronID(\"ptbiased\")")


        ROC.plotROCs([fpr], [tpr], [roc_auc], args.output, labels=["electronID(\"unbiased\")"], figName=f"SEED_ROC_{reg}")
    
        #load new models
        if not args.newModels is None:
            for name,pred in modelPreds.items():
                print(name)
                fpr, tpr, roc_auc = ROC.getROCs(pred["matchedToGenEle"], pred["bdtScore"])
                fprs.append(fpr); tprs.append(tpr); aucs.append(roc_auc); labels.append(name.split('/')[-1])
                ROC.plotROCs([fpr], [tpr], [roc_auc], args.output, labels=[name], figName=name.split('/')[-1]+'_'+reg)
            del modelPreds;

        ROC.plotROCs(fprs, tprs, aucs, args.output, labels=labels, figName=f"rocComp_{reg}")
        del bdtScores_run2; del seedBDT;
    
        fscores={}
        for feat,s in stats.items():
            if args.skipFeatures: break
            if s['count_sig'] == 0 or s['count_bkg'] == 0:
                fscores[feat] = 0
                continue

            mu1 = s['sum_sig'] / s['count_sig']
            mu0 = s['sum_bkg'] / s['count_bkg']
            var1 = s['sum2_sig'] / s['count_sig'] - mu1**2
            var0 = s['sum2_bkg'] / s['count_bkg'] - mu0**2
            
            fscore = (mu1 - mu0)**2 / (var1 + var0 + 1e-8) #small epsilon
            fscores[feat] = fscore

            #sorted_fscores = sorted(fscores.items(), key=lambda x: x[1], reverse=True)

        print("Feature F-scores:")
        if not args.skipFeatures:
            pass
            #for feature, score in sorted_fscores: print(f"{feature}, {reg}: {score:.4f}")
    
        #plotFScores(sorted_fscores, args.output)
        if not args.skipFeatures: plotHists(hists_sig, hists_bkg, bin_edges_dict, args.output, reg)
    
    cern_transfer('/eos/user/w/wterrill/www/projects/retraining/', args.output, args.skipFeatures)

if __name__=="__main__": main()
 
