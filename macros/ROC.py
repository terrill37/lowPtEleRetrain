import uproot

import numpy as np
import matplotlib.pyplot as plt
import mplhep

from sklearn.metrics import roc_curve, auc

def plotROC(fpr, tpr, auc, output=''):
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0,1],[0,1], color='gray', linestyle='--') #random guess line
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(output+'/roc.png')

def main():
    import argparse
    p=argparse.ArgumentParser(description="make ROC curves from input ntuple")
    p.add_argument('ntupleFile', help="<REQUIRED> ntuple location")
    p.add_argument('--output', required=True, help="<REQUIRED> output location for plot")
    p.add_argument('--debug', default=False, action="store_true", help="Turn on debug prints")

    args=p.parse_args()

    ntupleFile = uproot.open(args.ntupleFile)

    if args.debug: print(ntupleFile.keys())

    tree = ntupleFile["ntuplizer/tree"]

    isEle = tree["matchedToGenEle"].array(library='np')
    isEle = (isEle>0).astype(int)

    bdtScores_run2 = tree["ele_ID"].array(library='np')

    isEle = np.array(isEle)
    bdtScores_run2 = np.array(bdtScores_run2)

    fpr, tpr, _ = roc_curve(isEle, bdtScores_run2)
    roc_auc     = auc(fpr, tpr)

    plotROC(fpr, tpr, roc_auc, args.output)

if __name__=="__main__": main()
    
