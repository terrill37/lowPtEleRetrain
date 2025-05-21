import uproot

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from sklearn.metrics import roc_curve, auc

def getROCs(isObj, scores):
    fpr, tpr, thresh = roc_curve(isObj, scores)
    roc_auc = auc(fpr,tpr)
    return fpr, tpr, roc_auc

def plotROCs(fprs, tprs, aucs, output, labels=[], figName=None):
    if figName is None: figName='roc'
    plt.clf()
    plt.figure()
    hep.cms.label('', data=False, llabel='Preliminary', rlabel='', fontsize=20)
    plt.plot(tprs[0],tprs[0], color='gray', linestyle='--') #random guess line
    #plt.fill_between(tprs[0], tprs[0], y2=1e-3, alpha=0.3, label='AUC (random guess)')
    for idx,fpr in enumerate(fprs):
        plt.plot(tprs[idx], fprs[idx], lw=2, label=f"ROC {labels[idx]} (AUC = {aucs[idx]:.2f})")

    plt.xlim([0.,1])
    plt.ylim([0.001,1.05])
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    plt.legend(loc='upper left')
    plt.gca().set_yscale('log')
    plt.grid()
    plt.savefig(output+'/'+figName+'.png')
    plt.savefig(output+'/'+figName+'.pdf')
    
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

    fpr, tpr, roc_auc = getROCs(isEle, bdtScores_run2)

    plotROCs([fpr], [tpr], [roc_auc], args.output)

if __name__=="__main__": main()
    
