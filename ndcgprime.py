'''Implementation of NDCG' '''

import numpy as np
import pandas as pd

def dcg(row,r=1,isDenominator=True,k=10):
        if isDenominator:
            row=[x for x in row if x!=0]
        
        if not isDenominator and len(row)>k:
            row = row[:k]
            
        row=np.array(row)
        if len(row)!=0:
            row=sorted(row, reverse=True)
            
        row = np.append(row,r)
        return np.sum(row / np.log2(np.arange(2, row.size + 2)))
    
def dcgp(row,k):
    if len(row)==0:
        return 0
    else:
        r=row[1]
        row=row[0]
        
        if len(row)>k:
              row = row[:k]
                
        row=np.array(row)
        row = np.append(row,r)
        return np.sum(row / np.log2(np.arange(2, row.size + 2)))

    
def rank_by(row,col,by):
    r2=row[by]   
    indices = list(range(len(r2)))
    indices.sort(key=lambda x: r2[x],reverse=True)
    r2 = [r2[i] for i in indices]
    
    r1 = row[col]
    r1 = [r1[i] for i in indices]
    r1 = [r1[index] for  index,value in enumerate(r2) if value>0]
         
    row[col+"_by_"+by] = r1
    return row

def r_value(row,k):
    r1=row[0]
    
    if len(r1)>k:
        r1 =r1[:k]
        
    r2=row[1]
    if sum(r1)==0 or sum(r2)==0:
        return 0
    else:
        return sum(r2)/sum(r1)

def NDCGPrime_PQA_beta(data,by,k=10):
    d = data.reset_index(drop=True)
    col = 'ave_annot_score'
    d = d.apply(rank_by,args=(col,by,),axis=1)
    d['rValue_'+col] = d[[col,col+"_by_"+by]].apply(r_value,args=(k,),axis=1)
    d['Ideal_DCG_'+col] = d[col].apply(dcg)
    d['DCG_'+col] = d[[col+"_by_"+by,'rValue_'+col]].apply(dcgp,args=(k,),axis=1)     
    d['NDCGPrime_'+col] = d.apply(lambda x: x['DCG_'+col]/x['Ideal_DCG_'+col]
                                                       if sum(x[col])>0 else x['DCG_'+col],axis=1)    
    d['NDCGPrime_'+col] = d.apply(lambda x: x['NDCGPrime_'+col]
                                                       if sum(x[col])>0 else dcg(x[col+"_by_"+by],
                                                                                 isDenominator=False,k=k),axis=1) 
    return d['NDCGPrime_'+col].mean()