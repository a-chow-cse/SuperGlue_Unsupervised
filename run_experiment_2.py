import numpy as np

_2d_list_path = 'Experiment2/2D_list_of_all_npzs_with_class.npz'
npz = np.load(_2d_list_path)
[row,column]=npz['arr_0'].shape
results=np.zeros((row,column-1))

total=147456
start=1

for r in range(row):
    for c in range(column-1):
        start+=1
        #print(npz['arr_0'][r,c])
        path=npz['arr_0'][r,c]
        if path=="":
            continue
        match_npz=np.load(path)
        matchNumber=np.sum(match_npz['matches']>-1)
        results[r,c]=matchNumber
        print("Done ",start," out of ",total)

np.savez("Experiment2/MatchNumbersTable",results)