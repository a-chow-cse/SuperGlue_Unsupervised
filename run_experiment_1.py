import os
import torch
import numpy as np
import matplotlib.pyplot as plt



#program to run .match.py with our defined pair files, outputs npz file with visualtion image
def run_match():
    path=os.path.abspath(os.getcwd())+"/ButterFlyPairs_SingleEach/"
    print("pair file path: ",path)
    for file in os.scandir(path):
        instruction="./match_pairs.py --resize 1600 --superglue outdoor --max_keypoints 2048 --nms_radius 3  --resize_float \
            --input_dir ./ --input_pairs ButterFlyPairs_SingleEach/"+file.name+" --output_dir " +file.name.removesuffix(".txt")+" "
        print(instruction)
        #os.system(instruction) 

def create_heatmap():
    root_path=os.path.abspath(os.getcwd())
    result_path=root_path+"/heatmap_results/"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    path=os.path.abspath(os.getcwd())+"/ButterFlyPairs_SingleEach/"

    for file in os.scandir(path):

        folder_path=root_path+"/"+file.name.removesuffix(".txt")+"/"

        if os.path.exists(folder_path)==True:
            if os.path.exists(result_path+file.name.removesuffix(".txt")+".npz")==True:
                print(file.name.removesuffix(".txt")+".npz"," Already exists")
            else:
                print(result_path+file.name.removesuffix(".txt"))
                print("Creating heatmap in ",folder_path)

                heatmap=np.zeros((1600,1600))
                for npzfile in os.scandir(folder_path):
                    #print(npzfile.name)
                    npz=np.load(folder_path+npzfile.name)
                    for i in range(npz['keypoints0'].shape[0]):
                        if npz['matches'][i]>-1:
                            heatmap[int(npz['keypoints0'][i][1])][int(npz['keypoints0'][i][0])]+=1
                np.savez(result_path+file.name.removesuffix(".txt"), heat_map=heatmap)

def show_heatmap():
    root_path=os.path.abspath(os.getcwd())
    result_path=root_path+"/heatmap_results/"
    butterfile_train_path=root_path+"/ButterFly/train/"

    butterfly_train_folders=[]
    for file in os.scandir(butterfile_train_path):
        if file.is_dir():
            butterfly_train_folders.append(file.name)
        


    for folder in butterfly_train_folders:
        print(folder)
        if os.path.exists(result_path+"/"+folder+"_Self.npz") and os.path.exists(result_path+"/"+folder+"_Cross.npz"):
            self_heat_map=np.load(result_path+"/"+folder+"_Self.npz")
            cross_heat_map=np.load(result_path+"/"+folder+"_Cross.npz")
            sheatmap=self_heat_map["heat_map"]
            sheatmap=np.divide(sheatmap,np.sum(sheatmap))
            
            #plt.imshow(sheatmap, interpolation='nearest',cmap="gray")
            #plt.show()

            cheatmap=cross_heat_map["heat_map"]
            print(np.max(cheatmap))
            cheatmap=np.divide(cheatmap,np.sum(cheatmap))

            difference_heatmap=np.subtract(sheatmap,cheatmap)
            difference_heatmap[difference_heatmap<0]=0

            #plt.imshow(difference_heatmap, interpolation='nearest',cmap="gray")
            #plt.show()

             #resize the image and plot the difference heatmap
            pairs_path=os.path.abspath(os.getcwd())+"/ButterFlyPairs_SingleEach/"
            with open(pairs_path+folder+'_Self.txt') as f:
                first_line = f.readline()
                Image=plt.imread(root_path+"/"+first_line.split()[0])

                plt.imshow(Image)
                f.close()
        


def save_knn():
    root_path=os.path.abspath(os.getcwd())
    result_path=root_path+"/knn_results/"

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    butterfile_train_path=root_path+"/ButterFly/train/"

    butterfly_train_folders=[]
    for file in os.scandir(butterfile_train_path):
        if file.is_dir():
            butterfly_train_folders.append(file.name)
        


    for folder in butterfly_train_folders:
        print("current species: " , folder)
        
        self_match_number=[]
        cross_match_number=[]
        
        self_folder_npz_path=root_path+"/"+folder+"_Self/"
        cross_folder_npz_path=root_path+"/"+folder+"_Cross/"
        
        if os.path.exists(self_folder_npz_path) and os.path.exists(cross_folder_npz_path):
            for file in os.scandir(self_folder_npz_path):
                npz=np.load(self_folder_npz_path+file.name)
                self_match_number.append(np.sum(npz['matches']>-1))
        
            #print(self_match_number)

            for file in os.scandir(cross_folder_npz_path):
                npz=np.load(cross_folder_npz_path+file.name)
                cross_match_number.append(np.sum(npz['matches']>-1))
        
            #print(cross_match_number)
            
            np.savez(result_path+folder, self_matches=self_match_number,cross_matches=cross_match_number)

if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)

    #run_match()
    #create_heatmap()
    #show_heatmap()
    save_knn()
    