import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import h5py    
from sklearn.cluster import DBSCAN
from pyevtk.hl import pointsToVTK
from pyevtk.hl import gridToVTK

class Postprocess_data():
    
    def __init__(self,params):
        self.params = params

    def get_post_centroids(self,Voxel_centroid_phases_files ,cluster_id):


        with h5py.File(Voxel_centroid_phases_files , "r") as hdfr:
            group = cluster_id
            Phase_arr =  np.array(hdfr.get(f"{group}/{group}"))
            Phase_columns = list(list(hdfr.get(f"{group}").attrs.values())[0])
            Phase_cent_df =pd.DataFrame(data=Phase_arr, columns=Phase_columns)

            Df_centroids = Phase_cent_df.copy()
            Df_centroids_no_files = Df_centroids.drop(['file_name'] , axis=1)
            files = Df_centroids['file_name']
            files.values

            return Df_centroids_no_files, Df_centroids, Phase_columns

        
    def DBSCAN_clustering(self, cluster_id, eps, min_samples,plot= False, plot3d = False, save =False):
        OutFile_path = self.params['output_path'] 
        Voxel_centroid_phases_files = self.params['output_path'] +"/Output_voxel_cetroids_phases.h5"

        Df_centroids_no_files, Df_centroids, Phase_columns = self.get_post_centroids( Voxel_centroid_phases_files ,cluster_id)

        db = DBSCAN(eps=eps, min_samples= min_samples).fit(Df_centroids_no_files.values) #eps=5., min_samples= 35
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        if plot == True:
            plt.hist(labels,bins = 100);

        cluster_combine_lst = []

        for i in np.unique(labels):
            if i !=-1:

                cl_idx =np.argwhere(labels==i).flatten()
                cl_cent=Df_centroids_no_files.iloc[cl_idx]
                cl_cent["ID"] = [i]*len(cl_cent)
                cluster_combine_lst.append(cl_cent)


        if plot3d == True: 
            OutFile = OutFile_path +"/Output_DBSCAN_segmentation_phase" + f"{cluster_id}"
            Df_comb = pd.concat(cluster_combine_lst)
            image = Df_comb.values
            FILE_PATH1 = OutFile
            x = np.ascontiguousarray(image[:,0])
            y= np.ascontiguousarray(image[:,1])
            z = np.ascontiguousarray(image[:,2])
            label = np.ascontiguousarray( image[:,3])
            pointsToVTK(FILE_PATH1,x,y,z, data = {"label" : label}  )


        if save == True:
            OutFile = OutFile_path + f"/Output_DBSCAN_segmentation_phase_{cluster_id}.h5"
            with h5py.File(OutFile, "w") as hdfw:

                G = hdfw.create_group(f"{cluster_id}")
                G.attrs["columns"] = Phase_columns

                for i in tqdm(np.unique(labels)):
                    if i !=-1:

                        cl_idx =np.argwhere(labels==i).flatten()
                        cl_cent=Df_centroids.iloc[cl_idx]
                        #cluster_combine_lst.append(cl_cent)
                        G.create_dataset("{}".format(i), data = cl_cent.values)
                        #l_cent.to_csv("cl_D3_Zr_{}.csv".format(i), index =False)



    


 