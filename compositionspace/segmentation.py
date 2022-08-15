from compositionspace.datautils import Prepare_data
from compositionspace.models import get_model
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import json 
import h5py
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from tqdm import tqdm
import os
from pyevtk.hl import pointsToVTK
from pyevtk.hl import gridToVTK#, pointsToVTKAsTIN

class Composition_clustering():
    
    def __init__(self,params):
        self.params = params

    def get_PCA_cumsum(self):

        VoxRatioFile = self.params['output_path'] + "/Output_voxel_composition.h5"
        VoxFile = self.params['output_path'] + "/Output_voxels.h5"
        with h5py.File(VoxFile,"r") as hdf:
            group = hdf.get("Group_sm_vox_xyz_Da_spec")
            group0 = hdf.get("0")
            spec_lst = list(list(group0 .attrs.values())[2])
        #print(spec_lst)

        with h5py.File(VoxRatioFile , "r") as hdfr:
            Ratios = np.array(hdfr.get("vox_ratios"))
            Ratios_colomns = list(list(hdfr.attrs.values())[0])
            Group_name = list(list(hdfr.attrs.values())[1])

        Ratios = pd.DataFrame(data=Ratios, columns=Ratios_colomns)   

        X_train=Ratios.drop(['Total_no','vox'], axis=1)
        PCAObj = PCA(n_components = len(spec_lst)) 
        PCATrans = PCAObj.fit_transform(X_train)
        PCAObj.explained_variance_
        PCACumsumArr = np.cumsum(PCAObj.explained_variance_ratio_)
        
        plt.figure(figsize=(5,5))
        plt.plot( range(1,len(PCACumsumArr)+1,1),PCACumsumArr,"-o")
        plt.ylabel("Explained Variance")
        plt.xlabel('Dimensions')
        plt.grid()
        output_path = self.params['output_path'] + "/PCA_cumsum.png"
        plt.savefig(output_path)
        plt.show()
        
        return PCACumsumArr

    
    
    def get_bics_minimization(self):
        
        VoxRatioFile = self.params['output_path'] + "/Output_voxel_composition.h5"
        VoxFile = self.params['output_path'] + "/Output_voxels.h5"
        with h5py.File(VoxFile,"r") as hdf:
            group = hdf.get("Group_sm_vox_xyz_Da_spec")
            group0 = hdf.get("0")
            spec_lst = list(list(group0 .attrs.values())[2])
        #print(spec_lst)

        with h5py.File(VoxRatioFile , "r") as hdfr:
            Ratios = np.array(hdfr.get("vox_ratios"))
            Ratios_colomns = list(list(hdfr.attrs.values())[0])
            Group_name = list(list(hdfr.attrs.values())[1])

        Ratios = pd.DataFrame(data=Ratios, columns=Ratios_colomns) 
        gm_scores=[]
        aics=[]
        bics=[]
        X = Ratios.drop(['Total_no','vox'], axis=1)
        n_clusters=list(range(1,self.params["bics_clusters"]))
        for n_cluster in tqdm(n_clusters):
            gm = GaussianMixture(n_components=n_cluster,verbose=0)
            gm.fit(X)
            y_pred=gm.predict(X)
            #gm_scores.append(homogeneity_score(y,y_pred))
            aics.append(gm.aic(X))
            bics.append(gm.bic(X))
            
        output_path = self.params['output_path'] + "/bics_aics.png"
        plt.plot(n_clusters, aics, "-o",label="AIC")
        plt.plot(n_clusters, bics, "-o",label="BIC")
        plt.legend()
        plt.savefig(output_path)
        plt.show()
        return self.params["bics_clusters"], aics, bics    
    
   
    def calculate_centroid(self, data):
        """
        Calculate centroid
        Parameters
        ----------
        data: pandas DataFrame or numpy array
        Returns
        -------
        centroid
        """
        length = len(data[:,0])
        sum_x = np.sum(data[:,0])
        sum_y = np.sum(data[:,1])
        sum_z = np.sum(data[:,2])
        return sum_x/length, sum_y/length, sum_z/length



    def get_voxel_centroid(self, VoxFile, FilesArr):

        with h5py.File(VoxFile,"r") as hdf:
            #group = hdf.get("Group_sm_vox_xyz_Da_spec")
            #spec_lst = list(list(group.attrs.values())[1])
            items = list(hdf.items())
            item_lst = []
            for item in range(len(items)):
                item_lst.append([100000*(item),100000*(item+1)])
            item_lst = np.array(item_lst)


            Dic_centroids = {}
            Dic_centroids["x"]=[]
            Dic_centroids["y"]=[]
            Dic_centroids["z"] = []
            Dic_centroids["file_name"] = []
            Df_centroids = pd.DataFrame(columns=['x', 'y', 'z','filename'],  dtype= float)
            for filename in tqdm(FilesArr):
                #file_name_grp = "Group_sm_vox_xyz_Da_spec/"+"{}".format(int(i))
                group = np.min(item_lst[[filename in range(j[0],j[1]) for j in item_lst]])
                xyz_Da_spec_atoms = np.array(hdf.get("{}/{}".format(group,filename)))
                x,y,z = self.calculate_centroid(xyz_Da_spec_atoms)
                Dic_centroids["x"].append(x)
                Dic_centroids["y"].append(y)
                Dic_centroids["z"].append(z)
                Dic_centroids["file_name"].append(filename)
                #Df_centroids = Df_centroids.append({'x': x, 'y': y, 'z': z, 'filename' : i}, ignore_index=True)

        return Dic_centroids

    
    def CompositionCluster(self, VoxRatioFile, VoxFile,n_components,ml_params):
        
        with h5py.File(VoxFile,"r") as hdf:
            group = hdf.get("Group_sm_vox_xyz_Da_spec")
            group0 = hdf.get("0")
            spec_lst = list(list(group0 .attrs.values())[2])
        #print(spec_lst)

        with h5py.File(VoxRatioFile , "r") as hdfr:
            Ratios = np.array(hdfr.get("vox_ratios"))
            Ratios_colomns = list(list(hdfr.attrs.values())[0])
            Group_name = list(list(hdfr.attrs.values())[1])

        Ratios = pd.DataFrame(data=Ratios, columns=Ratios_colomns) 
        
        X = Ratios.drop(['Total_no','vox'], axis=1)
        gm = get_model(ml_params=ml_params)
        #gm = GaussianMixture(n_components=n_components, max_iter=100000,verbose=0)
        gm.fit(X)
        y_pred=gm.predict(X)
        
        cluster_lst = []
        for phase in range(n_components):
            cluster_lst.append(np.argwhere(y_pred == phase).flatten())        
        df_lst = []
        for cluster in cluster_lst:
            df_lst.append(Ratios.iloc[cluster])
            
        #sorting
        cluster_lst_sort = []
        len_arr = np.array([len(x) for x in cluster_lst])
        sorted_len_arr = np.sort(len_arr)
        for length in sorted_len_arr:
            print()

            cluster_lst_sort.append(cluster_lst[np.argwhere(len_arr == length)[0,0]])

        print([len(x) for x in cluster_lst_sort])
        cluster_lst = cluster_lst_sort
        
        return cluster_lst,Ratios
    
    def get_composition_clusters(self,ml_params):
        OutFile = self.params['output_path'] + "/Output_voxel_cetroids_phases.h5"
        n_components = self.params["n_phases"]
        VoxRatioFile = self.params['output_path'] + "/Output_voxel_composition.h5"
        VoxFile = self.params['output_path'] + "/Output_voxels.h5"

        cluster_lst,Ratios = self.CompositionCluster(VoxRatioFile, VoxFile, n_components,ml_params)

        plot_files = []
        for phase in range(len(cluster_lst)):
            cluster_files = []
            cluster = cluster_lst[phase]
            for voxel_id in cluster:
                cluster_files.append(Ratios['vox'][voxel_id])
            plot_files.append(cluster_files)

        plot_files_group = []
        for cluster_files in plot_files:
            plot_files_group.append([int(file_num) for file_num in cluster_files ])
            
        with h5py.File(VoxFile,"r") as hdf_sm_r:
            hdf_sm_r = h5py.File(VoxFile,"r")
            group = hdf_sm_r.get("0")
            Total_Voxels =list(list(group.attrs.values())[0])

            Total_Voxels_int =""
            for number in Total_Voxels:
                Total_Voxels_int = Total_Voxels_int+ number

            Total_Voxels_int = int(Total_Voxels_int)
            hdf_sm_r.close()
            plot_files_cl_All_group = [file_num for file_num in range(Total_Voxels_int)]
            
        plot_files_group.append(plot_files_cl_All_group)

        with h5py.File(OutFile,"w") as hdfw:
            for cluster_file_id in range(len(plot_files_group)):

                G = hdfw.create_group(f"{cluster_file_id}")
                G.attrs["what"] = ["Centroid of voxels"]
                G.attrs["howto_Group_name"] = ["Group_sm_vox_xyz_Da_spec/"]
                G.attrs["colomns"] = ["x","y","z","file_name"]

                CentroidsDic = self.get_voxel_centroid(VoxFile= VoxFile, FilesArr= plot_files_group[cluster_file_id])
                G.create_dataset(f"{cluster_file_id}", data = pd.DataFrame.from_dict(CentroidsDic).values)

    def plot3d(self):
        
        OutFile = self.params['output_path'] + "/Output_voxel_cetroids_phases"

        Voxel_centroid_phases_files = self.params['output_path'] + "/Output_voxel_cetroids_phases.h5"

        with h5py.File(Voxel_centroid_phases_files , "r") as hdfr:
            
            groups =list(hdfr.keys())
            for group in range(len(groups)-2):
                Phase_arr =  np.array(hdfr.get(f"{group}/{group}"))
                Phase_columns = list(list(hdfr.get(f"{group}").attrs.values())[0])
                Phase_cent_df =pd.DataFrame(data=Phase_arr, columns=Phase_columns)
                
                image = Phase_cent_df.values
                FILE_PATH1 = OutFile + f"_{group}"
                print(FILE_PATH1)
                x = np.ascontiguousarray(image[:,0])
                y= np.ascontiguousarray(image[:,1])
                z = np.ascontiguousarray(image[:,2])
                label = np.ascontiguousarray( image[:,3])
                pointsToVTK(FILE_PATH1,x,y,z, data = {"label" : label}  )



