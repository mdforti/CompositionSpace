from Segmentation.datautils.data_utils import Prepare_data
from Segmentation.models.models import get_model
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import json 
import h5py
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from tqdm import tqdm
import os

class Composition_Space():
    
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

        cluster_0 = np.argwhere(y_pred == 0)
        cluster_1 = np.argwhere(y_pred == 1)
        cluster_2 = np.argwhere(y_pred == 2)

        df_dis2 = Ratios.iloc[cluster_2.flatten()]
        df_dis1 = Ratios.iloc[cluster_1.flatten()]
        df_dis0 = Ratios.iloc[cluster_0.flatten()]
        return df_dis2, df_dis1,df_dis0, cluster_0,cluster_1,cluster_2,Ratios
    
    def SaveCompositionClusters(self,ml_params):
        OutFile = self.params['output_path'] + "/Output_voxel_cetroids_phases.h5"
        n_components = self.params["n_phases"]
        VoxRatioFile = self.params['output_path'] + "/Output_voxel_composition.h5"
        VoxFile = self.params['output_path'] + "/Output_voxels.h5"

        df_dis2, df_dis1,df_dis0, cluster_0,cluster_1,cluster_2,Ratios = self.CompositionCluster(VoxRatioFile, VoxFile, n_components,ml_params)

        plot_files_cl_0 = []
        plot_files_cl_1 = []
        plot_files_cl_2 = []

        #for i in cluster_2.flatten():
        #    plot_files_cl_2.append(Ratios['filename'][i])

        for i in cluster_0.flatten():
            plot_files_cl_0.append(Ratios['vox'][i])
        for i in cluster_1.flatten():
            plot_files_cl_1.append(Ratios['vox'][i])

        for i in cluster_2.flatten():
            plot_files_cl_2.append(Ratios['vox'][i])


        #get the item name for h5 file
        plot_files_cl_0_group = [int(file_num) for file_num in plot_files_cl_0 ]
        plot_files_cl_1_group = [int(file_num) for file_num in plot_files_cl_1 ]
        plot_files_cl_2_group = [int(file_num) for file_num in plot_files_cl_2 ]

        hdfw = h5py.File(OutFile,"w")

        G1 = hdfw.create_group("All")
        G1.attrs["what"] = ["Centroid of All voxels"]
        G1.attrs["howto_Group_name"] = ["Group_sm_vox_xyz_Da_spec/"]
        G1.attrs["colomns"] = ["x","y","z","file_name"]

        G2 = hdfw.create_group("Phase1")
        G2.attrs["what"] = ["Centroid of Phase1 voxels"]
        G2.attrs["howto_Group_name"] = ["Group_sm_vox_xyz_Da_spec/"]
        G2.attrs["colomns"] = ["x","y","z","file_name"]

        G3 = hdfw.create_group("Phase2")
        G3.attrs["what"] = ["Centroid of Phase2 voxels"]
        G3.attrs["howto_Group_name"] = ["Group_sm_vox_xyz_Da_spec/"]
        G3.attrs["colomns"] = ["x","y","z","file_name"]


        ##All
       
        #Small_chunk_file_name = "./file_file_D3_High_Hc_R5076_53143_apt_large_chunks_test_arr_h5_small_chunks_test_arr4.h5"
        hdf_sm_r = h5py.File(VoxFile,"r")
        group = hdf_sm_r.get("0")
        Total_Voxels =list(list(group.attrs.values())[0])

        Total_Voxels_int =""
        for number in Total_Voxels:
            Total_Voxels_int = Total_Voxels_int+ number

        Total_Voxels_int = int(Total_Voxels_int)
        hdf_sm_r.close()
        plot_files_cl_All_group = [file_num for file_num in range(Total_Voxels_int)]
        #files = [file_num for file_num in range(905667)]

        FilesList = [plot_files_cl_All_group, plot_files_cl_1_group, plot_files_cl_2_group]
        GroupsList = [G1,G2,G3]
        PhasesNames = ["All","Ph1", "Ph2"] 

        for i in range(len(FilesList)):

            CentroidsDic = self.get_voxel_centroid(VoxFile= VoxFile, FilesArr= FilesList[i])
            GroupsList[i].create_dataset("{}".format(PhasesNames[i]), data = pd.DataFrame.from_dict(CentroidsDic).values)
        hdfw.close()


