import pandas as pd
import re
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import pickle
#import paraprobe_transcoder
import h5py

def label_ions(pos,rrngs):
    pos['comp'] = ''
    pos['colour'] = '#FFFFFF'
    pos['nature'] = ''
    count=0;
    for n,r in rrngs.iterrows():
        count= count+1;
        pos.loc[(pos.Da >= r.lower) & (pos.Da <= r.upper),['comp','colour', 'nature']] = [r['comp'],'#' + r['colour'],count]
    
    return pos

#
def atom_filter(x, Atom_range):
    """
    Get a list of atom species and their counts
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    Assuming all the data
    """
    Atom_total = pd.DataFrame()
    for i in range(len(Atom_range)):
        Atom = x[x['Da'].between(Atom_range['lower'][i], Atom_range['upper'][i], inclusive=True)]
        Atom_total = Atom_total.append(Atom)
        Count_Atom= len(Atom_total['Da'])   
    return Atom_total, Count_Atom  


def read_pos(file_name):
    """
    Read the pos file 
    
    Parameters
    ----------
    file_name: string
        Name of the input file
    
    Returns
    -------
    pos: np structured array
        The atom positions and ---- ratio
    
    Notes
    -----
    Assumptions
    
    Examples
    --------
    
    Raises
    ------
    FileNotFoundError: describe
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError("filename does not exist")
    
    with open(file_name, 'rb') as f:
        dt_type = np.dtype({'names':['x', 'y', 'z', 'm'], 
                      'formats':['>f4', '>f4', '>f4', '>f4']})
        pos = np.fromfile(f, dt_type, -1)
    
    return pos

def read_rrng(file_name):
    """
    Read the data 
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    """
    patterns = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')
    ions = []
    rrngs = []
    
    with open(file_name, "r") as rf:
        for line in rf:
            m = patterns.search(line)
            if m:
                if m.groups()[0] is not None:
                    ions.append(m.groups()[:2])
                else:
                    rrngs.append(m.groups()[2:])
    
    ions = pd.DataFrame(ions, columns=['number','name'])
    ions.set_index('number',inplace=True)
    rrngs = pd.DataFrame(rrngs, columns=['number','lower','upper','vol','comp','colour'])
    rrngs.set_index('number',inplace=True) 
    rrngs[['lower','upper','vol']] = rrngs[['lower','upper','vol']].astype(float)
    rrngs[['comp','colour']] = rrngs[['comp','colour']].astype(str)
    return ions, rrngs
                          
def read_apt_to_df(folder):
    """
    Read the data 
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    """
    df_Mass_POS_lst = []
    file_name_lst=[]
    
    for filename in tqdm(os.listdir(folder)):
        if filename.lower().endswith(".pos"):
            path = os.path.join(folder, filename)            
            pos = read_pos(path)
            df_POS_MASS = pd.DataFrame({'x':pos['x'],'y': pos['y'],'z': pos['z'],'Da': pos['m']})
            df_Mass_POS_lst.append(df_POS_MASS)
            file_name_lst.append(filename)
          
        elif filename.lower().endswith(".rrng"):
            path = os.path.join(folder, filename) 
            ions,rrngs = read_rrng(path)
            
    return (df_Mass_POS_lst, file_name_lst, ions, rrngs) 





def apt_df_2_big_chunks_arr(folder, NoSlices = 10):
    """
    Cut the data into specified portions
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    """
    df_lst, files,ions,rrngs= read_apt_2_df(folder)
    for file_idx in range(len(files)):

        Org_file =df_lst[file_idx]  

        atoms_spec = []
        c = np.unique(rrngs.comp.values)
        for i in range(len(c)):
            #print(c[i])

            range_element = rrngs[rrngs['comp']=='{}'.format(c[i])]
            total, Count = atom_filter(Org_file,range_element)
            #name = c[i]
            name = i 
            total["spec"] = [name for j in range(len(total))]
            atoms_spec.append(total)

        Df_atom_spec = pd.concat(atoms_spec)
        x_wu=Df_atom_spec
        sort_x = x_wu.sort_values(by=['z'])
        
        ## open hdf5 file 
        #hdf = pd.HDFStore("./file_{}_large_chunks.h5".format(files[file_idx].replace(".","_")))
        hdf = h5py.File("./file_{}_large_chunks_test_arr.h5".format(files[file_idx].replace(".","_")), "w")
        #print("./file_{}_large_chunks_test_arr.h5".format(files[file_idx].replace(".","_")))
        G1 = hdf.create_group("Group_xyz_Da_spec")
        G1.attrs["columns"] = ["x","y","z","Da","spec"]
        G1.attrs["spec_name_order"] = list(c)
        ##end
        
        sublength_x= abs((max(sort_x['z'])-min(sort_x['z']))/NoSlices)
        #print(sublength_x)
        start = min(sort_x['z'])
        end = min(sort_x['z']) +sublength_x
        for i in tqdm(range(NoSlices)):
            #temp = sort_x.iloc[start:end]
            #print(start)
            #print(end)
            temp = sort_x[sort_x['z'].between(start, end, inclusive=True)]

            #temp.to_csv('B3_Hi_ent_cubes/{}.csv'.format(i), index=False)
            #hdf.put("chunk_{}".format(i), temp, format = "table", data_columns= True)
            ##Put data into hdf5 file
            G1.create_dataset("chunk_{}".format(i), data = temp.values)
            ##end
            
            start += sublength_x
            end += sublength_x
            #print(end) 
        hdf.close()
        
        
def Voxelisation(Size = 2 ):
    """
    
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    """
    import os
    #path = os.getcwd()+"\\file_R5096_64816_BFA.apt_large_chunks.h5"
    #hdfr = pd.HDFStore(path, mode = "r")

    for filename in tqdm(os.listdir(os.getcwd())):    
            if (filename.endswith("apt_large_chunks_test_arr.h5")):
                #hdfr = pd.HDFStore(filename, mode = "r")
                #hdf = pd.HDFStore("./file_{}_small_chunks.h5".format(filename.replace(".","_")))
                hdfr = h5py.File(filename, "r")
                with h5py.File("./file_{}_small_chunks_test_arr4.h5".format(filename.replace(".","_")), "w") as hdfw:

                    Group_r = hdfr.get("Group_xyz_Da_spec")
                    Group_keys = list(Group_r.keys())
                    columns_r = list(list(Group_r.attrs.values())[0])

                    #G1 = hdfw.create_group("Group_sm_vox_xyz_Da_spec")
                    G1 = hdfw.create_group("0")
                    prev_attri =list(list(Group_r.attrs.values())[0])
                    prev_attri.append("vox_file")
                    G1.attrs["columns"] =  prev_attri
                    G1.attrs["spec_name_order"] = list(list(Group_r.attrs.values())[1])

                    cube_size = Size #2
                    #file_to_h5 = []
                    name_sub_file = 0
                    step = 0
                    m=0
                    for key in tqdm(Group_keys):
                        print(key)
                        read_array = np.array(Group_r.get(key))

                        s= pd.DataFrame(data = read_array, columns =  columns_r)
                        x_min = round(min(s['x']))
                        x_max = round(max(s['x']))
                        y_min = round(min(s['y']))
                        y_max = round(max(s['y']))
                        z_min = round(min(s['z']))
                        z_max = round(max(s['z']))   
                        p=[]
                        x=[]

                        for i in range(z_min,z_max,cube_size):
                            print(i)
                            cubic = s[s['z'].between(i, i+cube_size, inclusive=True)]
                            for j in range(y_min,y_max,cube_size):
                                p = cubic[cubic['y'].between(j, j+cube_size, inclusive=True)]
                        #        print(j)
                                for k in range(x_min, x_max, cube_size):
                    #                print(k)
                        #            print(k+5)

                                    x = p[p['x'].between(k,k+cube_size, inclusive=True)]
                                    if len(x['x'])>20:
                                        name ='cubes_z{}_x{}_y{}'.format(i,j,k).replace("-","m")
                                        if step>99999:
                                            print(name_sub_file)
                                            step=0
                                            m=m+1
                                            G1 = hdfw.create_group("{}".format(100000*m))

                                        x["vox_file"] = [name_sub_file for n_file in range(len(x))]
                                        G1.create_dataset("{}".format(name_sub_file), data = x.values)
                                        name_sub_file = name_sub_file+1
                                        step=step+1
                                        #file_to_h5.append("{}".format(name_sub_file))
                    G1 = hdfw.get("0")
                    G1.attrs["Total_voxels"]="{}".format(name_sub_file)
                hdfr.close()
                    #hdfw.close()
                    
                    
                    
def VoxelComposition():
    
    import os
    Small_chunk_file_name = "./file_file_D1 High Hc R5076_52126_apt_large_chunks_test_arr_h5_small_chunks_test_arr4.h5"
    hdf_sm_r = h5py.File(Small_chunk_file_name,"r")
    group = hdf_sm_r.get("0")
    Total_Voxels =list(list(group.attrs.values())[0])
    SpecLstLen = len(list(list(group.attrs.values())[2]))

    items = list(hdf_sm_r.items())
    item_lst = []
    for item in range(len(items)):
        item_lst.append([100000*(item),100000*(item+1)])
    item_lst = np.array(item_lst)
    
    #Total_Voxels[0]+Total_Voxels[1]
    Total_Voxels_int =""
    for number in Total_Voxels:
        Total_Voxels_int = Total_Voxels_int+ number

    Total_Voxels_int = int(Total_Voxels_int)

    #files = ["Group_sm_vox_xyz_Da_spec/"+"{}".format(file_num) for file_num in range(905667)]
    files = [file_num for file_num in range(Total_Voxels_int)]
    
    import time
    Spec_names =  np.arange(SpecLstLen)
    Dic_ratios = {}
    for Spec_name in Spec_names:
        Dic_ratios["{}".format(Spec_name)]=[]

    Dic_ratios["Total_no"]=[]
    Dic_ratios["file_name"]=[]
    Dic_ratios["vox"] = []

    Ratios = []
    f_count = 0
    for filename in tqdm(files):
        group = np.min(item_lst[[filename in range(j[0],j[1]) for j in item_lst]])
        #print(group)
        arr = np.array(hdf_sm_r.get("{}/{}".format(group,filename))[:,4])
        N_x = len(arr)

        #start = time.time()
        for Spec in (Spec_names):
            #print(Spec)

            ratio = (len(np.argwhere(arr==Spec)))/N_x
            Dic_ratios["{}".format(Spec)].append(ratio)

        #end = time.time()
        #print(end - start)


        Dic_ratios["file_name"].append(filename)
        Dic_ratios["vox"].append(f_count)
        Dic_ratios["Total_no"].append(N_x)
        f_count = f_count+1
        
    df = pd.DataFrame.from_dict(Dic_ratios)

    
    with h5py.File("3Vox_ratios_filenames_num_MR_Grp.h5", "w") as hdfw:
        hdfw.create_dataset("vox_ratios", data =df.drop("file_name", axis = 1).values )
        hdfw.attrs["what"] = ["All the Vox ratios for a given APT smaple"]
        hdfw.attrs["howto_Group_name"] = ["Group_sm_vox_xyz_Da_spec/"]
        hdfw.attrs["colomns"]= ['0.0', '1.0', '2.0', '3.0', '4.0', 'Total_no', 'vox']

    hdf_sm_r.close()
    
    
    
def centeroid_df(data_frame):
    length = len(data_frame['x'])
    sum_x = np.sum(data_frame['x'])
    sum_y = np.sum(data_frame['y'])
    sum_z = np.sum(data_frame['z'])
    return sum_x/length, sum_y/length, sum_z/length

def centeroid_np_array(array):
    length = len(array[:,0])
    sum_x = np.sum(array[:,0])
    sum_y = np.sum(array[:,1])
    sum_z = np.sum(array[:,2])
    return sum_x/length, sum_y/length, sum_z/length


def PCA_cumsum(VoxRatioFile, VoxFile ):
    from sklearn.decomposition import PCA

    with h5py.File(VoxFile,"r") as hdf:
        group = hdf.get("Group_sm_vox_xyz_Da_spec")
        group0 = hdf.get("0")
        spec_lst = list(list(group0 .attrs.values())[2])
    #print(spec_lst)
    
    with h5py.File(VoxRatioFile , "r") as hdfr:
        Ratios = np.array(hdfr.get("vox_ratios"))
        Ratios_colomns = list(list(hdfr.attrs.values())[0])
        Group_name = list(list(hdfr.attrs.values())[1])
        
    Ratios =pd.DataFrame(data=Ratios, columns=Ratios_colomns)   

    X_train=Ratios.drop(['Total_no','vox'], axis=1)
    PCAObj = PCA(n_components = len(spec_lst)) 
    PCATrans = PCAObj.fit_transform(X_train)
    PCAObj.explained_variance_
    PCACumsumArr = np.cumsum(PCAObj.explained_variance_ratio_)
    return PCACumsumArr, Ratios

def NumClusters(VoxRatioFile, VoxFile, NumClustersVal ):
    from sklearn.mixture import GaussianMixture

    PCACumsumArr, Ratios = PCA_cumsum(VoxRatioFile, VoxFile )
    #import tqdm
    gm_scores=[]
    aics=[]
    bics=[]
    X = Ratios.drop(['Total_no','vox'], axis=1)
    n_clusters=list(range(1,NumClustersVal))
    for n_cluster in tqdm(n_clusters):
        gm = GaussianMixture(n_components=n_cluster,verbose=0)
        gm.fit(X)
        y_pred=gm.predict(X)
        #gm_scores.append(homogeneity_score(y,y_pred))
        aics.append(gm.aic(X))
        bics.append(gm.bic(X))
        
    return NumClustersVal, aics, bics


def VoxRead(VoxFile, FilesArr):

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
        Df_centroids = pd.DataFrame(columns=['x', 'y', 'z','filename'])
        for filename in tqdm(FilesArr):
            #file_name_grp = "Group_sm_vox_xyz_Da_spec/"+"{}".format(int(i))
            group = np.min(item_lst[[filename in range(j[0],j[1]) for j in item_lst]])
            xyz_Da_spec_atoms = np.array(hdf.get("{}/{}".format(group,filename)))
            x,y,z = centeroid_np_array(xyz_Da_spec_atoms)
            Dic_centroids["x"].append(x)
            Dic_centroids["y"].append(y)
            Dic_centroids["z"].append(z)
            Dic_centroids["file_name"].append(filename)
            #Df_centroids = Df_centroids.append({'x': x, 'y': y, 'z': z, 'filename' : i}, ignore_index=True)
            
    return Dic_centroids


def CompositionCluster(VoxRatioFile, VoxFile, n_components):
    from sklearn.mixture import GaussianMixture

    PCACumsumArr, Ratios = PCA_cumsum(VoxRatioFile, VoxFile )
    X = Ratios.drop(['Total_no','vox'], axis=1)
    
    gm = GaussianMixture(n_components=3, max_iter=100000,verbose=0)
    gm.fit(X)
    y_pred=gm.predict(X)
    
    cluster_0 = np.argwhere(y_pred == 0)
    cluster_1 = np.argwhere(y_pred == 1)
    cluster_2 = np.argwhere(y_pred == 2)
    
    df_dis2 = Ratios.iloc[cluster_2.flatten()]
    df_dis1 = Ratios.iloc[cluster_1.flatten()]
    df_dis0 = Ratios.iloc[cluster_0.flatten()]
    return df_dis2, df_dis1,df_dis0, cluster_0,cluster_1,cluster_2,Ratios
    
def SaveCompositionClusters(VoxRatioFile, VoxFile, OutFile,n_components):
    df_dis2, df_dis1,df_dis0, cluster_0,cluster_1,cluster_2,Ratios = CompositionCluster(VoxRatioFile, VoxFile, n_components)
    
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
    import os
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
        
        CentroidsDic = VoxRead(VoxFile, FilesArr= FilesList[i])
        GroupsList[i].create_dataset("{}".format(PhasesNames[i]), data = pd.DataFrame.from_dict(CentroidsDic).values)
    hdfw.close()
