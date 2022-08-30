import pandas as pd
import re
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import pickle
import time
#import paraprobe_transcoder
import h5py
import warnings
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

#really check this!
pd.options.mode.chained_assignment = None

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
def atom_filter(x, atom_range):
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
    dfs = []
    for i in range(len(atom_range)):
        atom = x[x['Da'].between(atom_range['lower'][i], atom_range['upper'][i], inclusive="both")]
        dfs.append(atom)
    atom_total = pd.concat(dfs)
    count_Atom= len(atom_total['Da'])   
    return atom_total, count_Atom  


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
        pos = pos.byteswap().newbyteorder()
    
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



def chunkify_apt_df(folder, no_of_slices = 10, prefix=None):
    """
    Cut the data into specified portions
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    """
    df_lst, files, ions, rrngs= read_apt_to_df(folder)
    filestrings = []
    prefix = os.getcwd() if prefix is None else prefix
    
    for idx, file in enumerate(files):
        org_file = df_lst[idx]
        atoms_spec = []
        c = np.unique(rrngs.comp.values)
        for i in range(len(c)):
            range_element = rrngs[rrngs['comp']=='{}'.format(c[i])]
            total, count = atom_filter(org_file, range_element)
            total["spec"] = [x for x in range(len(total))]
            atoms_spec.append(total)

        df_atom_spec = pd.concat(atoms_spec)
        sorted_df = df_atom_spec.sort_values(by=['z'])

        filestring = "file_{}_large_chunks_arr.h5".format(file.replace(".","_"))
        filestring = os.path.join(prefix, filestring)
        filestrings.append(filestring)

        hdf = h5py.File(filestring, "w")
        group1 = hdf.create_group("group_xyz_Da_spec")
        group1.attrs["columns"] = ["x","y","z","Da","spec"]
        group1.attrs["spec_name_order"] = list(c)
        sublength_x= abs((max(sorted_df['z'])-min(sorted_df['z']))/no_of_slices)
        start = min(sorted_df['z'])
        end = min(sorted_df['z']) + sublength_x
        
        for i in tqdm(range(no_of_slices)):
            temp = sorted_df[sorted_df['z'].between(start, end, inclusive="both")]
            group1.create_dataset("chunk_{}".format(i), data = temp.values)
            start += sublength_x
            end += sublength_x 
        hdf.close()                

    return filestrings 

        
def voxelise(filenames, size = 2, prefix=None):
    """
    
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----
    """
    filestrings = []
    prefix = os.getcwd() if prefix is None else prefix

    for filename in filenames:
        hdfr = h5py.File(filename, "r")
        filestring = filename.replace("large", "small")
        filestring = os.path.join(prefix, filestring)
        filestrings.append(filestring)

        with h5py.File(filestring, "w") as hdfw:
            group_r = hdfr.get("group_xyz_Da_spec")
            group_keys = list(group_r.keys())
            columns_r = list(list(group_r.attrs.values())[0])

            group1 = hdfw.create_group("0")
            prev_attri =list(list(group_r.attrs.values())[0])
            prev_attri.append("vox_file")
            group1.attrs["columns"] =  prev_attri
            group1.attrs["spec_name_order"] = list(list(group_r.attrs.values())[1])

            name_sub_file = 0
            step = 0
            m=0

            for key in tqdm(group_keys):
                read_array = np.array(group_r.get(key))
                s= pd.DataFrame(data = read_array, columns =  columns_r)
                x_min = round(min(s['x']))
                x_max = round(max(s['x']))
                y_min = round(min(s['y']))
                y_max = round(max(s['y']))
                z_min = round(min(s['z']))
                z_max = round(max(s['z']))   
                p=[]
                x=[]

                for i in range(z_min, z_max, size):
                    cubic = s[s['z'].between(i, i+size, inclusive="both")]
                    for j in range(y_min, y_max, size):
                        p = cubic[cubic['y'].between(j, j+size, inclusive="both")]
                        for k in range(x_min, x_max, size):
                            x = p[p['x'].between(k, k+size, inclusive="both")]
                            if len(x['x'])>20:
                                #warnings.warn("I am running some code with hardcoded numbers. Really recheck what's up here!")
                                name ='cubes_z{}_x{}_y{}'.format(i,j,k).replace("-","m")
                                if step>99999:
                                    step=0
                                    m=m+1
                                    group1 = hdfw.create_group("{}".format(100000*m))

                                x["vox_file"] = [name_sub_file for n_file in range(len(x))]
                                group1.create_dataset("{}".format(name_sub_file), data = x.values)
                                name_sub_file = name_sub_file+1
                                step=step+1
            group1 = hdfw.get("0")
            group1.attrs["total_voxels"]="{}".format(name_sub_file)

    return filestrings


                   
                    
def calculate_voxel_composition(small_chunk_file_name, outfilename="3Vox_ratios_filenames_num_MR_Grp.h5"):
    """
    This works only a single filename; check
    """
    hdf_sm_r = h5py.File(small_chunk_file_name, "r")
    group = hdf_sm_r.get("0")
        
    #SRM: changed to indice 2 for the first one. Please check.
    total_voxels =list(list(group.attrs.values())[2])
    spec_lst_len = len(list(list(group.attrs.values())[2]))

    items = list(hdf_sm_r.items())
    item_lst = []
    ### CHECK THESE NUMBERS!
    for item in range(len(items)):
        item_lst.append([100000*(item), 100000*(item+1)])
    item_lst = np.array(item_lst)
    
    
    total_voxels_int =""
    for number in total_voxels:
        total_voxels_int = total_voxels_int + number

    total_voxels_int = int(total_voxels_int)

    files = [file_num for file_num in range(total_voxels_int)]
    
    
    spec_names =  np.arange(spec_lst_len)
    dic_ratios = {}
    for spec_name in spec_names:
        dic_ratios["{}".format(spec_name)] = []

    dic_ratios["Total_no"]=[]
    dic_ratios["file_name"]=[]
    dic_ratios["vox"] = []

    ratios = []
    f_count = 0
    for filename in tqdm(files):
        group = np.min(item_lst[[filename in range(j[0],j[1]) for j in item_lst]])
        arr = np.array(hdf_sm_r.get("{}/{}".format(group,filename))[:,4])
        N_x = len(arr)

        for spec in (spec_names):
            ratio = (len(np.argwhere(arr==spec)))/N_x
            dic_ratios["{}".format(spec)].append(ratio)

        dic_ratios["file_name"].append(filename)
        dic_ratios["vox"].append(f_count)
        dic_ratios["Total_no"].append(N_x)
        f_count = f_count+1
        
    df = pd.DataFrame.from_dict(dic_ratios)

    
    with h5py.File(outfilename, "w") as hdfw:
        hdfw.create_dataset("vox_ratios", data =df.drop("file_name", axis = 1).values )
        hdfw.attrs["what"] = ["All the Vox ratios for a given APT smaple"]
        hdfw.attrs["howto_Group_name"] = ["Group_sm_vox_xyz_Da_spec/"]
        hdfw.attrs["columns"]= ['0.0', '1.0', '2.0', '3.0', '4.0', 'Total_no', 'vox']

    hdf_sm_r.close()
    
    
    
def calculate_centroid(data):
    """
    Calculate centroid

    Parameters
    ----------
    data: pandas DataFrame or numpy array

    Returns
    -------
    centroid
    """
    if isinstance(data, pd.DataFrame):
        length = len(data['x'])
        sum_x = np.sum(data['x'])
        sum_y = np.sum(data['y'])
        sum_z = np.sum(data['z'])
        return sum_x/length, sum_y/length, sum_z/length
    else:
        length = len(data[:,0])
        sum_x = np.sum(data[:,0])
        sum_y = np.sum(data[:,1])
        sum_z = np.sum(data[:,2])
        return sum_x/length, sum_y/length, sum_z/length


def calculate_PCA_cumsum(vox_ratio_file, vox_file):

    with h5py.File(vox_file,"r") as hdf:
        group = hdf.get("Group_sm_vox_xyz_Da_spec")
        group0 = hdf.get("0")
        spec_lst = list(list(group0.attrs.values())[2])
    
    with h5py.File(vox_ratio_file , "r") as hdfr:
        ratios = np.array(hdfr.get("vox_ratios"))
        ratios_columns = list(list(hdfr.attrs.values())[0])
        group_name = list(list(hdfr.attrs.values())[1])
        
    ratios = pd.DataFrame(data=ratios, columns=ratios_columns)   

    X_train = ratios.drop(['Total_no', 'vox'], axis=1)
    
    PCAObj = PCA(n_components = len(spec_lst)) 
    PCATrans = PCAObj.fit_transform(X_train)
    PCACumsumArr = np.cumsum(PCAObj.explained_variance_ratio_)
    
    return PCACumsumArr, ratios

def find_num_clusters(vox_ratio_file, vox_file, num_clusters):
    

    PCACumsumArr, Ratios = calculate_PCA_cumsum(vox_ratio_file, vox_file)
    
    gm_scores=[]
    aics=[]
    bics=[]
    X = Ratios.drop(['Total_no','vox'], axis=1)
    n_clusters=list(range(1, num_clusters))
    
    for n_cluster in tqdm(n_clusters):
        gm = GaussianMixture(n_components=n_cluster, verbose=0)
        gm.fit(X)
        y_pred=gm.predict(X)
        aics.append(gm.aic(X))
        bics.append(gm.bic(X))
        
    return aics, bics


def get_centroid_from_vox(vox_file, files_arr):

    with h5py.File(vox_file, "r") as hdf:
        items = list(hdf.items())
        item_lst = []
        for item in range(len(items)):
            item_lst.append([100000*(item),100000*(item+1)])
        item_lst = np.array(item_lst)

        dic_centroids = {}
        dic_centroids["x"]=[]
        dic_centroids["y"]=[]
        dic_centroids["z"] = []
        dic_centroids["file_name"] = []
        df_centroids = pd.DataFrame(columns=['x', 'y', 'z','filename'])
        
        for filename in tqdm(files_arr):
            group = np.min(item_lst[[filename in range(j[0],j[1]) for j in item_lst]])
            xyz_Da_spec_atoms = np.array(hdf.get("{}/{}".format(group, filename)))
            x, y, z = calculate_centroid(xyz_Da_spec_atoms)
            dic_centroids["x"].append(x)
            dic_centroids["y"].append(y)
            dic_centroids["z"].append(z)
            dic_centroids["file_name"].append(filename)            
    return dic_centroids


def find_cluster_composition(vox_ratio_file, vox_file, n_components, save=False, outfile=None):

    PCACumsumArr, Ratios = calculate_PCA_cumsum(vox_ratio_file, vox_file)
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

    if save:
        save_composition_clusters(df_dis2, df_dis1, df_dis0, cluster_0, cluster_1, cluster_2, Ratios, outfile, vox_file)

    return df_dis2, df_dis1,df_dis0, cluster_0,cluster_1,cluster_2, Ratios
    
def save_composition_clusters(df_dis2, df_dis1, df_dis0, cluster_0, cluster_1, cluster_2, Ratios, outfile, vox_file):
    
    plot_files_cl_0 = []
    plot_files_cl_1 = []
    plot_files_cl_2 = []

    for i in cluster_0.flatten():
        plot_files_cl_0.append(Ratios['vox'][i])
    for i in cluster_1.flatten():
        plot_files_cl_1.append(Ratios['vox'][i])

    for i in cluster_2.flatten():
        plot_files_cl_2.append(Ratios['vox'][i])
        
    plot_files_cl_0_group = [int(file_num) for file_num in plot_files_cl_0 ]
    plot_files_cl_1_group = [int(file_num) for file_num in plot_files_cl_1 ]
    plot_files_cl_2_group = [int(file_num) for file_num in plot_files_cl_2 ]
    
    hdfw = h5py.File(outfile,"w")

    G1 = hdfw.create_group("All")
    G1.attrs["what"] = ["Centroid of All voxels"]
    G1.attrs["howto_Group_name"] = ["Group_sm_vox_xyz_Da_spec/"]
    G1.attrs["columns"] = ["x","y","z","file_name"]

    G2 = hdfw.create_group("Phase1")
    G2.attrs["what"] = ["Centroid of Phase1 voxels"]
    G2.attrs["howto_Group_name"] = ["Group_sm_vox_xyz_Da_spec/"]
    G2.attrs["columns"] = ["x","y","z","file_name"]

    G3 = hdfw.create_group("Phase2")
    G3.attrs["what"] = ["Centroid of Phase2 voxels"]
    G3.attrs["howto_Group_name"] = ["Group_sm_vox_xyz_Da_spec/"]
    G3.attrs["columns"] = ["x","y","z","file_name"]
    
    hdf_sm_r = h5py.File(vox_file, "r")
    group = hdf_sm_r.get("0")
    total_voxels =list(list(group.attrs.values())[2])

    total_voxels_int =""
    for number in total_voxels:
        total_voxels_int = total_voxels_int + number

    total_voxels_int = int(total_voxels_int)
    hdf_sm_r.close()
    plot_files_cl_All_group = [file_num for file_num in range(total_voxels_int)]

    files_list = [plot_files_cl_All_group, plot_files_cl_1_group, plot_files_cl_2_group]
    groups_list = [G1, G2, G3]
    phases_names = ["All","Ph1", "Ph2"] 
    
    for i in range(len(files_list)):
        
        centroids_dic = get_centroid_from_vox(vox_file, files_arr = files_list[i])
        groups_list[i].create_dataset("{}".format(phases_names[i]), data = pd.DataFrame.from_dict(centroids_dic).values)
    hdfw.close()
