#from Segmentation.tools import paraprobe_transcoder
import compositionspace.paraprobe_transcoder as paraprobe_transcoder
import pandas as pd 
import re
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import h5py    
import time

class Prepare_data():
    
    def __init__(self,params):
        self.params = params
        self.version="1.0.0"

        
    def get_pos(self,file_name):
        f = open(file_name, 'rb') # reading in binary (SA).
        dt_type = np.dtype({'names':['x', 'y', 'z', 'm'], 
                      'formats':['>f4', '>f4', '>f4', '>f4']})
        pos = np.fromfile(f, dt_type, -1)
        f.close()
        return pos

    
    def get_label_ions(self,pos,rrngs):
        pos['comp'] = ''
        pos['colour'] = '#FFFFFF'
        pos['nature'] = ''
        count=0;
        for n,r in rrngs.iterrows():
            count= count+1;
            pos.loc[(pos.Da >= r.lower) & (pos.Da <= r.upper),['comp','colour', 'nature']] = [r['comp'],'#' + r['colour'],count]
        return pos


    def atom_filter(self,x, Atom_range):
        Atom_total = pd.DataFrame()
        for i in range(len(Atom_range)):
            Atom = x[x['Da'].between(Atom_range['lower'][i], Atom_range['upper'][i], inclusive=True)]
            Atom_total = Atom_total.append(Atom)
            Count_Atom= len(Atom_total['Da'])   
        return Atom_total, Count_Atom          
        
 
    def get_rrng(self,f):
        rf = open(f,'r').readlines()
        patterns = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')
        ions = []
        rrngs = []
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

    
    def get_apt_dataframe(self,folder):
        df_Mass_POS_lst = []
        file_name_lst=[]
        for filename in tqdm(os.listdir(folder)):
            if filename.endswith(".pos"):
                #print(filename)
                PATH = folder+'/'+filename
                pos = self.get_pos(PATH)
                df_POS_MASS = pd.DataFrame({'x':pos['x'],'y': pos['y'],'z': pos['z'],'Da': pos['m']})
                df_Mass_POS_lst.append(df_POS_MASS)
                file_name_lst.append(filename)
            if filename.endswith(".apt"):
                #print(filename)
                PATH = folder+'/'+filename
                apt = paraprobe_transcoder.paraprobe_transcoder( PATH )
                apt.read_cameca_apt()
                POS = apt.Position
                MASS = apt.Mass
                POS_MASS = np.concatenate((POS,MASS),axis = 1)
                df_POS_MASS = pd.DataFrame(POS_MASS, columns = ["x","y","z","Da"])
                df_Mass_POS_lst.append(df_POS_MASS)
                file_name_lst.append(filename)
            if filename.endswith(".RRNG"):
                rrange_file = folder +"/"+filename
                #print(rrange_file)
                ions,rrngs = self.get_rrng(rrange_file)
        return(df_Mass_POS_lst,file_name_lst,ions,rrngs ) 

    
    def get_big_slices(self): #folder
        df_lst, files,ions,rrngs= self.get_apt_dataframe(self.params["input_path"])# folder
        #print("hi")
        for file_idx in range(len(files)):
            Org_file =df_lst[file_idx]  
            atoms_spec = []
            c = np.unique(rrngs.comp.values)
            for i in range(len(c)):
                #print(c[i])
                range_element = rrngs[rrngs['comp']=='{}'.format(c[i])]
                total, Count = self.atom_filter(Org_file,range_element)
                #name = c[i]
                name = i 
                total["spec"] = [name for j in range(len(total))]
                atoms_spec.append(total)
            Df_atom_spec = pd.concat(atoms_spec)
            x_wu=Df_atom_spec
            sort_x = x_wu.sort_values(by=['z'])
            ## open hdf5 file 
            #hdf = pd.HDFStore("./file_{}_large_chunks.h5".format(files[file_idx].replace(".","_")))
            #output_path = self.params['output_path'] + "/Output_{}_large_chunks_test_arr.h5".format(files[file_idx].replace(".","_"))
            output_path = self.params['output_path'] + "/Output_big_slices.h5"
            hdf = h5py.File(output_path, "w")
            #print("./file_{}_large_chunks_test_arr.h5".format(files[file_idx].replace(".","_")))
            G1 = hdf.create_group("Group_xyz_Da_spec")
            G1.attrs["columns"] = ["x","y","z","Da","spec"]
            G1.attrs["spec_name_order"] = list(c)
            sublength_x= abs((max(sort_x['z'])-min(sort_x['z']))/self.params["n_big_slices"])
            #print(sublength_x)
            start = min(sort_x['z'])
            end = min(sort_x['z']) +sublength_x
            for i in tqdm(range(self.params["n_big_slices"])):
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




    def get_big_slices_molecules(self):
        df_lst, files,ions,rrngs = self.get_apt_dataframe(self.params["input_path"])
        for file_idx in range(len(files)):
            Org_file =df_lst[file_idx]  

            atoms_spec = []
            c = np.unique(rrngs.comp.values)
            for i in range(len(c)):
                print(c[i])

                range_element = rrngs[rrngs['comp']=='{}'.format(c[i])]
                total, Count = atom_filter(Org_file,range_element)
                name = c[i]
                #name = i 
                total["spec"] = [name for j in range(len(total))]
                atoms_spec.append(total)

            Df_atom_spec = pd.concat(atoms_spec)
    #############################
            #check molecules:
            print("MOLECULE CHECK")
            molecule_check = np.array([len(c[i].split(" ")) for i in range(len(c))])
            molecules = c[np.argwhere(molecule_check == 2)]

            Df_lst = [] 
            for mol in molecules:
                spec_mol = mol[0].split(" ")
                Df = Df_atom_spec.loc[Df_atom_spec['spec'] == mol[0]].copy()
                Df["spec"] = [spec_mol[1]]*len(Df)

                Df_atom_spec.loc[Df_atom_spec['spec'] == mol[0], ['spec'] ] = spec_mol[0]

                Df_lst.append(Df)
            Df_lst.append(Df_atom_spec) 
            Df_atom_spec = pd.concat(Df_lst)

            SpecSemi = np.unique(Df_atom_spec.spec.values)

            #check doubles
            print(SpecSemi)
            mol_doub_check = np.array([int(SpecSemi[j].split(":")[1]) for j in range(len(SpecSemi))])
            mol_doub = SpecSemi[np.argwhere(mol_doub_check == 2)] 

            Df_lst = []
            for mol in mol_doub:
                spec_mol = mol[0].split(":")[0]+":1"
                Df = Df_atom_spec.loc[Df_atom_spec['spec'] == mol[0]].copy()
                Df["spec"] = [spec_mol]*len(Df)

                Df_atom_spec.loc[Df_atom_spec['spec'] == mol[0], ['spec'] ] = spec_mol

                Df_lst.append(Df)
            Df_lst.append(Df_atom_spec) 
            Df_atom_spec = pd.concat(Df_lst)
            SpecFinal = np.unique(Df_atom_spec.spec.values)

            Df_spec_lst = []
            for spec_ID in range(len(SpecFinal)):
                print( SpecFinal[spec_ID])

                Df = Df_atom_spec.loc[Df_atom_spec['spec'] == SpecFinal[spec_ID]].copy()
                name = spec_ID 
                Df["spec"] = [name for j in range(len(Df))]
                Df_spec_lst.append(Df)
            Df_atom_spec = pd.concat(Df_spec_lst)        


    ############################        

            x_wu=Df_atom_spec
            sort_x = x_wu.sort_values(by=['z'])

            ## open hdf5 file 
            #hdf = pd.HDFStore("./file_{}_large_chunks.h5".format(files[file_idx].replace(".","_")))
            
            output_path = self.params['output_path'] + "/Output_big_slices.h5"
            hdf = h5py.File(output_path, "w")          
            G1 = hdf.create_group("Group_xyz_Da_spec")
            G1.attrs["columns"] = ["x","y","z","Da","spec"]
            G1.attrs["spec_name_order"] = list(SpecFinal)
            ##end

            sublength_x= abs((max(sort_x['z'])-min(sort_x['z']))/self.params["n_big_slices"])
            print(sublength_x)
            start = min(sort_x['z'])
            end = min(sort_x['z']) +sublength_x
            for i in tqdm(range(self.params["n_big_slices"])):
                #temp = sort_x.iloc[start:end]
                print(start)
                print(end)
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

            
            
          
            
    def get_voxels(self):

        #path = os.getcwd()+"\\file_R5096_64816_BFA.apt_large_chunks.h5"
        #hdfr = pd.HDFStore(path, mode = "r")
        folder = self.params['output_path']
        for filename in tqdm(os.listdir(folder)):    
                if (filename.endswith("slices.h5")):
                    #hdfr = pd.HDFStore(filename, mode = "r")
                    #hdf = pd.HDFStore("./file_{}_small_chunks.h5".format(filename.replace(".","_")))
                    hdfr = h5py.File(folder + "/" +filename, "r")
                    
                    #output_path = self.params['output_path'] + "/Output_{}_voxels.h5".format(filename.replace(".","_"))
                    output_path = self.params['output_path'] + "/Output_voxels.h5"
                    with h5py.File(output_path, "w") as hdfw:

                        Group_r = hdfr.get("Group_xyz_Da_spec")
                        Group_keys = list(Group_r.keys())
                        columns_r = list(list(Group_r.attrs.values())[0])

                        #G1 = hdfw.create_group("Group_sm_vox_xyz_Da_spec")
                        G1 = hdfw.create_group("0")
                        prev_attri =list(list(Group_r.attrs.values())[0])
                        prev_attri.append("vox_file")
                        G1.attrs["columns"] =  prev_attri
                        G1.attrs["spec_name_order"] = list(list(Group_r.attrs.values())[1])

                        cube_size = self.params["voxel_size"] #2
                        #file_to_h5 = []
                        name_sub_file = 0
                        step = 0
                        m=0
                        for key in tqdm(Group_keys):
                            #print(key)
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
                                #print(i)
                                cubic = s[s['z'].between(i, i+cube_size, inclusive=True)]
                                for j in range(y_min,y_max,cube_size):
                                    p = cubic[cubic['y'].between(j, j+cube_size, inclusive=True)]
                            #        #print(j)
                                    for k in range(x_min, x_max, cube_size):
                        #                #print(k)
                            #            #print(k+5)

                                        x = p[p['x'].between(k,k+cube_size, inclusive=True)]
                                        if len(x['x'])>20:
                                            name ='cubes_z{}_x{}_y{}'.format(i,j,k).replace("-","m")
                                            if step>99999:
                                                #print(name_sub_file)
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



    def get_voxel_composition(self):
    
        folder = self.params['output_path']
        for filename in tqdm(os.listdir(folder)):    
                if (filename.endswith("voxels.h5")):
                    Small_chunk_file_name = folder + "/" +filename
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
        output_path = self.params['output_path'] + "/Output_voxel_composition.h5"
        with h5py.File(output_path, "w") as hdfw:
            hdfw.create_dataset("vox_ratios", data =df.drop("file_name", axis = 1).values )
            hdfw.attrs["what"] = ["All the Vox ratios for a given APT smaple"]
            hdfw.attrs["howto_Group_name"] = ["Group_sm_vox_xyz_Da_spec/"]
            hdfw.attrs["colomns"]= ['0.0', '1.0', '2.0', '3.0', '4.0', 'Total_no', 'vox']
        hdf_sm_r.close()


            
            
 