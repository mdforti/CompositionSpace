import pytest
from compositionspace.datautils import DataPreparation
import numpy as np
import os

def test_file_rrng():
    data = DataPreparation("tests/experiment_params.yaml")
    datarrng = data.read_rrng("tests/data/R31_06365-v02.rrng")
    assert datarrng[0]["name"].values[0] == "C"
    
def test_file_pos():
    data = DataPreparation("tests/experiment_params.yaml")
    datapos = data.read_pos("tests/data/R31_06365-v02.pos")
    assert np.isclose(datapos[0][0]+5.3784895, 0)

def test_file_df():
    data = DataPreparation("tests/experiment_params.yaml")
    data = data.read_apt_to_df()
    assert np.isclose(data[0][0]["x"].values[0]+5.3784895, 0)
    assert data[1][0] == 'R31_06365-v02.pos'
    assert data[2]["name"].values[0] == "C"
    assert np.isclose(data[3]["lower"].values[0]-5.974, 0)

def test_chunkify():
    data = DataPreparation("tests/experiment_params.yaml")
    data.chunkify_apt_df()
    assert os.path.exists("tests/output/file_R31_06365-v02_pos_large_chunks_arr.h5") == True

def test_voxelise():
    data = DataPreparation("tests/experiment_params.yaml")
    data.chunkify_apt_df()
    data.get_voxels()
    data.calculate_voxel_composition()
    assert os.path.exists("tests/output/file_R31_06365-v02_pos_vox_ratios_arr.h5") == True
