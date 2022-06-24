import pytest
from compositionspace import *
import numpy as np

def test_file_rrng():
    datarrng = read_rrng("tests/data/R31_06365-v02.rrng")
    assert datarrng[0]["name"].values[0] == "C"
    
def test_file_pos():
    datapos = read_pos("tests/data/R31_06365-v02.pos")
    assert np.isclose(datapos[0][0]+5.3784895, 0)

def test_file_df():
    data = read_apt_to_df("tests/data")
    assert np.isclose(data[0][0]["x"].values[0]+5.3784895, 0)
    assert data[1][0] == 'R31_06365-v02.pos'
    assert data[2]["name"].values[0] == "C"
    assert np.isclose(data[3]["lower"].values[0]-5.974, 0)
