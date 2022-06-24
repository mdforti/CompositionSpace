import pytest
from compositionspace import *
import numpy as np

def test_file_rrng():
    datarrng = read_rrng("tests/data/R31_06365-v02.rrng")
    assert datarrng[0]["name"].values[0] == "C"
    
def test_file_pos():
    datapos = read_pos("tests/data/R31_06365-v02.pos")
    assert np.isclose(datapos[0][0]+5.3784895, 0)