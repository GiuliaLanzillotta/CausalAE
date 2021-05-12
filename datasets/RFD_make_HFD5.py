""" Python script to convert RFD POSIX storage into HDF5
Note: if you want to use the Random Access RFD dataset you first need to download the .tar archive
and run this script."""
from datasets import RFDtoHDF5


if __name__ == '__main__':

    toHDF5 = RFDtoHDF5(read_root='./robot_finger_datasets/',
                       save_root='./robot_finger_datasets/',
                       chunksize=32, heldout_colors=False)
    toHDF5(overwrite=True, num_files=10)
    toHDF5_heldout = RFDtoHDF5(read_root='./robot_finger_datasets/',
                               save_root='./robot_finger_datasets/',
                               chunksize=32, heldout_colors=True)
    toHDF5_heldout(overwrite=True, num_files=2)