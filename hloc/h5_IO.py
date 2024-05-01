import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py


# h5
def recursive_list(group, indent=0):
            """ Recursively lists all groups and datasets in an HDF5 group """
            for key, item in group.items():
                print('  ' * indent + key, 'Group' if isinstance(item, h5py.Group) else 'Dataset')
                if isinstance(item, h5py.Group):
                    recursive_list(item, indent + 1)


def copy_group(src, dest):
    """
    Recursively copy a group from source HDF5 to destination HDF5.
    """
    for name, item in src.items():
        if isinstance(item, h5py.Dataset):
            # Copy dataset from source to destination
            if name in dest:
                del dest[name]  # Delete the existing dataset
            dest.copy(item, name)
        elif isinstance(item, h5py.Group):
            # Create new group in destination and copy its content
            if name not in dest:
                dest.create_group(name)
            copy_group(item, dest[name])


def merge_hdf5(source_file, target_file):
    """
    Merge two HDF5 files. All groups and datasets from source_file will be
    copied to target_file.
    """
    with h5py.File(source_file, 'r') as src, h5py.File(target_file, 'a') as dest:
        copy_group(src, dest)


# csv
def get_file_path(root_directory, file_name):
    for root, dirs, files in os.walk(root_directory):
        # print(root, dirs, files)
        if file_name in files:
            return os.path.join(root, file_name)
        

def vertices_from_csv(b_name, v_name="vertices.csv"):
    """
    Args:
        path: path to csv file
    """
    if not isinstance(b_name, str):
        b_name = str(b_name)

    path = get_file_path(f'/data/localization/maps/{b_name}', v_name)
    
    df = pd.read_csv(path)
    return df


def find_closest_img(df, pool):
    """
    Args:
        df: vertex timestamp
        pool: image timestamp
    """

    df.loc[:, 'img_name'] = df.apply(lambda x: pool[np.argmin(np.abs(pool - x.iloc[0]))], axis=1)
    df.loc[:, 'img_name'] = df.apply(lambda x: f"{x.iloc[1]}.png", axis=1)
    return df


if __name__ == "__main__":
    vertices_from_csv("20240404_032834")