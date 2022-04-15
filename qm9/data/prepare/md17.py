from os.path import join as join
import urllib.request

import numpy as np
import torch

import logging, os, urllib

from qm9.data.prepare.utils import download_data, is_int, cleanup_file

md17_base_url = 'http://quantum-machine.org/gdml/data/npz/'

md17_subsets = {'benzene': 'benzene_old_dft',
               'uracil': 'uracil_dft',
               'naphthalene': 'naphthalene_dft',
               'aspirin': 'aspirin_dft',
               'salicylic_acid': 'salicylic_dft',
               'malonaldehyde': 'malonaldehyde_dft',
               'ethanol': 'ethanol_dft',
               'toluene': 'toluene_dft',
               'paracetamol': 'paracetamol_dft',
               'azobenzene': 'azobenzene_dft'
               }

def download_dataset_md17(datadir, dataname, subset, splits=None, cleanup=True):
    """
    Downloads the MD17 dataset.
    """
    if subset not in md17_subsets:
        logging.info('Molecule {} not included in list of downloadable MD17 datasets! Attempting to download based directly upon input key.'.format(subset))
        md17_molecule = subset
    else:
        md17_molecule = md17_subsets[subset]

    # Define directory for which data will be output.
    md17dir = join(*[datadir, dataname, subset])

    # Important to avoid a race condition
    os.makedirs(md17dir, exist_ok=True)

    logging.info('Downloading and processing molecule {} from MD17 dataset. Output will be in directory: {}.'.format(subset, md17dir))

    md17_data_url = md17_base_url + md17_molecule + '.npz'
    md17_data_npz = join(md17dir, md17_molecule + '.npz')

    download_data(md17_data_url, outfile=md17_data_npz, binary=True)

    # Convert raw MD17 data to torch tensors.
    md17_raw_data = np.load(md17_data_npz)

    # Number of molecules in dataset:
    num_tot_mols = len(md17_raw_data['E'])

    # Dictionary to convert keys in MD17 database to those used in this code.
    md17_keys = {'E': 'energies', 'R': 'positions', 'F': 'forces'}

    # Convert numpy arrays to torch.Tensors
    md17_data = {new_key: md17_raw_data[old_key] for old_key, new_key in md17_keys.items()}

    # Reshape energies to remove final singleton dimension
    md17_data['energies'] = md17_data['energies'].squeeze(1)

    # Add charges to md17_data
    md17_data['charges'] = np.tile(md17_raw_data['z'], (num_tot_mols, 1))

    # If splits are not specified, automatically generate them.
    if splits is None:
        splits = gen_splits_md17(num_tot_mols)

    # Process GDB9 dataset, and return dictionary of splits
    md17_data_split = {}
    for split, split_idx in splits.items():
        md17_data_split[split] = {key: val[split_idx] if type(val) is np.ndarray else val for key, val in md17_data.items()}

    # Save processed GDB9 data into train/validation/test splits
    logging.info('Saving processed data:')
    for split, data_split in md17_data_split.items():
        savefile = join(md17dir, split + '.npz')
        np.savez_compressed(savefile, **data_split)

    cleanup_file(md17_data_npz, cleanup)


def gen_splits_md17(num_pts):
    """
    Generate the splits used to train/evaluate the network in the original Cormorant paper.
    """
    # deterministically generate random split based upon random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(num_pts)

    # Create masks for which splits to invoke
    mask_train = np.zeros(num_pts, dtype=np.bool)
    mask_valid = np.zeros(num_pts, dtype=np.bool)
    mask_test = np.zeros(num_pts, dtype=np.bool)

    # For historical reasons, this is the indexing on the
    # 50k/10k/10k train/valid/test splits used in the paper.
    mask_train[:10000] = True
    mask_valid[10000:20000] = True
    mask_test[20000:30000] = True
    mask_train[30000:70000] = True

    # COnvert masks to splits
    splits = {}
    splits['train'] = torch.tensor(data_perm[mask_train])
    splits['valid'] = torch.tensor(data_perm[mask_valid])
    splits['test'] = torch.tensor(data_perm[mask_test])

    return splits
