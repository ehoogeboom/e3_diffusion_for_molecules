# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import pickle
import torch.nn.functional as F
from qm9.analyze import analyze_stability_for_molecules
import numpy as np
import torch


def flatten_sample_dictionary(samples):
    results = {'one_hot': [], 'x': [], 'node_mask': []}
    for number_of_atoms in samples:
        positions = samples[number_of_atoms]['_positions']
        atom_types = samples[number_of_atoms]['_atomic_numbers']

        for positions_single_molecule, atom_types_single_molecule in zip(positions, atom_types):
            mask = np.ones(positions.shape[1])

            one_hot = F.one_hot(
                        torch.from_numpy(atom_types_single_molecule),
                        num_classes=10).numpy()

            results['x'].append(torch.from_numpy(positions_single_molecule))
            results['one_hot'].append(torch.from_numpy(one_hot))
            results['node_mask'].append(torch.from_numpy(mask))

    return results


def main():
    with open('generated_samples/gschnet/gschnet_samples.pickle', 'rb') as f:
        samples = pickle.load(f)

    from configs import datasets_config

    dataset_info = {'atom_decoder': [None, 'H', None, None, None,
                                     None, 'C', 'N', 'O', 'F'],
                    'name': 'qm9'}

    results = flatten_sample_dictionary(samples)

    print(f'Analyzing {len(results["x"])} molecules...')

    validity_dict, rdkit_metrics = analyze_stability_for_molecules(results, dataset_info)
    print(validity_dict, rdkit_metrics[0])


if __name__ == '__main__':
    main()
