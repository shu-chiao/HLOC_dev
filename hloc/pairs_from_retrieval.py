import argparse
import collections.abc as collections
from pathlib import Path
from typing import Optional

import h5py
import pickle
import numpy as np
import torch

from . import logger
from .utils.io import list_h5_names
from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary


def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
        if len(names) == 0:
            raise ValueError(f"Could not find any image with the prefix `{prefix}`.")
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(
                f"Unknown type of image list: {names}."
                "Provide either a list or a path to a list file."
            )
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key="global_descriptor"):
    if name2idx is None:
        with h5py.File(str(path), "r", libver="latest") as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), "r", libver="latest") as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(
    scores: torch.Tensor,
    invalid: np.array,
    num_select: int,
    min_score: Optional[float] = None,
):
    """
    Arg:
        invalid: np.array(N, N), diagonal is True otherwise False
    Return:
        pairs: Tuple (query_idx, db_idx, score)
    """
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    invalid = torch.from_numpy(invalid).to(scores.device)
    
    if min_score is not None:
        # invalid |= scores < min_score
        invalid.logical_or_(scores < min_score)

    scores.masked_fill_(invalid, float("-inf"))

    topk = torch.topk(scores, num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    # valid = topk.values.isfinite().cpu().numpy()
    scores_topk = topk.values.cpu().numpy()         # new (add scores_topk)
    valid = np.isfinite(scores_topk)
    
    pairs = []
    for row, col in zip(*np.where(valid)):
        score = scores_topk[row, col]
        # (query_idx, db_idx, score)
        pairs.append((row, indices[row, col], score))
    return pairs


def main(
    descriptors,
    output,
    num_matched,
    query_prefix=None,
    query_list=None,
    db_prefix=None,
    db_list=None,
    db_model=None,
    db_descriptors=None,
    query_score:float=0.0,    # minimum score for a pair to query(1.0~0.0)
):
    logger.info("Extracting image pairs from a retrieval database.")

    # We handle multiple reference feature files.
    # We only assume that names are unique among them and map names to files.
    if db_descriptors is None:
        db_descriptors = descriptors
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    
    name2db = {n: i for i, p in enumerate(db_descriptors) for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    query_names_h5 = list_h5_names(descriptors)


    if db_model:
        images = read_images_binary(db_model / "images.bin")
        db_names = [i.name for i in images.values()]
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    if len(db_names) == 0:
        raise ValueError("Could not find any database image.")
    query_names = parse_names(query_prefix, query_list, query_names_h5)
    logger.info(f"Matching {len(query_names)} query images to {len(db_names)} database images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    query_desc = get_descriptors(query_names, descriptors)
    logger.info(f"Matching {query_desc.shape} query images to {db_desc.shape} database images.")

    sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))
    print("score: ", sim.shape)
    

    # Avoid self-matching
    self = np.array(query_names)[:, None] == np.array(db_names)[None]
    # print(">>", self.shape, self)
    pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=query_score)
    logger.info(f"Found {len(pairs)} pairs.")
    for i, j, s in pairs:
            print(f"query:{query_names[i]}, db:{db_names[j]}, {s}")

    if set(query_names) != set(db_names):
        logger.info(f"In query mode")
        pairs_np = np.array(pairs)
        if len(pairs_np) == 0:
            return None
        sort_np = pairs_np[pairs_np[:, -1].argsort()]
        # print("Best pair: ", sort_np[-1])
        best = (query_names[int(sort_np[-1][0])], db_names[int(sort_np[-1][1])], sort_np[-1][-1])
        return best
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_matched", type=int, required=True)
    parser.add_argument("--query_prefix", type=str, nargs="+")
    parser.add_argument("--query_list", type=Path)
    parser.add_argument("--db_prefix", type=str, nargs="+")
    parser.add_argument("--db_list", type=Path)
    parser.add_argument("--db_model", type=Path)
    parser.add_argument("--db_descriptors", type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
