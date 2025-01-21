import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, List

class EMNLP18(Dataset):
    """Custom dataset loader for EMNLP18 data."""

    def __init__(self, root: str, train: bool = True, classification: str = 'bias', transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.classification = classification
        self.transform = transform
        
        # File paths for the features to load
        self.feature_files = [
            "articles_body_glove.json"#,
            #"articles_title_glove.json"
        ]

        self.classes: List[str] = ["low", "mixed", "high"] #factuality
        if self.classification == 'bias':
            self.classes: List[str] = ["left", "center", "right"]
        
        # Load the data
        self._load_data()

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def _load_data(self):
        # Load corpus metadata
        corpus_path = os.path.join(self.root, "News-Media-Reliability-master/data/emnlp18/corpus.tsv")
        corpus = pd.read_csv(corpus_path, sep="\t")

        # Ensure columns exist
        if "fact" not in corpus.columns or "bias" not in corpus.columns:
            raise ValueError("Corpus file is missing required columns 'fact' and 'bias'.")

        # Map fact and bias labels to integers
        fact_mapping = {"low": 0, "mixed": 1, "high": 2}
        bias_mapping = {"extreme-left": 0, "left": 0, "left-center": 0, "center": 1, "center-right": 2, "right": 2, "extreme-right": 2}

        corpus["fact"] = corpus["fact"].map(fact_mapping)
        corpus["bias"] = corpus["bias"].map(bias_mapping)

        # Split into train and test sets
        splits_path = os.path.join(self.root, "News-Media-Reliability-master/data/emnlp18/splits.json")
        splits = pd.read_json(splits_path)
        split_ids = splits[0]["train" if self.train else "test"]
        for i in [1,2,3,4]: #make init input later
            split_ids.append(splits[i]["train" if self.train else "test"])

        corpus = corpus[corpus["source_url_normalized"].isin(split_ids)]

        # Load features
        feature_data = []
        feat_file_num = 0
        for feature_file in self.feature_files:
            feature_path = os.path.join(self.root, f"News-Media-Reliability-master/data/emnlp18/features/{feature_file}")
            with open(feature_path, "r") as f:
                cur_df = pd.read_json(f, orient="index")
                cur_df.columns = [str(feat_file_num)+"-"+str(col) for col in cur_df.columns] # to distinguish features from different files
                feature_data.append(cur_df)
            feat_file_num+=1
            
        
        
        # Merge features into a single DataFrame
        features = pd.concat(feature_data, axis=1)

        # Align features with the corpus
        aligned_data = corpus.join(features, on="source_url_normalized", how="inner") #pandas df
        # Drop rows with missing values
        aligned_data = aligned_data.dropna()

        # Extract data and labels
        self.data = torch.tensor(np.array(aligned_data.iloc[:, 5:].values), dtype=torch.float)
        self.targets = torch.tensor(np.array(aligned_data[[self.classification]].values), dtype=torch.long).flatten()
