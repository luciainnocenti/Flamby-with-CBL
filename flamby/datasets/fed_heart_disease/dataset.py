import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from flamby.utils import check_dataset_from_config


class HeartDiseaseRaw(Dataset):
    """Pytorch dataset containing all the features, labels and
    metadata for the Heart Disease dataset.

    Parameters
    ----------
    X_dtype : torch.dtype, optional
        Dtype for inputs `X`. Defaults to `torch.float32`.
    y_dtype : torch.dtype, optional
        Dtype for labels `y`. Defaults to `torch.int64`.
    debug : bool, optional,
        Whether or not to use only the part of the dataset downloaded in
        debug mode. Defaults to False.
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.

    Attributes
    ----------
    data_dir: str
        Where data files are located
    labels : pd.DataFrame
        The labels as a dataframe.
    features: pd.DataFrame
        The features as a dataframe.
    centers: list[int]
        The list with the center ids associated with the dataframes.
    sets: list[str]
        For each sample if it is from the train or the test.
    X_dtype: torch.dtype
        The dtype of the X features output
    y_dtype: torch.dtype
        The dtype of the y label output
    debug: bool
        Whether or not we use the dataset with only part of the features
    normalize: bool
        Whether or not to normalize the features. We use the corresponding
        training client to compute the mean and std per feature used to
        normalize.
        Defaults to True.
    """

    def __init__(
            self,
            X_dtype=torch.float32,
            y_dtype=torch.float32,
            debug=False,
            data_path=None,
            normalize=True,
    ):
        """See description above"""

        if data_path is None:
            dict = check_dataset_from_config("fed_heart_disease", debug)
            self.data_dir = Path(dict["dataset_path"])
        else:
            if not (os.path.exists(data_path)):
                raise ValueError(f"The string {data_path} is not a valid path.")
            self.data_dir = Path(data_path)

        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug
        self.scaler = None

        self.centers_number = {
            "cleveland": 0,
            "hungarian": 1,
            "switzerland": 2,
            "va": 3,
        }

        self.features = pd.DataFrame()
        self.labels = pd.DataFrame()
        self.centers = []
        self.sets = []

        self.train_fraction = 0.75

        for center_data_file in self.data_dir.glob("*.data"):

            center_name = os.path.basename(center_data_file).split(".")[1]

            df = pd.read_csv(center_data_file, header=None)
            df = df.replace("?", np.NaN).drop([10, 11, 12], axis=1).dropna(axis=0)
            df = df.apply(pd.to_numeric)

            center_X = df.iloc[:, :-1]
            center_y = df.iloc[:, -1]

            self.features = pd.concat((self.features, center_X), ignore_index=True)
            self.labels = pd.concat((self.labels, center_y), ignore_index=True)
            self.centers += [self.centers_number[center_name]] * center_X.shape[0]

            # proposed modification to introduce shuffling before splitting the center
            nb = int(center_X.shape[0])
            indices_train, indices_test = train_test_split(
                np.arange(nb),
                test_size=1.0 - self.train_fraction,
                train_size=self.train_fraction,
                random_state=0,
                shuffle=True
            )

            for i in np.arange(nb):
                if i in indices_test:
                    self.sets.append("test")
                else:
                    self.sets.append("train")

        # encode dummy variables for categorical variables
        self.features = pd.get_dummies(self.features, columns=[1, 2, 5, 6, 8])
        self.features = [
            torch.from_numpy(self.features.loc[i].values.astype(np.float32)).to(
                self.X_dtype
            )
            for i in range(len(self.features))
        ]

        # keep 0 (no disease) and put 1 for all other values (disease)
        self.labels.where(self.labels == 0, 1, inplace=True)
        self.labels = torch.from_numpy(self.labels.values).to(self.X_dtype)

        # Per-center Normalization much needed
        self.centers_stats = {}
        for center in [0, 1, 2, 3]:
            # We normalize on train only
            to_select = [
                (self.sets[idx] == "train") and (self.centers[idx] == center)
                for idx, _ in enumerate(self.features)
            ]
            features_center = [
                fp for idx, fp in enumerate(self.features) if to_select[idx]
            ]
            features_tensor_center = torch.cat(
                [features_center[i][None, :] for i in range(len(features_center))],
                axis=0,
            )
            self.centers_stats[center] = MinMaxScaler()
            self.centers_stats[center].fit(features_tensor_center)

        # We normalize on train only for pooled as well
        to_select = [(self.sets[idx] == "train") for idx, _ in enumerate(self.features)]
        features_train = [fp for idx, fp in enumerate(self.features) if to_select[idx]]
        features_tensor_train = torch.cat(
            [features_train[i][None, :] for i in range(len(features_train))],
            axis=0,
        )
        self.centers_stats_pool = MinMaxScaler()
        self.centers_stats_pool.fit(features_tensor_train)

        # We convert everything back into lists
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        assert idx < len(self.features), "Index out of range."
        y = self.labels[idx]
        X = self.features[idx].reshape((-1, 18))
        if self.normalize:
            X = self.scaler.transform(X)
        X = np.squeeze(X, axis=0)
        return X, y


class FedHeartDisease(HeartDiseaseRaw):
    def __init__(
            self,
            center: int = 0,
            train: bool = True,
            pooled: bool = False,
            X_dtype: torch.dtype = torch.float32,
            y_dtype: torch.dtype = torch.float32,
            debug: bool = False,
            data_path: str = None,
            normalize: bool = True,
    ):
        """Cf class description"""

        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            debug=debug,
            data_path=data_path,
            normalize=normalize,
        )
        assert center in [0, 1, 2, 3]

        self.chosen_centers = [center]
        if pooled:
            self.chosen_centers = [0, 1, 2, 3]
            # We set the apropriate statistics
            self.scaler = self.centers_stats_pool
        else:
            self.scaler = self.centers_stats[center]

        if train:
            self.chosen_sets = ["train"]
        else:
            self.chosen_sets = ["test"]

        to_select = [
            (self.sets[idx] in self.chosen_sets)
            and (self.centers[idx] in self.chosen_centers)
            for idx, _ in enumerate(self.features)
        ]

        self.features = [fp for idx, fp in enumerate(self.features) if to_select[idx]]
        self.sets = [fp for idx, fp in enumerate(self.sets) if to_select[idx]]
        self.labels = [fp for idx, fp in enumerate(self.labels) if to_select[idx]]
        self.centers = [fp for idx, fp in enumerate(self.centers) if to_select[idx]]
