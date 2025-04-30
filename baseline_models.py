from typing import List, Optional

import numpy as np
from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

_DESCRIPTOR_LIST = [x[0] for x in Descriptors._descList]


class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, radius: int = 3, count: bool = True, fp_size: int = 2048):
        """
        Transformer to convert rdkit Mol objects to Morgan fingerprints.

        :param radius: The radius of the Morgan fingerprint.
        :param count: Whether to use the count fingerprint.
        :param fp_size: The size of the fingerprint vector.
        """
        self.radius = radius
        self.count = count
        self.fp_size = fp_size

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[Chem.Mol]) -> np.ndarray:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.fp_size,
        )
        if self.count:
            return np.array([generator.GetCountFingerprintAsNumPy(mol) for mol in X])
        else:
            return np.array([generator.GetFingerprintAsNumPy(mol) for mol in X])


class RdkitDescriptorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_value: float = 0.0):
        """
        Transformer to convert rdkit Mol objects to RDKit descriptors.

        :param missing_value: The value to fill for missing values.
        """
        self.missing_value = missing_value

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[Chem.Mol]) -> np.ndarray:
        descriptors = []
        for mol in X:
            mol_desc = Descriptors.CalcMolDescriptors(
                mol, missingVal=self.missing_value
            )
            descriptors.append([mol_desc[desc] for desc in _DESCRIPTOR_LIST])
        return np.array(descriptors)


def get_fingerprint_baseline():
    return make_pipeline(
        MorganFingerprintTransformer(),
        GridSearchCV(
            SVR(),
            param_grid={"C": [0.1, 0.3, 1, 3, 10]},
        ),
    )


def get_descriptor_baseline():
    return make_pipeline(
        RdkitDescriptorTransformer(),
        RandomForestRegressor(
            n_estimators=500,
            random_state=1,
        ),
    )
