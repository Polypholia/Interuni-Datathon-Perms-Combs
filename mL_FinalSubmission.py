import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

"""
Overview:
This script classifies gas types using sensor measurements.
The approach combines:

1. Feature engineering
2. Principal Component Analysis (PCA)
3. Batch drift correction
4. Multiple Logistic Regression models
5. Probability blending (ensemble)

The goal is to improve robustness against sensor drift and
distribution changes between batches while maintaining
high classification accuracy.
"""

# =========================================================
# CONFIG
# =========================================================

RAW_WEIGHT = 0.30
BATCH_STD_WEIGHT = 0.50
BATCH_CENTER_WEIGHT = 0.20

RAW_C = 0.10
BATCH_STD_C = 0.30
BATCH_CENTER_C = 0.20

PCA_COMPONENTS = 20

OUTPUT_FILE = (
    "/Users/sebastian/Downloads/submission_085882.csv"
)

# =========================================================
# LOAD DATA
# =========================================================

train1 = pd.read_csv(
    "/Users/sebastian/Downloads/train_version_1.csv"
)

train2 = pd.read_csv(
    "/Users/sebastian/Downloads/train_version_2.csv"
)

train = pd.concat(
    [train1, train2],
    ignore_index=True
)

test = pd.read_csv(
    "/Users/sebastian/Downloads/inter-uni-datathon-stream-3-gas-sensor-array-drift-dataset/test.csv"
)

# =========================================================
# FEATURE LIST
# =========================================================

sensor_cols = [
    c for c in train.columns
    if c.startswith("feat_")
]

base_cols = ["concentration"] + sensor_cols

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_row_features(df):

    """
    Create statistical summary features from sensor readings.

    Features include:
    - Mean
    - Standard deviation
    - Skewness
    - Kurtosis
    - Quartiles
    """

    X = df[base_cols].copy()

    values = X.values

    X["row_mean"] = np.mean(values, axis=1)
    X["row_std"] = np.std(values, axis=1)

    X["row_min"] = np.min(values, axis=1)
    X["row_max"] = np.max(values, axis=1)

    X["row_range"] = (
        X["row_max"] - X["row_min"]
    )

    X["row_median"] = np.median(
        values,
        axis=1
    )

    X["row_skew"] = skew(
        values,
        axis=1
    )

    X["row_kurt"] = kurtosis(
        values,
        axis=1
    )

    # concentration features

    X["log_concentration"] = np.log1p(
        X["concentration"]
    )

    X["sqrt_concentration"] = np.sqrt(
        X["concentration"]
    )

    X["conc_squared"] = (
        X["concentration"] ** 2
    )

    X["conc_cubed"] = (
        X["concentration"] ** 3
    )

    # robust spread measures

    X["row_q25"] = np.percentile(
        values,
        25,
        axis=1
    )

    X["row_q75"] = np.percentile(
        values,
        75,
        axis=1
    )

    X["row_iqr"] = (
        X["row_q75"]
        - X["row_q25"]
    )

    return X

# =========================================================
# GENERATE FEATURES
# =========================================================

train_feat = add_row_features(train)
test_feat = add_row_features(test)

# =========================================================
# PCA
# =========================================================

"""
Sensor measurements are highly correlated. PCA compresses the engineered feature space into
20 principal components while preserving most of the information. 

PCA_COMPONENTS = 20 was experimentally found to perform better than smaller or larger values. 

The resulting PCA features are added back to the feature set and used alongside engineered features.
"""

combined = pd.concat(
    [train_feat, test_feat],
    ignore_index=True
)

pca = PCA(
    n_components=PCA_COMPONENTS,
    random_state=42
)

pca.fit(combined)

train_pca = pca.transform(
    train_feat
)

test_pca = pca.transform(
    test_feat
)

for i in range(PCA_COMPONENTS):

    train_feat[f"pca_{i}"] = train_pca[:, i]
    test_feat[f"pca_{i}"] = test_pca[:, i]

feature_cols = train_feat.columns.tolist()

# =========================================================
# BATCH TRANSFORMATIONS
# =========================================================

def batch_standardize(df):

    """
    Standardize features separately within each batch.

    Formula:
        (value - batch_mean) / batch_std

    Purpose:
        Reduce sensor drift effects between batches.
    """

    X = df[feature_cols].copy()

    for batch in df["batch"].unique():

        mask = (
            df["batch"] == batch
        )

        subset = X.loc[mask]

        std = subset.std()
        std = std.replace(0, 1)

        X.loc[mask] = (
            subset - subset.mean()
        ) / std

    return X

def batch_center(df):

    X = df[feature_cols].copy()

    for batch in df["batch"].unique():

        mask = (
            df["batch"] == batch
        )

        subset = X.loc[mask]

        X.loc[mask] = (
            subset - subset.mean()
        )

    return X

# =========================================================
# MODEL FACTORY
# =========================================================

"""
Create a machine learning pipeline consisting of:

1. StandardScaler
2. Logistic Regression

StandardScaler ensures all features have a
comparable scale before training.

C controls regularization strength.
Smaller values provide stronger protection
against overfitting.
"""

def make_lr(c):

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=5000
        )
    )

# =========================================================
# DATA PREP
# =========================================================

y = train["gas_class"]

raw_train = train_feat
raw_test = test_feat

std_train = batch_standardize(
    train.assign(
        **train_feat.to_dict("series")
    )
)

std_test = batch_standardize(
    test.assign(
        **test_feat.to_dict("series")
    )
)

center_train = batch_center(
    train.assign(
        **train_feat.to_dict("series")
    )
)

center_test = batch_center(
    test.assign(
        **test_feat.to_dict("series")
    )
)

# =========================================================
# TRAIN MODELS
# =========================================================

"""
Train Three Independent Models

Model 1:
Raw engineered features.

Model 2:
Batch standardized features.

Model 3:
Batch centered features.

Each model sees a different representation of the same sensor data.

Using multiple representations increases model diversity and improves ensemble performance.
"""

raw_model = make_lr(RAW_C)
batch_std_model = make_lr(BATCH_STD_C)
batch_center_model = make_lr(BATCH_CENTER_C)

raw_model.fit(raw_train, y)
batch_std_model.fit(std_train, y)
batch_center_model.fit(center_train, y)

# =========================================================
# PROBABILITIES
# =========================================================

raw_proba = raw_model.predict_proba(raw_test)

std_proba = batch_std_model.predict_proba(
    std_test
)

center_proba = batch_center_model.predict_proba(
    center_test
)

blended = (
    RAW_WEIGHT * raw_proba
    + BATCH_STD_WEIGHT * std_proba
    + BATCH_CENTER_WEIGHT * center_proba
)

classes = raw_model.classes_

# =========================================================
# FINAL PREDICTIONS
# =========================================================

predictions = classes[
    np.argmax(
        blended,
        axis=1
    )
]

submission = pd.DataFrame({
    "measurement_id":
        test["measurement_id"],
    "gas_class":
        predictions
})

submission.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:", OUTPUT_FILE)

print(
    submission["gas_class"]
    .value_counts()
    .sort_index()
)
