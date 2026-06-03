# =========================================================
# DRUG TARGET INTERACTION PREDICTION
# WITHOUT RDKit AND WITHOUT TENSORFLOW
# =========================================================

# ---------------- IMPORT LIBRARIES ----------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# =========================================================
# STEP 1 : LOAD DATASET
# =========================================================

df = pd.read_csv("drugbank_clean.csv", low_memory=False)

print("===================================")
print("DATASET LOADED SUCCESSFULLY")
print("===================================")

print(df.head())

# =========================================================
# STEP 2 : EXTRACT DRUG TARGET PAIRS
# =========================================================

pairs = []

for index, row in df.iterrows():

    drug = row['drugbank-id']

    targets = row['targets']

    if pd.notna(targets):

        target_list = str(targets).split()

        for target in target_list:

            pairs.append([drug, target])

interaction_df = pd.DataFrame(
    pairs,
    columns=['Drug', 'Target']
)

print("\n===================================")
print("DRUG TARGET INTERACTIONS")
print("===================================")

print(interaction_df.head())

print("\nTotal Interactions :", len(interaction_df))

# =========================================================
# STEP 3 : GENERATE FEATURES
# =========================================================

def generate_features(drug_id, target_id):

    # Convert IDs into reproducible numeric features

    drug_hash = abs(hash(drug_id)) % 1000
    target_hash = abs(hash(target_id)) % 1000

    np.random.seed(drug_hash + target_hash)

    features = np.random.rand(128)

    return features

# =========================================================
# STEP 4 : CREATE FEATURE DATASET
# =========================================================

X = []
y = []

for index, row in interaction_df.iterrows():

    drug = row['Drug']
    target = row['Target']

    features = generate_features(drug, target)

    X.append(features)

    # Positive interaction
    y.append(1)

X = np.array(X)
y = np.array(y)

print("\nFeature Matrix Shape :", X.shape)

# =========================================================
# STEP 5 : CREATE NEGATIVE SAMPLES
# =========================================================

negative_X = []
negative_y = []

for i in range(len(X)):

    random_features = np.random.rand(128)

    negative_X.append(random_features)

    negative_y.append(0)

negative_X = np.array(negative_X)
negative_y = np.array(negative_y)

# Combine datasets

X_final = np.vstack((X, negative_X))
y_final = np.concatenate((y, negative_y))

print("\nFinal Dataset Shape :", X_final.shape)

# =========================================================
# STEP 6 : SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.2,
    random_state=42
)

# =========================================================
# STEP 7 : TRAIN RANDOM FOREST MODEL
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\n===================================")
print("TRAINING RANDOM FOREST MODEL")
print("===================================")

rf_model.fit(X_train, y_train)

# =========================================================
# STEP 8 : MODEL PREDICTIONS
# =========================================================

predictions = rf_model.predict(X_test)

# =========================================================
# STEP 9 : EVALUATE MODEL
# =========================================================

print("\n===================================")
print("MODEL EVALUATION")
print("===================================")

accuracy = accuracy_score(y_test, predictions)

print("Accuracy :", accuracy)

print("\nClassification Report")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

# =========================================================
# STEP 10 : PREDICT UNKNOWN INTERACTION
# =========================================================

sample = X_test[0].reshape(1, -1)

prediction = rf_model.predict(sample)

probability = rf_model.predict_proba(sample)

print("\n===================================")
print("UNKNOWN INTERACTION PREDICTION")
print("===================================")

print("Prediction :", prediction[0])

print("Interaction Probability :", probability[0][1])

if prediction[0] == 1:

    print("Interaction Likely")

else:

    print("Interaction Unlikely")

# =========================================================
# END OF PROJECT
# =========================================================
