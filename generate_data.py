import pandas as pd
from sklearn.datasets import make_classification
import numpy as np

# Create a synthetic dataset
# n_samples: total number of transactions
# n_features: number of features for each transaction
# n_informative: number of features that are actually useful
# n_redundant: number of features that are linear combinations of informative features
# flip_y: proportion of samples whose class is assigned randomly (noise)
# weights: proportion of samples assigned to each class. Here, 99.5% are class 0 (legit) and 0.5% are class 1 (fraud).
X, y = make_classification(
    n_samples=50000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    weights=[0.995, 0.005],
    flip_y=0.01,
    random_state=42
)

# Create a DataFrame
feature_names = [f'V{i}' for i in range(1, 21)]
df = pd.DataFrame(X, columns=feature_names)

# Add a realistic 'Amount' column
df['Amount'] = np.round(np.random.lognormal(mean=3.0, sigma=1.0, size=50000) * 100, 2)
df['Class'] = y

# Save to CSV
df.to_csv('transactions.csv', index=False)

print("Synthetic dataset 'transactions.csv' created successfully.")
print(f"Dataset shape: {df.shape}")
print("Class distribution:")
print(df['Class'].value_counts(normalize=True))
