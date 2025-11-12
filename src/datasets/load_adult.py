from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset
import torch

def load_adult(batch_size=256):
    df = fetch_openml("adult", version=2, as_frame=True).frame
    df["target"] = (df["class"] == ">50K").astype(int)
    df = df.drop(columns=["class"])

    X = df.drop(columns=["target"])
    y = df["target"]

    categorical = X.select_dtypes(include=["category", "object"]).columns
    numeric = X.select_dtypes(exclude=["category", "object"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    X_proc = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, X_train.shape[1]
