import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    data_path = "../data/adult.data"
    cols = ["age","workclass","fnlwgt","education","education-num",
            "marital-status","occupation","relationship","race","sex",
            "capital-gain","capital-loss","hours-per-week","native-country","income"]
    df = pd.read_csv(data_path, names=cols, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop("income", axis=1)
    y = df["income"]
    return train_test_split(X, y, test_size=0.2, random_state=42)