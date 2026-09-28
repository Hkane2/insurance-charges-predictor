"""
Builds the insurance charges model (Linear Regression with a smoker x obese feature).

The Streamlit app imports build_model() and trains on insurance.csv when it starts.
You can also run this file directly to see the scores:  python train_model.py
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ['age', 'is_female', 'bmi', 'children', 'is_smoker',
            'region_southeast', 'is_obese', 'smoker_obese']


def make_features(raw):
    """Turn raw columns (age, sex, bmi, children, smoker, region) into model features."""
    data = pd.DataFrame()
    data['age'] = raw['age']
    data['is_female'] = (raw['sex'] == 'female').astype(int)
    data['bmi'] = raw['bmi']
    data['children'] = raw['children']
    data['is_smoker'] = (raw['smoker'] == 'yes').astype(int)
    data['region_southeast'] = (raw['region'] == 'southeast').astype(int)
    data['is_obese'] = (raw['bmi'] >= 30).astype(int)
    data['smoker_obese'] = data['is_smoker'] * data['is_obese']
    return data[FEATURES]


def build_model(csv_path='insurance.csv'):
    """Train and return (model, test_r2, test_mae)."""
    df = pd.read_csv(csv_path).drop_duplicates()
    X = make_features(df)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, r2_score(y_test, pred), mean_absolute_error(y_test, pred)


if __name__ == '__main__':
    model, r2, mae = build_model()
    print(f"Test R2:  {r2:.3f}")
    print(f"Test MAE: ${mae:,.0f}")
