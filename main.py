import pandas as pd
from data_preprocessing import fetch_data, create_features
from model import train_random_forest, train_xgboost
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def main():
    # Fetch and preprocess data
    ticker = "AAPL"
    data = fetch_data(ticker)
    X, y = create_features(data)

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=pd.unique(y), y=y)
    class_weight_dict = dict(zip(pd.unique(y), class_weights))

    # Train and evaluate models
    print("Training Random Forest...")
    train_random_forest(X_train, y_train, X_test, y_test, class_weight_dict)

    print("\nTraining XGBoost...")
    train_xgboost(X_train, y_train, X_test, y_test, class_weight_dict)

if __name__ == "__main__":
    main()
