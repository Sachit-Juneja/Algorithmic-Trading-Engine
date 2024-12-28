from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

def train_random_forest(X_train, y_train, X_test, y_test, class_weight_dict):
    # Random Forest Model with GridSearchCV for hyperparameter tuning
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train, sample_weight=y_train.map(class_weight_dict))

    # Best Model
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def train_xgboost(X_train, y_train, X_test, y_test, class_weight_dict):
    # XGBoost Model
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=y_train.map(class_weight_dict))
    dtest = xgb.DMatrix(X_test)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': class_weight_dict[1] / class_weight_dict[0]
    }

    xg_model = xgb.train(params, dtrain, num_boost_round=100)

    y_pred_xgb = xg_model.predict(dtest)
    y_pred_xgb = [1 if i > 0.5 else 0 for i in y_pred_xgb]
    
    print(f"Accuracy (XGBoost): {accuracy_score(y_test, y_pred_xgb):.4f}")
    print("Classification Report (XGBoost):")
    print(classification_report(y_test, y_pred_xgb))
