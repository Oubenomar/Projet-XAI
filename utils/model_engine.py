from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path to find utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.talent_wrapper import TalentModel

TALENT_MODELS = [
    'ftt', 'resnet', 'switchtab', 'tabnet', 'tabr', 'tabpfn', 'mlp',
    'FT-Transformer', 'ResNet', 'SwitchTab', 'TabNet', 'TabR', 'TabPFN', 'MLP'
]

def train_model(X_train, y_train, X_val, y_val, model_type="Random Forest", task_type="binclass", data_dir="data/current_dataset"):
    """
    Trains a model based on the selected type.
    """
    model = None
    
    if model_type in TALENT_MODELS:
        # TALENT Model
        model = TalentModel(model_type, data_dir)
        model.fit()
        return model
        
    # Classical Models
    if task_type in ["binclass", "multiclass"]:
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42)
        elif model_type == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
            
    elif task_type == "regression":
         if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
         elif model_type == "Linear Regression":
            model = LinearRegression()
         elif model_type == "Decision Tree":
            model = DecisionTreeRegressor(max_depth=5, random_state=42)
         elif model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(random_state=42)
         elif model_type == "KNN":
            model = KNeighborsRegressor(n_neighbors=5)
            
    if model:
        # For classical models, we can concatenate train and val for better performance 
        # or just train on train. Let's train on train to be consistent with TALENT split.
        model.fit(X_train, y_train)
        
    return model

def predict(model, X):
    """Returns predictions (class labels or regression values)."""
    if isinstance(model, TalentModel):
        return model.predict(X)
    return model.predict(X)

def predict_proba(model, X):
    """Returns prediction probabilities (for classification)."""
    if isinstance(model, TalentModel):
        return model.predict_proba(X)
        
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None

def evaluate_model(model, X_test, y_test, task_type):
    """
    Evaluates the model and returns a dict of metrics.
    """
    metrics = {}
    
    y_pred = predict(model, X_test)
    
    if task_type == "regression":
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['R2'] = r2_score(y_test, y_pred)
    else:
        metrics['Accuracy'] = accuracy_score(y_test, y_pred)
        
        # F1 Score
        avg = 'binary' if task_type == 'binclass' else 'macro'
        metrics['F1 Score'] = f1_score(y_test, y_pred, average=avg)
        
        # Recall
        metrics['Recall'] = recall_score(y_test, y_pred, average=avg)
        
        # Confusion Matrix
        metrics['Confusion Matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # ROC AUC
        y_prob = predict_proba(model, X_test)
        if y_prob is not None:
            try:
                if task_type == 'binclass':
                    metrics['ROC AUC'] = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    metrics['ROC AUC'] = roc_auc_score(y_test, y_prob, multi_class='ovr')
            except Exception as e:
                metrics['ROC AUC'] = 0.0 # Failed (e.g. strict single class in test)
                
    return metrics

# --- Interpretation Helpers ---
def get_lr_coeffs(model, feature_names):
    """Returns DataFrame of Logistic/Linear Regression coefficients."""
    if not isinstance(model, (LogisticRegression, LinearRegression)):
        return None
    
    # Check for coef_ attribute
    if not hasattr(model, 'coef_'):
        return None

    # Linear Regression or Binary Class LR: coef_ is shape (n_features,) or (1, n_features)
    # Multiclass LR: (n_classes, n_features)
    
    coeffs = model.coef_
    
    if coeffs.ndim == 1:
        # Standard Linear or flattened Binary
        final_coeffs = coeffs
    elif coeffs.shape[0] == 1:
        # Binary Classification (1, n_features)
        final_coeffs = coeffs[0]
    else:
        # Multiclass: simplified to max absolute influence
        final_coeffs = np.max(np.abs(coeffs), axis=0)
        
    df_coeffs = pd.DataFrame({'Feature': feature_names, 'Coefficient': final_coeffs})
    df_coeffs['Abs_Coeff'] = df_coeffs['Coefficient'].abs()
    return df_coeffs.sort_values(by='Abs_Coeff', ascending=False)

def get_dt_text(model, feature_names):
    """Returns text representation of Decision Tree."""
    if not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        return None
    return export_text(model, feature_names=feature_names)

def get_knn_neighbors(model, instance, X_train, n_neighbors=5):
    """Returns indices and distances of nearest neighbors for a single instance."""
    if not isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
        return None, None
    
    # Instance should be 2D array (1, n_features)
    if isinstance(instance, pd.DataFrame):
        instance = instance.values
    if instance.ndim == 1:
        instance = instance.reshape(1, -1)
        
    distances, indices = model.kneighbors(instance, n_neighbors=n_neighbors)
    return distances[0], indices[0]
