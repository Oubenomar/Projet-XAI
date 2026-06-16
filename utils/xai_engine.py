import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error

def compute_shap(model, X_train, X_instance, task_type="Classification", feature_names=None):
    """
    Computes SHAP values.
    Uses KernelExplainer as a generic fallback for TalentModels.
    Returns: shap_values (Explanation object or array), expected_value
    """
    # Summary background data
    background = shap.sample(X_train, 100) if len(X_train) > 100 else X_train
    
    # Use predict_proba for classification, predict for regression
    pred_fn = model.predict_proba if task_type == "Classification" else model.predict
    
    # KernelExplainer is robust for black-box models
    explainer = shap.KernelExplainer(pred_fn, background)
    
    # X_instance should be a DataFrame for feature names consistency
    if feature_names is None:
        feature_names = X_train.columns.tolist()
        
    shap_values = explainer.shap_values(X_instance)
    expected_value = explainer.expected_value
    
    # Handle classification structure: shap_values is list of arrays [class0, class1] or 3D array
    if task_type == "Classification":
        if isinstance(shap_values, list):
            # We usually focus on the positive class (last index)
            # If binary, index 1 is positive.
            idx = 1 if len(shap_values) == 2 else 0
            return shap_values[idx], expected_value[idx]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # (n_samples, n_features, n_classes)
            idx = 1 if shap_values.shape[2] == 2 else 0
            return shap_values[:, :, idx], expected_value[idx]
            
    return shap_values, expected_value

def compute_lime(model, X_train, X_instance, feature_names, task_type="Classification", 
                 class_names=None, num_samples=5000, num_features=10, distance_metric='euclidean'):
    """
    Computes LIME explanation with tunable parameters.
    """
    # 1. Identify Categorical Features
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
    
    # 2. Encode Data for LIME (LIME requires numeric input)
    data_encoded = X_train.copy()
    instance_encoded = X_instance.copy()
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on train, handle unknown in instance/generated data carefully
        # We convert to string to be safe
        data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
        
        # Transform instance (handle new labels if any - sophisticated way needed normally, simplified here)
        # We use a helper to map safely
        def safe_transform(val):
            try:
                return le.transform([str(val)])[0]
            except:
                return -1 # Unknown
                
        if isinstance(instance_encoded, pd.DataFrame): 
             # instance is usually a DataFrame row
             instance_encoded[col] = instance_encoded[col].apply(safe_transform)
        else:
             # Series
             instance_encoded[col] = safe_transform(instance_encoded[col])
             
        encoders[col] = le
    
    # Convert to numpy for LIME
    X_train_np = data_encoded.to_numpy().astype(float)
    if isinstance(instance_encoded, pd.DataFrame):
        instance_np = instance_encoded.to_numpy()[0].astype(float)
    else:
        instance_np = instance_encoded.to_numpy().astype(float)
    
    # 3. Create Wrapped Prediction Function
    def wrapped_predict(X_numpy):
        # X_numpy is 2D array of encoded values
        # We need to decode back to original DataFrame with Strings
        
        df_decoded = pd.DataFrame(X_numpy, columns=feature_names)
        
        for col, le in encoders.items():
            # Decode. Round to nearest int because LIME might generate floats
            # Clip to valid range
            vals = df_decoded[col].values
            vals = np.round(vals).astype(int)
            vals = np.clip(vals, 0, len(le.classes_) - 1)
            
            df_decoded[col] = le.inverse_transform(vals)
            
        # Ensure numeric cols are float
        num_cols = [c for c in feature_names if c not in cat_cols]
        for c in num_cols:
            df_decoded[c] = df_decoded[c].astype(float)
            
        if task_type == "Classification":
            return model.predict_proba(df_decoded)
        else:
            return model.predict(df_decoded)
    
    # 4. Explain
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        categorical_features=cat_indices,
        class_names=class_names,
        mode="classification" if task_type == "Classification" else "regression",
        discretize_continuous=True
    )
    
    exp = explainer.explain_instance(
        data_row=instance_np,
        predict_fn=wrapped_predict,
        num_samples=num_samples,
        num_features=num_features,
        distance_metric=distance_metric
    )
    return exp

def calculate_consistency(shap_vals, lime_exp, feature_names):
    """
    Calculates consistency between SHAP and LIME.
    Returns: Consistency Score (-1 to 1), Rank DataFrame
    """
    # Handle SHAP
    if isinstance(shap_vals, np.ndarray):
         shap_vals = shap_vals.flatten()
    
    shap_dict = dict(zip(feature_names, shap_vals))
    
    # Process LIME
    lime_weights_map = {}
    try:
        available_keys = list(lime_exp.local_exp.keys())
        target_key = available_keys[-1] # Usually 1 in binary [0, 1]
        
        lime_list = lime_exp.local_exp[target_key]
        for idx, weight in lime_list:
            fname = feature_names[idx]
            lime_weights_map[fname] = weight
            
    except Exception as e:
        print(f"Error parsing LIME: {e}")
        return 0.0, pd.DataFrame()

    # Create comparison dataframe
    data = []
    for f in feature_names:
        s_val = shap_dict.get(f, 0)
        l_val = lime_weights_map.get(f, 0)
        data.append({
            "Feature": f,
            "SHAP_Val": s_val,
            "LIME_Val": l_val,
            "SHAP_Abs": abs(s_val),
            "LIME_Abs": abs(l_val)
        })
        
    df_comp = pd.DataFrame(data)
    
    # Rank by Absolute Importance (Descending)
    df_comp["SHAP_Rank"] = df_comp["SHAP_Abs"].rank(ascending=False)
    df_comp["LIME_Rank"] = df_comp["LIME_Abs"].rank(ascending=False)
    
    # Calculate Correlation
    if len(df_comp) > 1:
        corr, _ = spearmanr(df_comp["SHAP_Rank"], df_comp["LIME_Rank"])
    else:
        corr = 1.0
        
    return corr, df_comp

def compute_permutation_importance(model, X_val, y_val, task_type='classification', scoring=None):
    """
    Computes global permutation feature importance.
    Returns: df_importance (Feature, Importance, Std)
    """
    if scoring is None:
        scoring = 'accuracy' if task_type == 'classification' else 'neg_root_mean_squared_error'
        
    try:
        r = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, scoring=scoring)
        
        df_imp = pd.DataFrame({
            'Feature': X_val.columns,
            'Importance': r.importances_mean,
            'Std': r.importances_std
        })
        return df_imp.sort_values(by='Importance', ascending=False)
    except Exception as e:
        print(f"Permutation Importance Error: {e}")
        return pd.DataFrame()
