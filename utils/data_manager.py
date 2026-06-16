import pandas as pd
import numpy as np
import os
import json
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

def load_data(file_buffer):
    """Loads a CSV or Parquet file into a Pandas DataFrame."""
    try:
        if file_buffer.name.endswith('.parquet'):
            return pd.read_parquet(file_buffer)
        else:
            return pd.read_csv(file_buffer)
    except Exception as e:
        return None

def preprocess_data(df, target_column, 
                    impute_strategy="Mean", 
                    scaling_strategy="StandardScaler",
                    encoding_strategy="Label Encoding",
                    handle_imbalance=None,
                    drop_duplicates=True,
                    output_dir="data/current_dataset", 
                    random_state=42):
    """
    Advanced preprocessing for TALENT and Classical models.
    """
    # 1. Clean Data
    df = df.copy()
    if drop_duplicates:
        df = df.drop_duplicates()
        
    df = df.dropna(subset=[target_column])
    
    # 2. Identify Features
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Detect Task Type
    if y.dtype == 'object' or len(y.unique()) < 20: 
        if len(y.unique()) == 2:
            task_type = "binclass"
        else:
            task_type = "multiclass"
        
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y), name=target_column)
        target_mapping = {int(k): str(v) for k, v in enumerate(le_target.classes_)}
    else:
        task_type = "regression"
        y = y.astype(float)
        target_mapping = None

    # Detect Feature Types
    # Binary features are those with exactly 2 unique values (e.g. 0/1, Yes/No)
    binary_features = []
    high_card_numeric = []
    
    for col in X.columns:
        if X[col].nunique() == 2:
            binary_features.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            high_card_numeric.append(col)
            
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    original_features = X.columns.tolist() # Save state before encoding
    
    # 3. Handle transformers (Return for later use)
    transformers = {
        "imputer": None,
        "scaler": None,
        "le_target": None if task_type == "regression" else le_target,
        "encoders": {} # For LabelEncoding if needed for reuse
    }

    # Missing Value Handling
    if impute_strategy == "Mean":
        imputer = SimpleImputer(strategy='mean')
        X[numeric_features] = imputer.fit_transform(X[numeric_features])
        transformers["imputer"] = imputer
    elif impute_strategy == "Median":
        imputer = SimpleImputer(strategy='median')
        X[numeric_features] = imputer.fit_transform(X[numeric_features])
        transformers["imputer"] = imputer
    elif impute_strategy == "Constant":
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X[numeric_features] = imputer.fit_transform(X[numeric_features])
        transformers["imputer"] = imputer
    elif impute_strategy == "Iterative":
        imputer = IterativeImputer(random_state=random_state)
        X[numeric_features] = imputer.fit_transform(X[numeric_features])
        transformers["imputer"] = imputer
    
    for col in categorical_features:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Missing")

    # Target Leakage Detection ... (kept)
    leaks = {}
    suspect_keywords = ['id', 'name', 'boat', 'body', 'cabin', 'ticket', 'home.dest', 'address']
    
    for col in X.columns:
        reason = None
        if col in numeric_features:
            try:
                # We need numeric target for correlation
                corr = np.abs(np.corrcoef(X[col], y)[0, 1])
                if corr > 0.95: reason = f"High Correlation ({corr:.2f})"
            except:
                pass
        if X[col].nunique() <= 1:
            reason = "Constant Feature (No Variance)"
        if any(key in col.lower() for key in suspect_keywords):
            if reason: reason += " + Suspect Identifier"
            else: reason = "Suspect Identifier/Metadata"
        if reason:
            leaks[col] = reason
    
    # Categorical Encoding
    cat_mappings = {}
    if encoding_strategy == "One-Hot Encoding":
        X = pd.get_dummies(X, columns=categorical_features).reindex()
        # Note: One-Hot is harder to reuse without Saving specific column list
        numeric_features = X.columns.tolist()
        categorical_features = []
    else: # Label Encoding
        # Re-identify object columns specifically
        object_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        for col in object_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            cat_mappings[col] = {int(k): str(v) for k, v in enumerate(le.classes_)}
            transformers["encoders"][col] = le

    # Selective Standardization
    if scaling_strategy == "StandardScaler" and high_card_numeric:
        scaler = StandardScaler()
        X[high_card_numeric] = scaler.fit_transform(X[high_card_numeric])
        transformers["scaler"] = scaler
    elif scaling_strategy == "MinMaxScaler" and high_card_numeric:
        scaler = MinMaxScaler()
        X[high_card_numeric] = scaler.fit_transform(X[high_card_numeric])
        transformers["scaler"] = scaler
    elif scaling_strategy == "RobustScaler" and high_card_numeric:
        scaler = RobustScaler()
        X[high_card_numeric] = scaler.fit_transform(X[high_card_numeric])
        transformers["scaler"] = scaler

    # 4. Split Data
    stratify = y if task_type != "regression" else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=stratify
    )
    
    stratify_temp = y_temp if task_type != "regression" else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=stratify_temp
    )

    # Imbalance Handling (SMOTE) ... (kept)
    smote_stats = None
    if handle_imbalance and task_type != "regression":
        try:
            from imblearn.over_sampling import SMOTE, RandomOverSampler
            from imblearn.under_sampling import RandomUnderSampler
            
            dist_before = y_train.value_counts().to_dict()
            if handle_imbalance == "SMOTE": sampler = SMOTE(random_state=random_state)
            elif handle_imbalance == "Oversampling": sampler = RandomOverSampler(random_state=random_state)
            elif handle_imbalance == "Undersampling": sampler = RandomUnderSampler(random_state=random_state)
            
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            dist_after = y_train.value_counts().to_dict()
            smote_stats = {"before": dist_before, "after": dist_after}
        except:
            pass

    # Serialization etc...
    def get_parts(df_split):
        n_part = df_split[numeric_features].to_numpy().astype(np.float32) if numeric_features else np.array([]).reshape(len(df_split), 0)
        c_part = df_split[categorical_features].to_numpy().astype(np.int32) if categorical_features else np.array([]).reshape(len(df_split), 0)
        return n_part, c_part

    N_train, C_train = get_parts(X_train)
    N_val, C_val = get_parts(X_val)
    N_test, C_test = get_parts(X_test)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, 'N_train.npy'), N_train)
    np.save(os.path.join(output_dir, 'N_val.npy'), N_val)
    np.save(os.path.join(output_dir, 'N_test.npy'), N_test)
    
    np.save(os.path.join(output_dir, 'C_train.npy'), C_train)
    np.save(os.path.join(output_dir, 'C_val.npy'), C_val)
    np.save(os.path.join(output_dir, 'C_test.npy'), C_test)
    
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train.to_numpy())
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val.to_numpy())
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test.to_numpy())
    
    info = {
        "target_col": target_column, # Store original target col name
        "task_type": task_type,
        "n_num_features": len(numeric_features),
        "n_cat_features": len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "high_card_numeric": high_card_numeric,
        "binary_features": binary_features,
        "original_features": original_features, # NEW
        "feature_names": X.columns.tolist(),
        "cat_mappings": cat_mappings,
        "target_mapping": target_mapping,
        "encoding_strategy": encoding_strategy,
        "potential_leaks": leaks,
        "smote_stats": smote_stats
    }
    
    with open(os.path.join(output_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)
        
    return X_train, X_val, X_test, y_train, y_val, y_test, info, transformers

def process_custom_test_set(df, info, transformers):
    """
    Processes a custom test set to match training features and transformations.
    """
    df = df.copy()
    target_col = info['target_col']
    orig_features = info['original_features']
    final_features = info['feature_names']
    
    # 1. Handle Target
    y = None
    if target_col in df.columns:
        y_raw = df[target_col]
        if info['task_type'] != "regression":
            le = transformers["le_target"]
            y = y_raw.map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        else:
            y = y_raw.astype(float)
        df = df.drop(columns=[target_col])
    
    # 2. Align with Original Features (Before Encoding)
    df = df.reindex(columns=orig_features)
    
    # 3. Impute
    if transformers["imputer"]:
        # Use original numeric features identified during fit
        num_cols = [c for c in info['numeric_features'] if c in orig_features]
        # Safety: check if imputer expects the same number of columns
        if len(num_cols) > 0:
            df[num_cols] = transformers["imputer"].transform(df[num_cols])
        
    # 4. Encode
    if info['encoding_strategy'] == "One-Hot Encoding":
        # Apply the same get_dummies logic
        cat_cols = [c for c in orig_features if c not in info['numeric_features']]
        df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns])
        # Realign with the final feature list (this adds missing levels and drops unknown ones)
        df = df.reindex(columns=final_features, fill_value=0)
    else: # Label Encoding
        for col, le in transformers["encoders"].items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        # Final realignment
        df = df.reindex(columns=final_features, fill_value=0)
    
    # 5. Scale
    if transformers["scaler"]:
        high_card = info['high_card_numeric']
        # Ensure high_card columns exist in the current DF
        existing_high_card = [c for c in high_card if c in df.columns]
        if existing_high_card:
            df[existing_high_card] = transformers["scaler"].transform(df[existing_high_card])
        
    return df, y

def plot_smote_comparison(smote_stats):
    """
    Generates a Plotly figure comparing class distribution before and after SMOTE.
    """
    if not smote_stats:
        return None
    
    import plotly.graph_objects as go
    
    before = smote_stats['before']
    after = smote_stats['after']
    
    classes = sorted(before.keys())
    
    fig = go.Figure(data=[
        go.Bar(name='Before', x=[str(c) for c in classes], y=[before[c] for c in classes], marker_color='indianred'),
        go.Bar(name='After', x=[str(c) for c in classes], y=[after[c] for c in classes], marker_color='lightseagreen')
    ])
    
    fig.update_layout(
        title="Class Distribution: Before vs After Rebalancing",
        xaxis_title="Class",
        yaxis_title="Count",
        barmode='group'
    )
    
    return fig
