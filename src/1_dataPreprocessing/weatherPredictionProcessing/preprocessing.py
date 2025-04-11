# src/1_dataPreprocessing/weatherPredictionProcessing/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import importlib
import os
import sys
import gc  # Add garbage collector for memory management

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reduce_mem_usage(df, verbose=True):
    """
    Reduces memory usage of a DataFrame by downcasting numeric columns to smaller types.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logging.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def selective_one_hot_encoding(df, categorical_cols, max_features=100):
    """
    Performs one-hot encoding in a memory-efficient way by:
    1. Limiting cardinality to most frequent values
    2. Using int8 instead of int64 for the encoded columns
    3. Processing each categorical column separately
    """
    # Start with a DataFrame containing only the non-categorical columns
    result_df = df.drop(columns=categorical_cols).copy()
    
    # Process each categorical column separately
    for col in categorical_cols:
        logging.info(f"One-hot encoding column: {col}")
        
        # Get value counts for this column
        value_counts = df[col].value_counts()
        
        # If there are too many unique values, limit to the most frequent ones
        if len(value_counts) > max_features:
            logging.warning(f"Column {col} has {len(value_counts)} unique values, limiting to top {max_features}")
            top_values = value_counts.index[:max_features]
            
            # Create a mask for rows with top values
            mask = df[col].isin(top_values)
            
            # Create one-hot encoding for only the top values
            dummies = pd.get_dummies(df.loc[mask, col], prefix=col, prefix_sep='_', dummy_na=False)
            
            # Create a sparse matrix for "other" values
            other_dummies = pd.DataFrame({f"{col}_OTHER": (~mask).astype(np.int8)}, index=df.index)
            
            # Combine the two sets of dummies
            all_dummies = pd.DataFrame(index=df.index)
            for dummy_col in dummies.columns:
                all_dummies[dummy_col] = np.int8(0)  # Initialize with zeros
                all_dummies.loc[mask, dummy_col] = dummies[dummy_col].values
            
            # Merge with the "other" category
            all_dummies = pd.concat([all_dummies, other_dummies], axis=1)
        else:
            # Create one-hot encoding for all values
            all_dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_', dummy_na=False)
            
        # Ensure all dummy columns are int8 (memory efficient)
        for dummy_col in all_dummies.columns:
            all_dummies[dummy_col] = all_dummies[dummy_col].astype(np.int8)
        
        # Concatenate with the result DataFrame
        result_df = pd.concat([result_df, all_dummies], axis=1)
        
        # Force garbage collection to free memory
        gc.collect()
    
    return result_df

def preprocess_notebook_style(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """Applies feature engineering and notebook-style preprocessing."""
    logging.info("Starting notebook-style preprocessing with feature engineering...")
    target = config.TARGET_VARIABLE

    # --- 0. Memory Optimization: Reduce memory usage of initial DataFrame ---
    df = reduce_mem_usage(df)
    gc.collect()  # Force garbage collection

    # --- 1. Feature Engineering ---
    try:
        # Import the feature_engineering module directly from its implementation path
        # to avoid circular imports
        fe_path = os.path.join(os.path.dirname(__file__), "feature_engineering.py")
        spec = importlib.util.spec_from_file_location("feature_engineering", fe_path)
        feature_engineering = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_engineering)
        
        # Now use the feature_engineering module
        df_engineered = feature_engineering.engineer_features(df.copy())
        
        # Force garbage collection after feature engineering
        del df
        gc.collect()
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}", exc_info=True)
        return pd.DataFrame(), pd.Series(), []

    # --- 2. Memory Optimization: Reduce memory usage after feature engineering ---
    df_engineered = reduce_mem_usage(df_engineered)
    gc.collect()  # Force garbage collection

    # --- 3. Select Features and Target ---
    # Use the MODEL_FEATURES list *updated* by engineer_features
    # Also ensure base categoricals and target are present for subsequent steps
    features_to_use_in_model = [f for f in config.MODEL_FEATURES if f in df_engineered.columns]
    base_cats_needed = [f for f in config.CATEGORICAL_FEATURES_BASE if f in df_engineered.columns]
    cols_to_keep = list(set(features_to_use_in_model + [target] + base_cats_needed))

    missing_features = set(config.MODEL_FEATURES) - set(features_to_use_in_model)
    if missing_features:
        logging.warning(f"Engineered features configured but not found in DataFrame after engineering: {missing_features}")

    if target not in df_engineered.columns:
        logging.error(f"Target variable '{target}' not found after engineering.")
        return pd.DataFrame(), pd.Series(), []
    if not features_to_use_in_model:
         logging.error("No usable features found after engineering.")
         return pd.DataFrame(), pd.Series(), []

    # Select only necessary columns - create a new DataFrame to avoid warnings
    df_model = df_engineered[cols_to_keep].copy()
    
    # Force garbage collection after selection
    del df_engineered
    gc.collect()
    
    logging.debug(f"Shape after initial column selection: {df_model.shape}")

    # --- 4. Handle Target ---
    # Fix to avoid pandas warning - use loc instead of direct assignment
    df_model.loc[:, target] = pd.to_numeric(df_model[target], errors='coerce').fillna(0)
    df_model.loc[:, target] = df_model[target].clip(lower=0)

    # --- 5. Handle Missing Values (IMPROVED) ---
    # First, identify features with significant NaN values
    features_to_check_na = [f for f in features_to_use_in_model if f in df_model.columns]
    if not features_to_check_na:
        logging.error("No features available to check for NaNs.")
        return pd.DataFrame(), pd.Series(), []
    
    # Log the missing data percentage for each column
    na_percentage = df_model[features_to_check_na].isna().mean() * 100
    na_percentage_nonzero = na_percentage[na_percentage > 0].sort_values(ascending=False)
    
    if len(na_percentage_nonzero) > 0:
        logging.info("Missing data percentage by column:")
        for col, pct in na_percentage_nonzero.items():
            logging.info(f"  {col}: {pct:.2f}%")
    else:
        logging.info("No missing data found in selected features.")
    
    # Strategy 1: For columns with less than 15% missing values, use imputation
    # Strategy 2: For columns with more than 15% missing, consider dropping the column
    
    # Columns with moderate missing data - impute
    moderate_missing = [col for col in features_to_check_na if 0 < na_percentage[col] <= 15]
    for col in moderate_missing:
        # Numeric columns: impute with median
        if np.issubdtype(df_model[col].dtype, np.number):
            median_val = df_model[col].median()
            # Avoid inplace to prevent pandas warning
            df_model.loc[:, col] = df_model[col].fillna(median_val)
            logging.info(f"Imputed missing values in {col} with median: {median_val}")
        # Non-numeric columns: impute with mode
        else:
            mode_val = df_model[col].mode()[0] if len(df_model[col].mode()) > 0 else ""
            # Avoid inplace to prevent pandas warning
            df_model.loc[:, col] = df_model[col].fillna(mode_val)
            logging.info(f"Imputed missing values in {col} with mode: {mode_val}")
    
    # Columns with high missing data - consider dropping
    high_missing = [col for col in features_to_check_na if na_percentage[col] > 15]
    if high_missing:
        logging.warning(f"Columns with high missing data (>15%): {high_missing}")
        if config.DROP_HIGH_MISSING_COLUMNS:
            df_model = df_model.drop(columns=high_missing)
            logging.warning(f"Dropped {len(high_missing)} columns with high missing data")
            # Update feature list
            features_to_use_in_model = [f for f in features_to_use_in_model if f not in high_missing]
        else:
            logging.warning("High missing columns are being kept and will be imputed")
            for col in high_missing:
                if np.issubdtype(df_model[col].dtype, np.number):
                    median_val = df_model[col].median()
                    # Avoid inplace to prevent pandas warning
                    df_model.loc[:, col] = df_model[col].fillna(median_val)
                else:
                    mode_val = df_model[col].mode()[0] if len(df_model[col].mode()) > 0 else ""
                    # Avoid inplace to prevent pandas warning
                    df_model.loc[:, col] = df_model[col].fillna(mode_val)
    
    # After imputation, check if there are still any remaining nulls
    remaining_nulls = df_model[features_to_use_in_model].isna().sum().sum()
    if remaining_nulls > 0:
        logging.warning(f"After imputation, still found {remaining_nulls} null values. Dropping those rows.")
        df_model = df_model.dropna(subset=features_to_use_in_model)
        logging.warning(f"DataFrame shape after dropping nulls: {df_model.shape}")
    
    if df_model.empty:
        logging.warning("DataFrame is empty after handling NaNs.")
        return pd.DataFrame(), pd.Series(), []
    
    logging.debug(f"Shape after handling NaNs: {df_model.shape}")

    # --- 6. Handle columns that need to be dropped ---
    # Exclude datetime columns and other non-useful columns before OHE
    columns_to_drop = []
    
    # Drop datetime columns - they cause issues when attempting to convert to numeric
    datetime_cols = ['timestamp', 'timestamp_hour', 'timestamp_dest', 'FlightDate']
    for col in datetime_cols:
        if col in df_model.columns:
            columns_to_drop.append(col)
    
    # Drop other non-useful columns (non-categorical text fields)
    text_cols = ['weather_desc', 'city', 'city_dest', 'airport_code_icao', 'airport_code_icao_dest', 'CRSDepTime']
    for col in text_cols:
        if col in df_model.columns:
            columns_to_drop.append(col)
    
    if columns_to_drop:
        logging.info(f"Dropping problematic columns before one-hot encoding: {columns_to_drop}")
        df_model = df_model.drop(columns=columns_to_drop, errors='ignore')
        
        # Force garbage collection after dropping columns
        gc.collect()

    # --- 7. One-hot encode BASE categoricals (Memory Efficient Version) ---
    categorical_to_encode = [f for f in config.CATEGORICAL_FEATURES_BASE if f in df_model.columns]
    logging.info(f"Applying memory-efficient one-hot encoding to categoricals: {categorical_to_encode}")
    
    if categorical_to_encode: # Only run if there are categoricals to encode
        try:
            # Extract target before encoding (we'll add it back later)
            y = df_model[target].copy()
            
            # Limit airport/airline categories to most common ones to save memory
            # Dest has highest cardinality, limit to 50 most common
            dest_limit = min(50, config.MAX_CATEGORIES_DEST if hasattr(config, 'MAX_CATEGORIES_DEST') else 50)
            origin_limit = min(50, config.MAX_CATEGORIES_ORIGIN if hasattr(config, 'MAX_CATEGORIES_ORIGIN') else 50)
            airline_limit = min(20, config.MAX_CATEGORIES_AIRLINE if hasattr(config, 'MAX_CATEGORIES_AIRLINE') else 20)
            
            limits = {
                'Dest': dest_limit,
                'Origin': origin_limit,
                'Reporting_Airline': airline_limit
            }
            
            # Use memory-efficient one-hot encoding
            df_model_encoded = df_model.drop(columns=[target])  # Remove target temporarily
            
            # Perform selective one-hot encoding
            for cat_col in categorical_to_encode:
                limit = limits.get(cat_col, 20)  # Default to 20 categories if not specified
                logging.info(f"Encoding {cat_col} with max {limit} categories")
                
                # Get value counts for this column
                value_counts = df_model_encoded[cat_col].value_counts().head(limit)
                top_values = value_counts.index.tolist()
                
                # Create dummy variables only for top values
                dummies = pd.get_dummies(df_model_encoded[cat_col].apply(
                    lambda x: x if x in top_values else 'OTHER'
                ), prefix=cat_col, prefix_sep='_', dummy_na=False).astype(np.int8)
                
                # Drop the original column and join the dummy variables
                df_model_encoded = df_model_encoded.drop(columns=[cat_col])
                df_model_encoded = pd.concat([df_model_encoded, dummies], axis=1)
                
                # Force garbage collection
                gc.collect()
            
            # Add target back
            df_model_encoded[target] = y
            df_model = df_model_encoded
            
            # Force garbage collection
            del df_model_encoded, y
            gc.collect()
            
            # Log the encoding results
            cat_cols = [col for col in df_model.columns if any(col.startswith(f"{cat}_") for cat in categorical_to_encode)]
            logging.info(f"Created {len(cat_cols)} one-hot encoded features (as int8 type)")
            
        except Exception as e:
            logging.error(f"Error during one-hot encoding: {e}", exc_info=True)
            return pd.DataFrame(), pd.Series(), []
    else:
        logging.warning("No base categorical features found to encode.")

    # --- 8. FINAL SELECTION: Ensure only numeric features remain in X ---
    if target not in df_model.columns:
        logging.error("Target column lost during preprocessing.")
        return pd.DataFrame(), pd.Series(), []

    X_potential = df_model.drop(target, axis=1)
    y = df_model[target]
    
    # Force garbage collection
    del df_model
    gc.collect()

    # After concatenation, ensure all columns are numeric
    non_numeric_cols = []
    for col in X_potential.columns:
        if not pd.api.types.is_numeric_dtype(X_potential[col]):
            try:
                # Convert to numeric explicitly, should be integers for dummies
                X_potential.loc[:, col] = X_potential[col].astype(np.int8)
                logging.info(f"Converted column {col} to int8 type")
            except Exception as e:
                logging.warning(f"Could not convert column {col} to numeric: {e}")
                non_numeric_cols.append(col)
    
    # Select only columns with numeric dtypes
    numeric_cols = X_potential.select_dtypes(include=np.number).columns.tolist()
    
    if non_numeric_cols:
        logging.warning(f"Non-numeric columns found in feature set, will be DROPPED: {len(non_numeric_cols)} columns")

    # Keep only numeric columns for the final feature set X
    X = X_potential[numeric_cols]
    
    # Force garbage collection
    del X_potential
    gc.collect()
    
    final_feature_columns = numeric_cols # This is the definitive list of features used

    if X.empty or len(final_feature_columns) == 0:
        logging.error("Feature set X is empty after selecting only numeric columns.")
        return pd.DataFrame(), pd.Series(), []

    # --- 9. Final Memory Optimization ---
    X = reduce_mem_usage(X)
    gc.collect()

    # Log feature breakdown for analysis
    # Identify categorical features based on column name patterns
    categorical_features = [col for col in final_feature_columns if 
                           any(col.startswith(f"{base}_") for base in config.CATEGORICAL_FEATURES_BASE)]
    
    # Identify weather features
    weather_features = [col for col in final_feature_columns if 
                       any(base in col for base in config.WEATHER_FEATURES_BASE)]
    
    # Identify engineered features
    engineered_features = [col for col in final_feature_columns if 
                          any(col in feature_list for feature_list in [
                              config.WEATHER_DESC_FEATURES_ALL,
                              config.CYCLICAL_FEATURES_ALL,
                              config.THRESHOLD_FEATURES_ALL,
                              config.TREND_FEATURES_ALL,
                              config.INTERACTION_FEATURES_ALL
                          ])]
    
    # Log feature counts
    logging.info(f"Final feature breakdown: {len(final_feature_columns)} total features")
    logging.info(f"  - Categorical features (one-hot encoded): {len(categorical_features)}")
    logging.info(f"  - Weather features: {len(weather_features)}")
    logging.info(f"  - Engineered features: {len(engineered_features)}")

    logging.info(f"Preprocessing complete. Final Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y, final_feature_columns

