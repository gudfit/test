import torch
import joblib
import json
import sys
import os
import traceback
import argparse
import warnings
import io
import contextlib

import numpy                 as np
import pandas                as pd
import torch.nn.functional   as F
from   datetime              import datetime, date, time
from   sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

warnings.filterwarnings("ignore")


# --- Context manager to suppress stdout and stderr ---
class SuppressOutput:
    def __enter__(self):
        # Save original stdout and stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        # Create string buffers to capture output
        self._stdout_buffer   = io.StringIO()
        self._stderr_buffer   = io.StringIO()
        # Redirect stdout and stderr to buffers
        sys.stdout            = self._stdout_buffer
        sys.stderr            = self._stderr_buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout and stderr
        sys.stdout            = self._original_stdout
        sys.stderr            = self._original_stderr
        # Discard captured output
        self._stdout_buffer.close()
        self._stderr_buffer.close()


# --- Configuration and Helper Import (with suppressed output) ---
with SuppressOutput():
    try:
        from .                import master_config as config
        script_dir = os.path.dirname(__file__)
        util_path  = os.path.join(script_dir, "utils")
        if str(util_path) not in sys.path:
            sys.path.insert(0, str(util_path))
        from datetime_helpers import (
            calculate_scheduled_datetimes,
            calculate_actual_datetimes,
        )
    except ImportError:
        import master_config as config

        script_dir = (
            os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
        )
        util_path = os.path.join(script_dir, "utils")
        if str(util_path) not in sys.path:
            sys.path.insert(0, str(util_path))
        from datetime_helpers import (
            calculate_scheduled_datetimes,
            calculate_actual_datetimes,
        )

DELAY_STATUS_MAP = {
    0: "On Time / Slight Delay (<= 15 min)",
    1: "Delayed (15-60 min)",
    2: "Significantly Delayed (60-120 min)",
    3: "Severely Delayed (120-240 min)",
    4: "Extremely Delayed (> 240 min)",
}
UNKNOWN_STATUS = "Unknown Delay Status"


class MasterPredictor:
    """Combines models, loading preferentially from local 'models' dir."""

    def __init__(self):
        with SuppressOutput():
            self.config                   = config
            self.device                   = config.DEVICE
            self.chain_length             = config.CHAIN_LENGTH
            self.min_turnaround_sec       = config.CHAIN_MIN_TURNAROUND_MINS * 60
            self.max_ground_sec           = config.CHAIN_MAX_GROUND_TIME.total_seconds()

            self.chain_classifier_model   = None
            self.chain_data_stats         = None
            self.chain_scaler             = None
            self.chain_encoder            = None
            self.chain_scaler_features    = []
            self.chain_encoder_features   = []
            self.chain_final_feature_cols = []
            self.regressor_model          = None

            # Determine paths *before* loading
            self.chain_model_path_to_load = self._get_model_path(
                config.LOCAL_MODELS_DIR,
                config.CHAIN_MODEL_FILENAME,
                config.ORIGINAL_CHAIN_MODEL_PATH,
            )
            self.regressor_model_path_to_load = self._get_model_path(
                config.LOCAL_MODELS_DIR,
                config.REGRESSOR_MODEL_FILENAME,
                config.ORIGINAL_REGRESSOR_MODEL_PATH,
            )

            self._load_chain_classifier_stats()
            self._load_chain_classifier_model()
            self._load_regressor_model()

    def _get_model_path(self, local_dir, filename, fallback_path):
        """Checks local dir first, then fallback path."""
        local_path = local_dir / filename
        if local_path.exists():
            return local_path
        elif fallback_path.exists():
            return fallback_path
        else:
            raise FileNotFoundError(
                f"Model '{filename}' not found in local path ({local_path}) or fallback path ({fallback_path})."
            )

    def _load_chain_classifier_stats(self):
        """Loads the data statistics (scalers, encoders) for the chain classifier."""
        stats_path = self.config.CHAIN_DATA_STATS_FILE
        if not stats_path.exists():
            raise FileNotFoundError(f"Chain classifier data stats not found: {stats_path}")
        try:
            with open(stats_path, "r") as f:
                self.chain_data_stats            = json.load(f)
            scaler_params = self.chain_data_stats.get("scaler_params")
            if scaler_params and "min" in scaler_params and "scale" in scaler_params:
                self.chain_scaler                = MinMaxScaler()
                self.chain_scaler.min_           = np.array(scaler_params["min"])
                self.chain_scaler.scale_         = np.array(scaler_params["scale"])
                self.chain_scaler_features       = scaler_params.get("feature_names", [])
                self.chain_scaler.n_features_in_ = len(self.chain_scaler_features)
            else:
                self.chain_scaler                = None

            encoder_cats                         = self.chain_data_stats.get("encoder_categories")
            if encoder_cats:
                try:
                    categories_list              = [
                        np.array(cats, dtype=object) for cats in encoder_cats.values()
                    ]
                except Exception:
                    categories_list              = None
                if categories_list:
                    self.chain_encoder_features  = list(encoder_cats.keys())
                    self.chain_encoder           = OrdinalEncoder(
                        categories     = categories_list,
                        handle_unknown = "use_encoded_value",
                        unknown_value  = -1,
                    )
                    dummy_data_dict              = {
                        feat: [cats[0]]
                        for feat, cats in zip(self.chain_encoder_features, self.chain_encoder.categories)
                    }
                    dummy_df_fit                 = pd.DataFrame(dummy_data_dict)
                    try:
                        self.chain_encoder.fit(dummy_df_fit[self.chain_encoder_features])
                    except Exception:
                        self.chain_encoder       = None
                else:
                    self.chain_encoder           = None
            else:
                self.chain_encoder               = None

            self.chain_final_feature_cols        = self.chain_data_stats.get(
                "feature_names", []
            )
        except Exception as e:
            raise

    def _load_chain_classifier_model(self):
        """Loads the PyTorch chain classifier model using the determined path."""
        # Avoid reloading        
        if self.chain_classifier_model:
            return
        if not self.chain_model_path_to_load:
            raise ValueError("Chain classifier model path not determined.")

        try:
            project_root_dir            = self.config.CHAIN_CLASSIFIER_DIR.parent
            if str(project_root_dir) not in sys.path:
                sys.path.insert(0, str(project_root_dir))
            hyperparams_file            = self.config.ORIGINAL_HYPERPARAMS_PATH
            best_params                 = None
            if hyperparams_file.exists():
                try:
                    with open(hyperparams_file, "r") as f:
                        best_params     = json.load(f)
                except Exception:
                    pass

            from flightChainClassifier.src.modeling.flight_chain_models import (
                SimAM_CNN_LSTM_Model,
            )

            classifier_config           = None
            try:
                from flightChainClassifier.src import (config as original_classifier_config,)
                classifier_config       = original_classifier_config
            except ImportError:
                pass

            num_features                = self.chain_data_stats.get("num_features")
            if num_features is None:
                raise ValueError("Num features missing from stats.")

            default_lstm_hidden         = (
                getattr(classifier_config, "DEFAULT_LSTM_HIDDEN_SIZE", 128)
                if classifier_config
                else 128
            )
            default_lstm_layers         = (
                getattr(classifier_config, "DEFAULT_LSTM_NUM_LAYERS", 1)
                if classifier_config
                else 1
            )
            default_dropout             = (
                getattr(classifier_config, "DEFAULT_DROPOUT_RATE", 0.2)
                if classifier_config
                else 0.2
            )
            default_bidir               = (
                getattr(classifier_config, "DEFAULT_LSTM_BIDIRECTIONAL", False)
                if classifier_config
                else False
            )
            lstm_hidden                 = (
                best_params.get("lstm_hidden_size", default_lstm_hidden)
                if best_params
                else default_lstm_hidden
            )
            lstm_layers                 = (
                best_params.get("lstm_num_layers", default_lstm_layers)
                if best_params
                else default_lstm_layers
            )
            dropout_rate                = (
                best_params.get("dropout_rate", default_dropout)
                if best_params
                else default_dropout
            )
            lstm_bidir                  = (
                best_params.get("lstm_bidirectional", default_bidir)
                if best_params
                else default_bidir
            )

            self.chain_classifier_model = SimAM_CNN_LSTM_Model(
                num_features            = num_features,
                num_classes             = self.config.CHAIN_TARGET_CLASSES,
                lstm_hidden             = lstm_hidden,
                lstm_layers             = lstm_layers,
                lstm_bidir              = lstm_bidir,
                dropout_rate            = dropout_rate,
            )
            self.chain_classifier_model.load_state_dict(
                torch.load(self.chain_model_path_to_load, map_location=self.device),
                strict                  = True,
            )
            self.chain_classifier_model.to(self.device)
            self.chain_classifier_model.eval()
        except ImportError as e:
            raise
        except RuntimeError as e:
            raise
        except Exception as e:
            raise

    def _load_regressor_model(self):
        """Loads the scikit-learn regressor model pipeline using the determined path."""
        if self.regressor_model:
            return  # Avoid reloading
        if not self.regressor_model_path_to_load:
            raise ValueError("Regressor model path not determined.")

        try:
            self.regressor_model = joblib.load(self.regressor_model_path_to_load)
            if not hasattr(self.regressor_model, "predict"):
                raise TypeError("Loaded object is not a scikit-learn model.")
        except Exception as e:
            raise

    def _preprocess_for_classifier(self, flight_chain_dicts):
        """Preprocesses a list of flight dictionaries for the chain classifier."""
        if (
            not self.chain_scaler
            or not self.chain_encoder
            or not self.chain_final_feature_cols
        ):
            return None
        if len(flight_chain_dicts)            != self.chain_length:
            return None

        all_features_list                      = []
        for flight_idx, flight_data in enumerate(flight_chain_dicts):
            try:
                sched_dep_dt, _                = calculate_scheduled_datetimes(flight_data)
                if pd.isna(sched_dep_dt):
                    return None

                single_flight_df               = pd.DataFrame([flight_data])
                single_flight_df["Month"]      = sched_dep_dt.month
                single_flight_df["DayOfMonth"] = sched_dep_dt.day
                single_flight_df["DayOfWeek"]  = sched_dep_dt.dayofweek
                single_flight_df["Hour"]       = sched_dep_dt.hour

                categorical_cols_present       = [
                    f
                    for f in self.chain_encoder_features
                    if f in single_flight_df.columns
                ]
                numerical_cols_for_scaling     = [
                    f
                    for f in self.chain_scaler_features
                    if f not in categorical_cols_present
                    and f in single_flight_df.columns
                ]

                df_cat                         = single_flight_df[categorical_cols_present].copy()
                df_num                         = single_flight_df[numerical_cols_for_scaling].copy()

                for col in df_cat.columns:
                    df_cat[col]                = df_cat[col].fillna("__MISSING__").astype(str)

                for col in df_num.columns:
                    df_num[col]                = pd.to_numeric(df_num[col], errors="coerce")
                    if df_num[col].isnull().any():
                        median_val             = 0.0  # Fallback
                        df_num[col]            = df_num[col].fillna(median_val)

                encoded_cats                   = self.chain_encoder.transform(df_cat)
                try:
                    encoded_cat_names          = self.chain_encoder.get_feature_names_out(
                        categorical_cols_present
                    )
                except AttributeError:
                    encoded_cat_names          = categorical_cols_present
                df_encoded_cats                = pd.DataFrame(encoded_cats, columns=encoded_cat_names, index=df_num.index)

                features_to_scale_df           = pd.concat([df_num, df_encoded_cats], axis=1)
                try:
                    features_to_scale_ordered  = features_to_scale_df.reindex(columns=self.chain_scaler_features, fill_value=0.0)
                except ValueError:
                    return None

                scaled_features                 = self.chain_scaler.transform(features_to_scale_ordered)
                df_scaled                       = pd.DataFrame(
                    scaled_features,
                    columns                     = self.chain_scaler_features,
                    index                       = df_num.index,
                )

                missing_final_features           = [
                    f
                    for f in self.chain_final_feature_cols
                    if f not in df_scaled.columns
                ]
                if missing_final_features:
                    return None

                final_flight_features            = (df_scaled[self.chain_final_feature_cols].iloc[0].values)
                all_features_list.append(final_flight_features)

            except Exception:
                return None

        if len(all_features_list)               != self.chain_length:
            return None

        try:
            chain_features_np                    = np.stack(all_features_list, axis=0).astype(np.float32)
            chain_tensor                         = torch.tensor(chain_features_np).unsqueeze(0)
            return chain_tensor.to(self.device)
        except Exception:
            return None

    def _is_valid_chain_transition(self, flight_prev, flight_curr):
        """Checks if the transition between two flights is valid for a chain."""
        act_dep_prev, act_arr_prev             = calculate_actual_datetimes(flight_prev)
        act_dep_curr, act_arr_curr             = calculate_actual_datetimes(flight_curr)

        if pd.isna(act_arr_prev) or pd.isna(act_dep_curr):
            sched_dep_prev, sched_arr_prev     = calculate_scheduled_datetimes(flight_prev)
            sched_dep_curr, _                  = calculate_scheduled_datetimes(flight_curr)
            if pd.isna(sched_arr_prev) or pd.isna(sched_dep_curr):
                return False
            ground_time_sec                    = (sched_dep_curr - sched_arr_prev).total_seconds()
        else:
            ground_time_sec                    = (act_dep_curr - act_arr_prev).total_seconds()

        if ground_time_sec                     < self.min_turnaround_sec:
            return False
        if ground_time_sec                     > self.max_ground_sec:
            return False

        sched_dep_prev_val, sched_arr_prev_val = calculate_scheduled_datetimes(
            flight_prev
        )
        sched_dep_curr_val, sched_arr_curr_val = calculate_scheduled_datetimes(
            flight_curr
        )
        if (
            pd.notna(sched_arr_prev_val)
            and pd.notna(sched_dep_prev_val)
            and sched_arr_prev_val <= sched_dep_prev_val
        ) or (
            pd.notna(sched_arr_curr_val)
            and pd.notna(sched_dep_curr_val)
            and sched_arr_curr_val <= sched_dep_curr_val
        ):
            return False

        return True

    def _get_category_from_minutes(self, minutes, thresholds):
        """Maps predicted delay minutes to a category index based on thresholds."""
        if pd.isna(minutes):
            return -1
        for i in range(len(thresholds) - 1):
            if thresholds[i] < minutes <= thresholds[i + 1]:
                return i
        return -1

    def predict(self, flight_context_dicts):
        """Predicts flight delay and returns a human-readable status along with a probability distribution for each category."""
        if not flight_context_dicts or not isinstance(flight_context_dicts, list):
            return {
                "prediction_type": None,
                "value"          : None,
                "status"         : "error",
                "message"        : "Input must be a non-empty list of flight dictionaries.",
            }

        target_flight_dict                   = flight_context_dicts[-1]
        can_try_classifier                   = len(flight_context_dicts) >= self.chain_length
        use_classifier                       = False
        chain_for_classifier                 = []
        prediction_details                   = ""
        prediction_type                      = "regressor"  

        if can_try_classifier:
            potential_chain                  = flight_context_dicts[-self.chain_length :]
            if all(f.get("Tail_Number") for f in potential_chain):
                tail_num                     = str(potential_chain[0].get("Tail_Number")).strip().upper()
                if tail_num and all(
                    str(f.get("Tail_Number")).strip().upper() == tail_num
                    for f in potential_chain
                ):
                    is_valid_chain           = True
                    for i in range(self.chain_length - 1):
                        if not self._is_valid_chain_transition(
                            potential_chain[i], potential_chain[i + 1]
                        ):
                            is_valid_chain   = False
                            break
                    if is_valid_chain:
                        use_classifier       = True
                        chain_for_classifier = potential_chain
                        prediction_type      = "classifier"

        if use_classifier:
            try:
                processed_chain_tensor       = self._preprocess_for_classifier(chain_for_classifier)
                if processed_chain_tensor is None:
                    prediction_type          = "regressor_fallback"
                else:
                    with torch.no_grad():
                        outputs              = self.chain_classifier_model(processed_chain_tensor)
                        # Compute probabilities using softmax
                        softmax_outputs      = F.softmax(outputs, dim=1)
                        probabilities_np     = softmax_outputs.cpu().numpy().flatten()
                        predicted_category   = int(np.argmax(probabilities_np))
                    status_string            = DELAY_STATUS_MAP.get(predicted_category, UNKNOWN_STATUS)
                    status_probabilities = {
                        DELAY_STATUS_MAP.get(i, UNKNOWN_STATUS): float(prob)
                        for i, prob in enumerate(probabilities_np)
                    }
                    prediction_details = f"Predicted category {predicted_category}."
                    return {
                        "prediction_type": prediction_type,
                        "value"          : status_string,
                        "probabilities"  : status_probabilities,
                        "status"         : "success",
                        "message"        : f"{prediction_details} Used chain classifier.",
                    }
            except Exception:
                prediction_type = "regressor_fallback"

        # --- Fallback or Default: Use Regressor ---
        if not use_classifier or prediction_type  == "regressor_fallback":
            final_prediction_type                  = (
                "regressor"
                if prediction_type                != "regressor_fallback"
                else prediction_type
            )
            try:
                ftd                                = 0.0
                pfd                                = 0.0
                if len(flight_context_dicts)      >= 2:
                    prev_flight_dict               = flight_context_dicts[-2]
                    sched_dep_prev, sched_arr_prev = calculate_scheduled_datetimes(
                        prev_flight_dict
                    )
                    sched_dep_target, _            = calculate_scheduled_datetimes(
                        target_flight_dict
                    )
                    if pd.notna(sched_arr_prev) and pd.notna(sched_dep_target):
                        ftd = max(
                            0.0,
                            (sched_dep_target - sched_arr_prev).total_seconds() / 60.0,
                        )
                    prev_arr_delay                 = prev_flight_dict.get("ArrDelayMinutes", 0.0)
                    pfd                            = float(prev_arr_delay) if pd.notna(prev_arr_delay) else 0.0

                sched_dep_target, _                = calculate_scheduled_datetimes(target_flight_dict)
                if pd.isna(sched_dep_target):
                    return {
                        "prediction_type" : final_prediction_type,
                        "value"           : None,
                        "status"          : "error",
                        "message"         : "Target flight schedule time is invalid.",
                    }

                target_features = {
                    "Origin"              : target_flight_dict.get("Origin"),
                    "Dest"                : target_flight_dict.get("Dest"),
                    "Reporting_Airline"   : target_flight_dict.get("Reporting_Airline"),
                    "Month"               : sched_dep_target.month,
                    "DayOfWeek"           : sched_dep_target.dayofweek,
                    "Hour"                : sched_dep_target.hour,
                    "FTD"                 : ftd,
                    "PFD"                 : pfd,
                }
                X_predict                          = pd.DataFrame([target_features])
                predicted_delay_minutes            = self.regressor_model.predict(X_predict)[0]
                predicted_delay_minutes            = max(0.0, float(predicted_delay_minutes))
                if not np.isfinite(predicted_delay_minutes):
                    predicted_delay_minutes = 0.0

                # Compute soft probabilities based on distance from category centers.
                thresholds                         = self.config.CHAIN_DELAY_THRESHOLDS
                category_centers                   = []
                # For categories defined by adjacent thresholds
                for i in range(len(thresholds) - 1):
                    center                         = (thresholds[i] + thresholds[i + 1]) / 2.0
                    category_centers.append(center)
                # For the final category, extrapolate a center
                last_center                        = thresholds[-1] + (thresholds[-1] - thresholds[-2]) / 2.0
                category_centers.append(last_center)

                # Use a gaussian kernel to compute a score for each category
                sigma                              = 15.0
                scores                             = [
                    np.exp(-((predicted_delay_minutes - center) ** 2) / (2 * sigma**2))
                    for center in category_centers
                ]
                score_sum                          = np.sum(scores)
                probabilities                      = [score / score_sum for score in scores]
                # Determine predicted category as the one with highest probability.
                predicted_category                 = int(np.argmax(probabilities))
                status_string                      = DELAY_STATUS_MAP.get(predicted_category, UNKNOWN_STATUS)
                prediction_details                 = (f"Predicted delay {predicted_delay_minutes:.2f} min.")
                soft_probabilities                 = {
                    DELAY_STATUS_MAP.get(i, UNKNOWN_STATUS): float(prob)
                    for i, prob in enumerate(probabilities)
                }
                return {
                    "prediction_type"     : final_prediction_type,
                    "value"               : status_string,
                    "probabilities"       : soft_probabilities,
                    "status"              : "success",
                    "message"             : f"{prediction_details} Used regressor model.",
                }
            except Exception as e:
                return {
                    "prediction_type"     : final_prediction_type,
                    "value"               : None,
                    "status"              : "error",
                    "message"             : f"Error during regressor prediction: {e}",
                }


# --- Main Execution Block ---
if __name__ == "__main__":
    # Temporarily suppress traceback printing
    sys.tracebacklimit = 0

    parser                                  = argparse.ArgumentParser(
        description                         = "Predict flight delay using a combined classifier/regressor model.",
        formatter_class                     = argparse.RawTextHelpFormatter,
    )
    input_group                             = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--flights-file",
        type                                = str,
        help                                = "Path to a JSON file containing flight context (list of dicts, ordered chronologically).",
    )
    input_group.add_argument(
        "--flights-cli",
        type                                = str,
        help                                = "JSON string containing flight context (list of dicts, ordered chronologically).",
    )
    args = parser.parse_args()

    with SuppressOutput():
        try:
            predictor                       = MasterPredictor()
            flight_context_data             = None
            if args.flights_file:
                if not os.path.exists(args.flights_file):
                    sys.exit(1)
                try:
                    with open(args.flights_file, "r") as f:
                        flight_context_data = json.load(f)
                except json.JSONDecodeError:
                    sys.exit(1)
            elif args.flights_cli:
                try:
                    flight_context_data     = json.loads(args.flights_cli)
                except json.JSONDecodeError:
                    sys.exit(1)
            if not isinstance(flight_context_data, list) or not flight_context_data:
                sys.exit(1)
            prediction_result               = predictor.predict(flight_context_data)
        except Exception:
            sys.exit(1)
    print(json.dumps(prediction_result, indent=4))
