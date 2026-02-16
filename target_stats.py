import pandas as pd
import numpy as np
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter # type: ignore
    from lifelines.utils import median_survival_times # type: ignore
except ImportError:
    raise ImportError("The 'lifelines' library is required. Please install it using `pip install lifelines`.")

all_std_stats = []
for g in [0, 1]:
    all_std_stats.extend([
        f"target_HR_{g}", f"target_HR_{g}_lcl", f"target_HR_{g}_ucl"
    ])
    # Median keys
    for z in [0, 1]:
        base = f"target_{g}_z{z}_median"
        all_std_stats.extend([
            base, f"{base}_lcl", f"{base}_ucl"
        ])

def safe_cox_hr_ci_generic(df_subset):
    """
    Safely computes Cox Hazard Ratio (HR) and 95% Confidence Intervals (CI).
    Equivalent to R's safe_cox_hr_ci_generic.
    """
    # R: if (nrow(df_subset) == 0L || length(unique(df_subset$Z)) < 2L)
    if df_subset.empty or df_subset['Z'].nunique() < 2:
        return {'HR': np.nan, 'LCL': np.nan, 'UCL': np.nan}

    try:
        cph = CoxPHFitter()
        # R: coxph(Surv(time, event) ~ Z, ...)
        # lifelines requires a dataframe and column names
        cph.fit(df_subset, duration_col='time', event_col='event', formula='Z')
        
        # Extract HR
        # cph.hazard_ratios_ is a Series indexed by covariate name ('Z')
        hr = cph.hazard_ratios_['Z']
        
        # Extract CI
        # cph.confidence_intervals_ is a DataFrame containing the CIs for the coefficients (log-HR).
        # We access via iloc to avoid KeyErrors from varying column names (e.g. '95% lower-bound' vs 'lower 0.95')
        # The DataFrame is indexed by covariate name ('Z').
        ci_df = cph.confidence_intervals_
        
        # Assuming standard structure: [lower, upper]
        log_lcl = ci_df.loc['Z'].iloc[0]
        log_ucl = ci_df.loc['Z'].iloc[1]
        
        return {'HR': hr, 'LCL': np.exp(log_lcl), 'UCL': np.exp(log_ucl)}
        
    except Exception:
        # Catch convergence errors, singular matrix errors, or data issues
        return {'HR': np.nan, 'LCL': np.nan, 'UCL': np.nan}

def safe_km_median_ci_generic(df_subset):
    """
    Safely computes Kaplan-Meier Median survival and 95% Confidence Intervals.
    Equivalent to R's safe_km_median_ci_generic.
    """
    if df_subset.empty:
        return {'median': np.nan, 'LCL': np.nan, 'UCL': np.nan}

    try:
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df_subset['time'], event_observed=df_subset['event'])
        
        median_val = kmf.median_survival_time_
        
        # Calculate CI for the median
        # median_survival_times takes the confidence interval of the survival function
        median_ci_df = median_survival_times(kmf.confidence_interval_)
        
        # The df columns are usually suffix_lower_0.95, suffix_upper_0.95
        # We extract the scalar values
        if median_ci_df.shape[1] >= 2:
            lcl = median_ci_df.iloc[0, 0]
            ucl = median_ci_df.iloc[0, 1]
        else:
            lcl = np.nan
            ucl = np.nan

        return {'median': median_val, 'LCL': lcl, 'UCL': ucl}
        
    except Exception:
        return {'median': np.nan, 'LCL': np.nan, 'UCL': np.nan}


def get_est_stats(data, label, stats=all_std_stats):
    """
    Computes summary statistics for simulation data.
    
    Args:
        data (pd.DataFrame): Must have columns 'time', 'event', 'Z'.
        label (list, np.array, pd.DataFrame): 
            - If 1D: Integer vector (0/1) corresponding to data rows.
            - If 2D (Matrix): 
                - If shape (n, len(data)): Iterates over rows (n output rows).
                - If shape (len(data), n): Iterates over columns (n output rows).
        stats (list, optional): List of statistic names strings. 
                                If provided, only these stats are computed.
    
    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a label set.
    """
   
    labels_arr = np.array(label)
   
    N = len(data)
    
    # --- 1. Determine Label Sets (Handle 1D or 2D inputs) ---
    label_iterator = []
    
    if labels_arr.ndim == 1:
        if len(labels_arr) != N:
             raise ValueError(f"Label 1D length ({len(labels_arr)}) must match data length ({N}).")
        label_iterator = [labels_arr]
        
    elif labels_arr.ndim == 2:
        # User specified case: shape (n, len(data)) -> Iterate rows
        if labels_arr.shape[1] == N:
             label_iterator = (labels_arr[i, :] for i in range(labels_arr.shape[0]))
        # Alternative case: shape (len(data), n) -> Iterate columns
        elif labels_arr.shape[0] == N:
             label_iterator = (labels_arr[:, i] for i in range(labels_arr.shape[1]))
        else:
             raise ValueError(f"Label matrix shape {labels_arr.shape} is incompatible with data length {N}.")
    else:
        raise ValueError("Label input must be 1D or 2D.")
    
    # --- 2. Compute Stats for Each Label Set ---
    results_list = []
    
    # We use a copy of data to avoid modifying the original dataframe repeatedly if passed by ref
    # However, to be efficient, we just modify the column 'label' on the fly.
    df = data.copy()
    
    for current_label in label_iterator:
        df['label'] = np.array(current_label, dtype=int)
        
        # Initialize row dictionary
        out = {k: np.nan for k in stats}
        
        # Filter for valid labels 0 and 1 present in this iteration
        valid_labs = [0, 1]
        present_labs = sorted([x for x in df['label'].unique() if x in valid_labs])

        for g in present_labs:
            # Subset data for the label group
            ss = df[df['label'] == g]
            
            # Cox Hazard Ratio
            hr_keys = [f"target_HR_{g}", f"target_HR_{g}_lcl", f"target_HR_{g}_ucl"]
            if any(k in stats for k in hr_keys):
                hr_res = safe_cox_hr_ci_generic(ss)
                if f"target_HR_{g}" in stats:
                    out[f"target_HR_{g}"] = round(hr_res['HR'], 2)
                if f"target_HR_{g}_lcl" in stats:
                    out[f"target_HR_{g}_lcl"] = round(hr_res['LCL'], 2)
                if f"target_HR_{g}_ucl" in stats:
                    out[f"target_HR_{g}_ucl"] = round(hr_res['UCL'], 2)

            # KM Medians
            for z in [0, 1]:
                base = f"target_{g}_z{z}_median"
                med_keys = [base, f"{base}_lcl", f"{base}_ucl"]
                if any(k in stats for k in med_keys):
                    ssz = ss[ss['Z'] == z]
                    med_res = safe_km_median_ci_generic(ssz)
                    if base in stats:
                        out[base] = round(med_res['median'], 1)
                    if f"{base}_lcl" in stats:
                        out[f"{base}_lcl"] = round(med_res['LCL'], 1)
                    if f"{base}_ucl" in stats:
                        out[f"{base}_ucl"] = round(med_res['UCL'], 1)
        
        results_list.append(out)

    return pd.DataFrame(results_list)

# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy data
    np.random.seed(42)
    n = 100
    df_sim = pd.read_csv('df_full.csv')
    labels = labels = np.random.randint(0, 2, 400)

    # 1. Compute All Stats
    print("--- Computing All Stats ---")
    results_all = get_est_stats(df_sim, labels)
    for k, v in results_all.items():
        print(f"{k}: {v}")
        
    # 2. Compute Filtered Stats (Optimization check)
    print("\n--- Computing Only HR for Group 0 ---")
    needed = ["target_HR_0", "target_HR_0_lcl", "target_HR_0_ucl"]
    results_filtered = get_est_stats(df_sim, labels, stats=needed)
    print(results_filtered)