""" Features created using the mlfinpy library. """
import pandas as pd
from mlfinpy.filters import cusum_filter, z_score_filter
from mlfinpy.structural_breaks import get_sadf


def add_mlfinpy_filter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features based on mlfinpy filters.
     
    Args:
        df: DataFrame with at least 'close' prices.
     
    Returns:
        DataFrame with new filter-based features.
    """
    out = df.copy()
    close = out['close']
    
    # 1. CUSUM Filter
    daily_std = close.pct_change().rolling(288).std().mean()
    threshold = daily_std * 2 if pd.notna(daily_std) and daily_std > 0 else 0.001
    event_timestamps_cusum = cusum_filter(close, threshold=threshold)
    cusum_series = pd.Series(1, index=event_timestamps_cusum)
    out['mlfinpy_cusum_event'] = cusum_series.reindex(out.index).fillna(0).astype(int)
    
    # 2. Z-Score Filter
    event_timestamps_zscore = z_score_filter(close, mean_window=288, std_window=288, z_score=3.0)
    z_score_series = pd.Series(1, index=event_timestamps_zscore)
    out['mlfinpy_z_score_event'] = z_score_series.reindex(out.index).fillna(0).astype(int)
    
    return out


def add_mlfinpy_structural_break_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features based on mlfinpy structural break tests.
     
    Args:
        df: DataFrame with at least 'close' prices.
     
    Returns:
        DataFrame with new structural break features.
    """
    out = df.copy()
    close = out['close']
    
    # Initialize placeholder columns
    out['sadf_log_tstat'] = 0.0
    out['sadf_log_crit'] = 0.0
    out['sadf_log_break'] = 0.0
    
    try:
        # To manage performance, run SADF on a sampled subset of the data
        # We run it once every 4 hours (48 * 5 mins)
        sampling_rate = 48
        close_sampled = close.iloc[::sampling_rate].dropna()
        
        print(f"[Info] Running SADF on {len(close_sampled)} sampled points...")
        
        # Kiểm tra dữ liệu đủ không
        if len(close_sampled) < 100:
            print(f"[Warning] Not enough data points ({len(close_sampled)}). Skipping SADF.")
            return out
            
        # Điều chỉnh tham số để phù hợp với dữ liệu
        min_length = min(30, len(close_sampled) // 4)  # Giảm min_length
        lags = min(3, len(close_sampled) // 30)  # Giảm lags xuống còn 3
        
        # get_sadf trả về pandas Series, không phải dictionary
        sadf_stats = get_sadf(
            series=close_sampled, 
            model='linear', 
            lags=lags, 
            min_length=min_length, 
            add_const=True, 
            phi=0,  # Thêm phi parameter
            num_threads=1, 
            verbose=False
        )
        
        print(f"[Debug] SADF result type: {type(sadf_stats)}")
        print(f"[Debug] SADF result shape: {sadf_stats.shape if hasattr(sadf_stats, 'shape') else 'No shape'}")
        
        # sadf_stats là một pandas Series chứa test statistics
        # Cần tính critical values riêng (thường là constant values)
        
        # Critical values cho ADF test (95% confidence level)
        # Đây là giá trị gần đúng, trong thực tế cần tính chính xác hơn
        critical_value_95 = -2.86  # Giá trị tiêu chuẩn cho ADF test
        
        # Tạo series critical values với cùng index
        sadf_crit_series = pd.Series(critical_value_95, index=sadf_stats.index)
        
        # Reindex về full dataset và forward fill
        sadf_tstat_full = sadf_stats.reindex(out.index).ffill()
        sadf_crit_full = sadf_crit_series.reindex(out.index).ffill()
        
        # Populate columns
        out['sadf_log_tstat'] = sadf_tstat_full.fillna(0.0)
        out['sadf_log_crit'] = sadf_crit_full.fillna(critical_value_95)
        out['sadf_log_break'] = (sadf_tstat_full > sadf_crit_full).fillna(False).astype(int)
        
        print("[Info] SADF calculation successful.")
        print(f"[Info] SADF stats range: [{sadf_stats.min():.4f}, {sadf_stats.max():.4f}]")
        
    except Exception as e:
        # If the test fails, log a warning but proceed with placeholder values
        print(f"[Warning] SADF calculation failed: {e}. Using neutral placeholders.")
        import traceback
        print(f"[Debug] Full traceback:")
        traceback.print_exc()
    
    return out


def add_mlfinpy_structural_break_features_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced version with proper critical value calculation.
     
    Args:
        df: DataFrame with at least 'close' prices.
     
    Returns:
        DataFrame with new structural break features.
    """
    out = df.copy()
    close = out['close']
    
    # Initialize placeholder columns
    out['sadf_log_tstat'] = 0.0
    out['sadf_log_crit'] = 0.0
    out['sadf_log_break'] = 0.0
    
    try:
        # Sample data for performance
        sampling_rate = 48
        close_sampled = close.iloc[::sampling_rate].dropna()
        
        print(f"[Info] Running SADF on {len(close_sampled)} sampled points...")
        
        if len(close_sampled) < 100:
            print(f"[Warning] Not enough data points ({len(close_sampled)}). Skipping SADF.")
            return out
        
        # Convert to log prices for better stationarity properties
        log_prices = np.log(close_sampled)
        
        min_length = max(20, len(log_prices) // 10)
        lags = min(2, len(log_prices) // 50)
        
        # Run SADF on log prices
        sadf_stats = get_sadf(
            series=log_prices, 
            model='linear', 
            lags=lags, 
            min_length=min_length, 
            add_const=True, 
            phi=0.0,
            num_threads=1, 
            verbose=False
        )
        
        # Calculate dynamic critical values based on sample size
        # This is a simplified approach - in practice you might want to use Monte Carlo
        n = len(log_prices)
        if n <= 50:
            crit_val = -3.75
        elif n <= 100:
            crit_val = -3.45
        elif n <= 250:
            crit_val = -3.15
        else:
            crit_val = -2.86
            
        sadf_crit_series = pd.Series(crit_val, index=sadf_stats.index)
        
        # Create rolling statistics for more granular analysis
        sadf_rolling_mean = sadf_stats.rolling(window=min(10, len(sadf_stats)//4)).mean()
        sadf_rolling_std = sadf_stats.rolling(window=min(10, len(sadf_stats)//4)).std()
        
        # Reindex to full dataset
        sadf_tstat_full = sadf_stats.reindex(out.index).ffill()
        sadf_crit_full = sadf_crit_series.reindex(out.index).ffill()
        
        # Additional features
        sadf_normalized = ((sadf_stats - sadf_rolling_mean) / sadf_rolling_std).fillna(0)
        sadf_normalized_full = sadf_normalized.reindex(out.index).ffill()
        
        # Populate columns
        out['sadf_log_tstat'] = sadf_tstat_full.fillna(0.0)
        out['sadf_log_crit'] = sadf_crit_full.fillna(crit_val)
        out['sadf_log_break'] = (sadf_tstat_full > sadf_crit_full).fillna(False).astype(int)
        out['sadf_normalized'] = sadf_normalized_full.fillna(0.0)
        
        print(f"[Info] Advanced SADF calculation successful.")
        print(f"[Info] SADF stats - Mean: {sadf_stats.mean():.4f}, Std: {sadf_stats.std():.4f}")
        print(f"[Info] Critical value used: {crit_val:.4f}")
        print(f"[Info] Structural breaks detected: {(sadf_tstat_full > sadf_crit_full).sum()} / {len(out)}")
        
    except Exception as e:
        print(f"[Warning] Advanced SADF calculation failed: {e}. Using neutral placeholders.")
        import traceback
        traceback.print_exc()
    
    return out