import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression

def process_pupil_data(file_path,min_pupil = 2,max_pupil = 8,start_time = 77740,end_time = 1192364):
    """
    This function processes a raw multimodal dataset from the iMotions dataset.
    
    Parameters
    ---------
    file_path: str
        file_path to the raw multimodal dataset (csv file).
    min_pupil : float, default : 2 mm
        Minimum valid pupil diameter.
    max_pupil : float, default :  8 mm
        Maximum valid pupil diameter.
    start_time : int, default :  77740
        The beginning of the test or analysis period.
    end_time : int, default : 1192364
        The end of the test or analysis period.
        
    Returns
    ------
    df_filtered : pabdas.DataFrame
        Filtered DataFrame within the annotation range, with invalid pupil diameter data removed, duplicate timestamps eliminated, and NaN values dropped.
        
    Notes
    ------
    - The valid range of pupil diameter is between 2 mm and 8 mm.
    - For this experiment, the start time is the moment of inserting the scope,
      and the end time is the moment of withdrawing the scope.

    """
    # Load the CSV file into a Pandas DataFrame
    df_full = pd.read_csv(file_path)
    
    # Select relevant columns from the raw dataset:
    # - 'Unnamed: 1' represents the timestamp.
    # - 'Unnamed: 84' represents the left eye pupil diameter.
    # - 'Unnamed: 85' represents the right eye pupil diameter.
    df_selected_full = df_full[['Unnamed: 1', 'Unnamed: 84', 'Unnamed: 85']]
    
    # Remove the first 27 rows as they are merely column descriptions, 
    # and drop any remaining NaN since facial expressions have different sampling rates.
    df_selected_full = df_selected_full.iloc[27:].dropna()
    
    # Rename columns using the first row as headers.
    df_selected_full.columns = df_selected_full.iloc[0]
    
    # Remove the first row after renaming and reset the index for cleaner access.
    df_selected_full = df_selected_full[1:].reset_index(drop=True)

    # Convert relevant columns to numeric format for data processing.
    df_selected_full['Timestamp'] = pd.to_numeric(df_selected_full['Timestamp'])
    df_selected_full['ET_PupilRight'] = pd.to_numeric(df_selected_full['ET_PupilRight'])
    df_selected_full['ET_PupilLeft'] = pd.to_numeric(df_selected_full['ET_PupilLeft'])

   

    # Identify the first valid row where both pupil diameters are :
    valid_rows = df_selected_full[
        (df_selected_full['ET_PupilRight'] > min_pupil) & (df_selected_full['ET_PupilRight'] < max_pupil) &
        (df_selected_full['ET_PupilLeft'] > min_pupil) & (df_selected_full['ET_PupilLeft'] < max_pupil)
    ]
    
    # Makesure there is at least one valid row.
    if valid_rows.empty:
        raise ValueError("No valid rows in the data")
    
    first_row = valid_rows.iloc[0]
    
    # Filter the dataset to include only data within the defined time range:
    # - Data starts from the first valid row's timestamp to include some baseline measurements before the actual test.
    # - Data ends at the manually defined `end_time`.
    df_filtered = df_selected_full[
        (df_selected_full['Timestamp'] >= first_row['Timestamp']) &
        (df_selected_full['Timestamp'] <= end_time)
    ]
    
    # Drop remaining NaN values
    df_filtered = df_filtered.dropna()

    # Remove duplicate timestamps, keeping the first occurrence
    df_filtered = df_filtered.drop_duplicates(subset='Timestamp', keep='first')
 
    
    return df_filtered


def finds_baseline_period(df_filtered, min_pupil=2, max_pupil=8, baseline_period=400):
    """
    Finds the start of the baseline period in the unprecessed dataset.

    Parameters
    ---------
    df_filtered : pandas.DataFrame
        The DataFrame containing pupil diameter data.
    min_pupil : float, default : 2 mm
        Minimum valid pupil diameter.
    max_pupil : float, default :  8 mm
        Maximum valid pupil diameter.
    baseline_period : int, default : 400 ms
        The time period (in milliseconds) used to calculate the baseline pupil diameter.
        
    Returns
    ------
    start_baseline : float
        The starting timestamp for the baseline period, rounded to 0 decimal places.

    """

    #Find the baseline timestamp range
    i = 0
    df_baseline = df_filtered[
        (df_filtered['Timestamp'] >= df_filtered['Timestamp'].iloc[0]) &
        (df_filtered['Timestamp'] <= df_filtered['Timestamp'].iloc[0] + baseline_period)
    ]
    # Check if the baseline window contains only valid rows
    while ((df_baseline['ET_PupilLeft'] <= min_pupil).any()|(df_baseline['ET_PupilLeft'] >= max_pupil).any()
            |(df_baseline['ET_PupilRight'] <= min_pupil).any()|(df_baseline['ET_PupilRight'] >= max_pupil).any() |df_baseline.empty):
        # Increment index if the current baseline window is invalid
        i += 1
        # Define the baseline window for the current iteration
        df_baseline = df_filtered[
            (df_filtered['Timestamp'] > df_filtered['Timestamp'].iloc[0 + i]) &
            (df_filtered['Timestamp'] <= df_filtered['Timestamp'].iloc[0 + i] + baseline_period)
        ]
    #Get the starting timestamp of the valid baseline range
    start_baseline = df_baseline['Timestamp'].iloc[0]
    # Round to 0 decimal places for consistency (e.g., for interpolation to 1000Hz)
    start_baseline = start_baseline.round(0)
    return start_baseline

def remove_invalid_pupil(df,min_size=2,max_size=8):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        DESCRIPTION.
    column : TYPE
        DESCRIPTION.
    min_size : TYPE, optional
        DESCRIPTION. The default is 2.
    max_size : TYPE, optional
        DESCRIPTION. The default is 8.

    Returns
    -------
    df_filtered without invalid pupil size
    """
    
    # Filter for valid left pupil data
    df_left = df[['Timestamp', 'ET_PupilLeft']][(df['ET_PupilLeft'] >= min_size) & (df['ET_PupilLeft'] <= max_size)]
    
    # Filter for valid right pupil data
    df_right = df[['Timestamp', 'ET_PupilRight']][(df['ET_PupilRight'] >= min_size) & (df['ET_PupilRight'] <= max_size)]
    
    return df_left, df_right

def remove_outliers(df, pupil_column='ET_PupilLeft', time_column='Timestamp', n=20):
    """
    Detects and removes outliers in pupil diameter data based on a normalized dilation speed metric (d_prime).
    Outliers are identified using the Median Absolute Deviation (MAD) method.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the pupil diameter and timestamp data.
    pupil_column : str, optional
        The name of the column containing pupil diameter measurements (default is 'ET_PupilLeft').
    time_column : str, optional
        The name of the column containing timestamp data (default is 'Timestamp').
    n : int, optional
        The scaling factor used to determine the outlier detection threshold (default is 20).

    Returns
    -------
    df_cleaned : pandas.DataFrame
        A DataFrame with the outliers removed based on the computed threshold.
    """

    # Extract the relevant columns
    df_pupil = df[pupil_column]
    df_time = df[time_column]
    d_prime = []

    # Calculate the normalized dilation speed metric (d_prime)
    for i in range(len(df_pupil)):
        if i == 0:
            d_prime.append(abs((df_pupil.iloc[i + 1] - df_pupil.iloc[i]) / (df_time.iloc[i + 1] - df_time.iloc[i])))
        elif i == len(df) - 1:
            d_prime.append(abs((df_pupil.iloc[i] - df_pupil.iloc[i - 1]) / (df_time.iloc[i] - df_time.iloc[i - 1])))
        else:
            d_prev = abs((df_pupil.iloc[i] - df_pupil.iloc[i - 1]) / (df_time.iloc[i] - df_time.iloc[i - 1]))
            d_next = abs((df_pupil.iloc[i + 1] - df_pupil.iloc[i]) / (df_time.iloc[i + 1] - df_time.iloc[i]))
            d_prime.append(max(d_prev, d_next))

    # Add d_prime to the DataFrame
    df['d_prime'] = d_prime

    # Calculate the Median Absolute Deviation (MAD)
    median_p = np.median(df['d_prime'])
    mad = []
    for i in range(len(d_prime)):
     mad.append(abs(d_prime[i]-median_p))
    df['mad'] = mad

    # Compute the threshold for outlier detection
    median_mad = np.median(df['mad'])
    threshold = median_p + n * median_mad

    # Identify outliers and remove them
    df['is_outlier'] = df['d_prime'] > threshold
    df_cleaned = df[df['is_outlier'] == 0]  # Keep non-outlier rows

    # Drop auxiliary columns before returning
    df_cleaned = df_cleaned.drop(columns=['d_prime', 'mad', 'is_outlier'])

    return df_cleaned

def remove_outliers_with_moving_average(df, pupil_column='ET_PupilLeft', window_size=23, threshold_multiplier=18):
    """
    Detects and removes outliers in pupil diameter data using a rolling mean and a deviation threshold.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the pupil diameter data.
    pupil_column : str, optional
        The name of the column containing pupil diameter measurements (default is 'ET_PupilLeft').
    window_size : int, optional
        The size of the rolling window for calculating the moving average (default is 23).
    threshold_multiplier : int or float, optional
        The multiplier for the deviation threshold based on the median absolute deviation (default is 18).

    Returns
    -------
    df_cleaned : pandas.DataFrame
        The cleaned DataFrame with outliers removed.
    df_processed : pandas.DataFrame
        The DataFrame with additional columns ('rolling_mean', 'deviation', 'is_outlier', 'drop') for analysis.
    """
    # Copy the input DataFrame to avoid modifying it directly
    df_processed = df.copy()

    # Calculate the rolling mean
    df_processed['rolling_mean'] = df_processed[pupil_column].rolling(window=window_size, center=True).mean()

    # Calculate deviations from the rolling mean
    df_processed['deviation'] = np.abs(df_processed[pupil_column] - df_processed['rolling_mean'])

    # Calculate the threshold for outlier detection
    threshold = threshold_multiplier * df_processed['deviation'].median()

    # Identify outliers
    df_processed['is_outlier'] = df_processed['deviation'] > threshold

    # Mark all rows as not dropped initially
    df_processed['drop'] = False

    # Mark outliers for dropping
    df_processed.loc[df_processed['is_outlier'], 'drop'] = True

    # Remove outliers from the DataFrame
    df_cleaned = df_processed[~df_processed['is_outlier']].reset_index(drop=True)

    return df_cleaned

def preprocess_pupil_data(df, pupil_column='ET_PupilLeft', timestamp_column='Timestamp',
                          interp_max_gap=250, interp_upsampling_freq=1000, lowpass_cutoff=4, butter_order=4):
    """
    Preprocess pupil diameter data by upsampling, interpolating, applying a low-pass filter, and handling large gaps.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the pupil diameter and timestamp data.
    pupil_column : str, optional
        Name of the column containing pupil diameter measurements (default is 'ET_PupilLeft').
    timestamp_column : str, optional
        Name of the column containing timestamps (default is 'Timestamp').
    interp_max_gap : int, optional
        Maximum allowable gap (in milliseconds) for interpolation (default is 250).
    interp_upsampling_freq : int, optional
        Frequency (in Hz) for upsampling during interpolation (default is 1000).
    lowpass_cutoff : int, optional
        Cutoff frequency (in Hz) for the low-pass filter (default is 4).
    butter_order : int, optional
        Order of the Butterworth low-pass filter (default is 4).

    Returns
    -------
    df_processed : pandas.DataFrame
        DataFrame containing the preprocessed and interpolated pupil diameter data.
    """
    # Calculate timestamp differences
    df['TimeDiff'] = df[timestamp_column].diff()

    # Extract valid timestamps and pupil diameter values
    valid_t_ms = df[timestamp_column].values
    valid_diams = df[pupil_column].values

    # Design the Butterworth low-pass filter
    LpFilt_B, LpFilt_A = butter(butter_order, 2 * lowpass_cutoff / interp_upsampling_freq)

    # Generate upsampled time series
    t_upsampled = np.arange(valid_t_ms[0] / 1000, valid_t_ms[-1] / 1000, 1 / interp_upsampling_freq)

    # Perform linear interpolation
    interp_func = interp1d(valid_t_ms / 1000, valid_diams, kind='linear')
    dia_interp = interp_func(t_upsampled)

    # Apply low-pass filtering
    dia_interp = filtfilt(LpFilt_B, LpFilt_A, dia_interp)

    # Handle gaps
    gaps_raw = df['TimeDiff'].iloc[1:].values
    gaps_raw = np.append(gaps_raw, gaps_raw[-1])  # Extend the last gap value
    bin_indices = np.digitize(t_upsampled * 1000, df[timestamp_column].values)
    gaps_ms = gaps_raw[np.clip(bin_indices - 1, 0, len(gaps_raw) - 1)]

    # Set tolerance for valid points
    not_touching_tolerance_ms = (0.5 * 1000) / interp_upsampling_freq
    min_time = df[timestamp_column].min() - not_touching_tolerance_ms / 1000
    max_time = df[timestamp_column].max() + not_touching_tolerance_ms / 1000
    t_upsampled = t_upsampled[(t_upsampled * 1000 >= min_time) & (t_upsampled * 1000 <= max_time)]
    close_to_valid = np.zeros(len(t_upsampled), dtype=bool)

    # Check proximity of upsampled points to valid timestamps in chunks
    chunk_size = 1000  # Adjust based on memory capacity
    for chunk_start in range(0, len(t_upsampled), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(t_upsampled))
        chunk = t_upsampled[chunk_start:chunk_end] * 1000
        distances = np.abs(chunk[:, None] - df[timestamp_column].values)
        chunk_close = np.any(distances <= not_touching_tolerance_ms, axis=1)
        close_to_valid[chunk_start:chunk_end] = chunk_close

    # Remove interpolated points with gaps greater than the maximum allowable gap
    gaps_ms[close_to_valid] = 0
    dia_interp[gaps_ms > interp_max_gap] = np.nan

    # Create processed DataFrame
    df_processed = pd.DataFrame({
        'Timestamp': t_upsampled * 1000,  # Convert back to milliseconds
        pupil_column: dia_interp
    }).dropna()

    # Round timestamps to nearest millisecond
    df_processed['Timestamp'] = df_processed['Timestamp'].round(0)

    return df_processed
