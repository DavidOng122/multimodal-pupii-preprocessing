import Pupil_Preprocessing_Functions as method
import pandas as pd

# Load the raw multimodal data from iMotions 10 
# If this line raises a FileNotFoundError, replace "data.csv" with the full file path to data.csv
df_full = pd.read_csv('data.csv')
start_time = 77740
end_time = 1192364

# Removing invalid pupil diameter, duplicate timestmps and empty data
df_filter = method.process_pupil_data(df_full,start_time,end_time)

# Find the baseline period from initial data
start_baseline = method.finds_baseline_period(df_filter, min_pupil=2, max_pupil=8, baseline_period=400)

# Remove invalid pupil data
df_left,df_right = method.remove_invalid_pupil(df_filter,min_size=2,max_size=8)

# Process left eye pupil diameter data
df_left = method.remove_outliers(df_left, pupil_column='ET_PupilLeft', time_column='Timestamp', n=20)
df_left = method.remove_outliers_with_moving_average(df_left, pupil_column='ET_PupilLeft', window_size=23, n=18)
method.plot_left_pupil(df_left, pupil_column='ET_PupilLeft', timestamp_column='Timestamp',
                    start_time=2900, end_time=4100, y_min=4.2, y_max=5.1)

# Interpolate left eye pupil diameter data 
df_pchip = method.pchip_interpolation_1000hz(df_left, pupil_column='ET_PupilLeft', timestamp_column='Timestamp',interp_upsampling_freq = 1000)
df_makima = method.makima_interpolation_1000hz(df_left, pupil_column='ET_PupilLeft', timestamp_column='Timestamp',interp_upsampling_freq = 1000)
df_akima = method.akima_interpolation_1000hz(df_left, pupil_column='ET_PupilLeft', timestamp_column='Timestamp',interp_upsampling_freq = 1000)
df_left = df_pchip
# Plot left eye pupil diameter data over time
method.plot_left_pupil(df_pchip, pupil_column='ET_PupilLeft', timestamp_column='Timestamp',
                    start_time=2900, end_time=4100, y_min=4.2, y_max=5.1)

# Process right eye pupil diameter data
df_right = method.remove_outliers(df_right, pupil_column='ET_PupilRight', time_column='Timestamp', n=16)
df_right = method.remove_outliers_with_moving_average(df_right, pupil_column='ET_PupilRight', window_size=23, n=18)
df_right = method.pchip_interpolation_1000hz(df_right, pupil_column='ET_PupilRight', timestamp_column='Timestamp',interp_upsampling_freq = 1000)

# Compute the mean pupil diameter
df_mean = method.compute_mean_pupil_size(df_left, df_right)

# Compute the normalize dilation pupil diameterChin Raise
df = method.normalize_pupil_dilation(df_mean, start_baseline, start_time, end_time)

# Select the emotion expression to integrate with pupil diameter data.
selected_columns = ['Unnamed: 1', 'Unnamed: 25', 'Unnamed: 28', 'Unnamed: 32', 'Unnamed: 34','Unnamed: 35', 'Unnamed: 41','Unnamed: 27', 'Unnamed: 42']
df_merge = method.preprocess_integrate_emotions(df_full, df,selected_columns, start_time, end_time)

# # Print the first few rows of the merged DataFrame
print(df_merge.head())

# Visualize pupil diameter data
method.plot_pupil_diameter(df_merge,start_time,end_time)

# Output merged data
df_merge.to_csv('result.csv', index=False)

