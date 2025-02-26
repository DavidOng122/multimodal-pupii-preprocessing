import method
import pandas as pd

#load the raw multimodal data from iMotions 10 
df_full = pd.read_csv("xxxxx.csv") #Replace "xxxxx.csv" with the actual filename
start_time = 1000
end_time = 200000

#emoving invalid pupil diameter, duplicate timestmps and empty data
df_filter = method.process_pupil_data(df_full,start_time,end_time)

# Find the baseline period from initial data
start_baseline = method.finds_baseline_period(df_filter, min_pupil=2, max_pupil=8, baseline_period=400)

#Remove invalid pupil data
df_left,df_right = method.remove_invalid_pupil(df_filter,min_size=2,max_size=8)

#Process left eye pupil diameter data
df_left = method.remove_outliers(df_left, pupil_column='ET_PupilLeft', time_column='Timestamp', n=20)
df_left = method.remove_outliers_with_moving_average(df_left, pupil_column='ET_PupilLeft', window_size=23, n=18)
df_left = method.pchip_interpolation_1000hz(df_left, pupil_column='ET_PupilLeft', timestamp_column='Timestamp',interp_upsampling_freq = 1000)

#Process right eye pupil diameter data
df_right = method.remove_outliers(df_right, pupil_column='ET_PupilRight', time_column='Timestamp', n=16)
df_right = method.remove_outliers_with_moving_average(df_right, pupil_column='ET_PupilRight', window_size=23, n=18)
df_right = method.pchip_interpolation_1000hz(df_right, pupil_column='ET_PupilRight', timestamp_column='Timestamp',interp_upsampling_freq = 1000)

#Compute the mean pupil diameter
df_mean = method.compute_mean_pupil_size(df_left, df_right)

#Compute the normalize dilation pupil diameter
df = method.normalize_pupil_dilation(df_mean, start_baseline, start_time, end_time)

#Select the emotion expression to integrate with pupil diameter data.
selected_columns = ['Unnamed: 1', 'Unnamed: 25', 'Unnamed: 28', 'Unnamed: 32', 'Unnamed: 34','Unnamed: 35', 'Unnamed: 41','Unnamed: 27', 'Unnamed: 42']
df_merge = method.preprocess_integrate_emotions(df_full, df,selected_columns, start_time, end_time)

#print the first few rows of the merged DataFrame
print(df_merge.head())