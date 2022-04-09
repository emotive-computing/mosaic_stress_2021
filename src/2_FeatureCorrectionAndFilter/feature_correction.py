import os
import pdb
import pytz
import numpy as np
import pandas as pd
from datetime import datetime

features_file_path = '../1_BaselineLabels/results/merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline.csv'
#out_file_path = './results/merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_all_feats.csv'
out_file_path = './results/merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_non_survey_context_feats.csv'
#keep_cols_file_path = './keep_all_cols.csv' # List of columns to retain after correction
#keep_cols_file_path = './keep_non_survey_context_cols.csv' # List of columns to retain after correction
keep_cols_file_path = './keep_non_survey_context_cols2.csv' # List of columns to retain after correction

tz_zip_dict = {}
dst_zip_dict = {}
known_zipcodes = []
def GetTimeZoneFromZipcode(zipcode):
    global tz_zip_dict
    global dst_zip_dict
    global known_zipcodes

    if np.isnan(zipcode):
        return None

    zipcode = str(int(zipcode))

    # One-time initialization
    if len(tz_zip_dict) == 0:
        with open('tz.data', 'r') as tz_infile:
            for line in tz_infile:
                zc, tz = line.rstrip().split('=')
                tz_zip_dict[zc] = tz
        with open('dst.data', 'r') as dst_infile:
            for line in dst_infile:
                zc, dst = line.rstrip().split('=')
                dst_zip_dict[zc] = int(dst) == 1
        known_zipcodes = sorted([int(x) for x in tz_zip_dict.keys()])


    timezone = None
    if zipcode in tz_zip_dict.keys():
        best_zipcode = zipcode
    else:
        # Find the closest zipcode
        best_zipcode = next((x for x in known_zipcodes if int(zipcode) < x),None)
        print("Unknown zipcode: %s. Zipcode difference = %d"%(zipcode, abs(best_zipcode-int(zipcode))))
    timezone = tz_zip_dict["%05d"%(int(best_zipcode))]

    #if timezone is not None:
    #    observes_dst = dst_zip_dict[best_zipcode]
        
    return timezone

def DoFeatureCorrection():
    df = pd.read_csv(features_file_path)
    day_sunrise = df['day_sunrise']
    day_sunset = df['day_sunset']

    survey_starttime = df['survey_start_datetime']
    survey_endtime = df['survey_end_datetime']
    survey_senttime = df['survey_sent_datetime']
    zipcode = df['zipcode']

    df['survey_start_seconds_since_midnight'] = np.nan*np.zeros(df.shape[0])
    df['survey_end_seconds_since_midnight'] = np.nan*np.zeros(df.shape[0])
    df['survey_sent_seconds_since_midnight'] = np.nan*np.zeros(df.shape[0])

    # Adjust the survey times so they are represented in local 24-hour time
    for row_idx in range(df.shape[0]):
        zipc = zipcode[row_idx]
        timezone = GetTimeZoneFromZipcode(zipc)

        if timezone is not None:
            local_timezone = pytz.timezone(timezone)
            if not np.isnan(survey_starttime[row_idx]):
                survey_start = datetime.fromtimestamp(survey_starttime[row_idx], local_timezone)
                seconds_since_midnight_survey_start = (survey_start - survey_start.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                df.loc[row_idx, 'survey_start_seconds_since_midnight'] = seconds_since_midnight_survey_start

            if not np.isnan(survey_endtime[row_idx]):
                survey_end = datetime.fromtimestamp(survey_endtime[row_idx], local_timezone)
                seconds_since_midnight_survey_end = (survey_end - survey_end.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                df.loc[row_idx, 'survey_end_seconds_since_midnight'] = seconds_since_midnight_survey_end

            if not np.isnan(survey_senttime[row_idx]):
                survey_sent = datetime.fromtimestamp(survey_senttime[row_idx], local_timezone)
                seconds_since_midnight_survey_sent = (survey_sent - survey_sent.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                df.loc[row_idx, 'survey_sent_seconds_since_midnight'] = seconds_since_midnight_survey_sent


    df['day_sunrise_seconds_since_midnight'] = np.nan*np.zeros(df.shape[0])
    df['day_sunset_seconds_since_midnight'] = np.nan*np.zeros(df.shape[0])

    # Use the sunrise time to group participants.  Find the mode zipcode and use it to convert
    # the sunrise time from unix time back to local time
    unique_day_sunrises = np.unique(day_sunrise)
    for unique_day_sunrise in unique_day_sunrises:
        row_mask = df['day_sunrise'] == unique_day_sunrise
        sunrise_group_zipcode = df.loc[row_mask,'zipcode'].mode()
        if len(sunrise_group_zipcode) > 0:
            sunrise_group_zipcode = sunrise_group_zipcode[0] # Pick one
            timezone = GetTimeZoneFromZipcode(sunrise_group_zipcode)
            
            if timezone is not None:
                local_timezone = pytz.timezone(timezone)
                local_time_sunrise = datetime.fromtimestamp(unique_day_sunrise, local_timezone)
                seconds_since_midnight_sunrise = (local_time_sunrise - local_time_sunrise.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                df.loc[row_mask, 'day_sunrise_seconds_since_midnight'] = seconds_since_midnight_sunrise

        else:
            # Do nothing for now and fixup these NaN zipcode values later
            pass

    unique_day_sunsets = np.unique(day_sunset)
    for unique_day_sunset in unique_day_sunsets:
        row_mask = df['day_sunset'] == unique_day_sunset
        sunset_group_zipcode = df.loc[row_mask,'zipcode'].mode()
        if len(sunset_group_zipcode) > 0:
            sunset_group_zipcode = sunset_group_zipcode[0] # Pick one
            timezone = GetTimeZoneFromZipcode(sunset_group_zipcode)
            
            if timezone is not None:
                local_timezone = pytz.timezone(timezone)
                local_time_sunset = datetime.fromtimestamp(unique_day_sunset, local_timezone)
                seconds_since_midnight_sunset = (local_time_sunset - local_time_sunset.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                df.loc[row_mask, 'day_sunset_seconds_since_midnight'] = seconds_since_midnight_sunset

        else:
            # Do nothing for now and fixup these NaN zipcode values later
            pass

    
    # Adjust the sunrise and sunset times for entries with no zipcode and no sunrise/sunset group
    sorted_unique_day_sunrises = sorted(unique_day_sunrises.astype(int))
    adjusted_sunrise_row_mask = pd.isnull(df['day_sunrise_seconds_since_midnight'])
    nan_row_idx = np.where(adjusted_sunrise_row_mask)[0]
    for idx in nan_row_idx:
        if np.isnan(df['day_sunrise'][idx]):
            continue
        sunrise_unixtime = int(df['day_sunrise'][idx])
        sunrise_idx = sorted_unique_day_sunrises.index(sunrise_unixtime)
        next_sunrise_idx = min(sunrise_idx+1,len(sorted_unique_day_sunrises)-1)
        prev_sunrise_idx = max(sunrise_idx-1,0)
        sunrise_unixtime_diff = np.diff(sorted_unique_day_sunrises[prev_sunrise_idx:next_sunrise_idx+1])
        if sunrise_unixtime_diff[0] < sunrise_unixtime_diff[1]:
            nearest_adjusted_sunrise_time = df['day_sunrise_seconds_since_midnight'][df['day_sunrise'] == sorted_unique_day_sunrises[prev_sunrise_idx]].values[0]
        else:
            nearest_adjusted_sunrise_time = df['day_sunrise_seconds_since_midnight'][df['day_sunrise'] == sorted_unique_day_sunrises[next_sunrise_idx]].values[0]
        df.loc[idx,'day_sunrise_seconds_since_midnight'] = nearest_adjusted_sunrise_time
            
    sorted_unique_day_sunsets = sorted(unique_day_sunsets.astype(int))
    adjusted_sunset_row_mask = pd.isnull(df['day_sunset_seconds_since_midnight'])
    nan_row_idx = np.where(adjusted_sunset_row_mask)[0]
    for idx in nan_row_idx:
        if np.isnan(df['day_sunset'][idx]):
            continue
        sunset_unixtime = int(df['day_sunset'][idx])
        sunset_idx = sorted_unique_day_sunsets.index(sunset_unixtime)
        next_sunset_idx = min(sunset_idx+1,len(sorted_unique_day_sunsets)-1)
        prev_sunset_idx = max(sunset_idx-1,0)
        sunset_unixtime_diff = np.diff(sorted_unique_day_sunsets[prev_sunset_idx:next_sunset_idx+1])
        if sunset_unixtime_diff[0] < sunset_unixtime_diff[1]:
            nearest_adjusted_sunset_time = df['day_sunset_seconds_since_midnight'][df['day_sunset'] == sorted_unique_day_sunsets[prev_sunset_idx]].values[0]
        else:
            nearest_adjusted_sunset_time = df['day_sunset_seconds_since_midnight'][df['day_sunset'] == sorted_unique_day_sunsets[next_sunset_idx]].values[0]
        df.loc[idx,'day_sunset_seconds_since_midnight'] = nearest_adjusted_sunset_time

    # Adjust the survey times with missing zipcodes using the sunrise or sunset time groups to infer timezone
    nan_row_idx = np.where(pd.isnull(df['zipcode']))[0]
    for row_idx in nan_row_idx:
        sunrise_seconds_since_midnight = df.loc[row_idx, 'day_sunrise_seconds_since_midnight']
        sunset_seconds_since_midnight = df.loc[row_idx, 'day_sunset_seconds_since_midnight']

        timezone = None
        row_mask = df['day_sunrise_seconds_since_midnight'] == sunrise_seconds_since_midnight
        sunrise_group_zipcode = df.loc[row_mask,'zipcode'].mode()
        if len(sunrise_group_zipcode) > 0:
            sunrise_group_zipcode = sunrise_group_zipcode[0] # Pick one
            timezone = GetTimeZoneFromZipcode(sunrise_group_zipcode)

        if timezone is None: # try again with sunset data
            row_mask = df['day_sunset_seconds_since_midnight'] == sunset_seconds_since_midnight
            sunset_group_zipcode = df.loc[row_mask,'zipcode'].mode()
            if len(sunset_group_zipcode) > 0:
                sunset_group_zipcode = sunset_group_zipcode[0] # Pick one
                timezone = GetTimeZoneFromZipcode(sunset_group_zipcode)
            
        if timezone is not None:
            local_timezone = pytz.timezone(timezone)
            if not np.isnan(survey_starttime[row_idx]):
                survey_start = datetime.fromtimestamp(survey_starttime[row_idx], local_timezone)
                seconds_since_midnight_survey_start = (survey_start - survey_start.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                df.loc[row_idx, 'survey_start_seconds_since_midnight'] = seconds_since_midnight_survey_start

            if not np.isnan(survey_endtime[row_idx]):
                survey_end = datetime.fromtimestamp(survey_endtime[row_idx], local_timezone)
                seconds_since_midnight_survey_end = (survey_end - survey_end.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                df.loc[row_idx, 'survey_end_seconds_since_midnight'] = seconds_since_midnight_survey_end

            if not np.isnan(survey_senttime[row_idx]):
                survey_sent = datetime.fromtimestamp(survey_senttime[row_idx], local_timezone)
                seconds_since_midnight_survey_sent = (survey_sent - survey_sent.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                df.loc[row_idx, 'survey_sent_seconds_since_midnight'] = seconds_since_midnight_survey_sent
        
    return df

def Run():
    df = DoFeatureCorrection()

    # Filter the results, dropping the rows where labels are null
    keep_cols = pd.read_csv(keep_cols_file_path, header=None).values.flatten()
    df = df.loc[:,keep_cols]
    df = df.dropna(subset=['stress.d'])

    if 'id' not in df.columns:
        df['id'] = range(df.shape[0])

    if not os.path.isdir(os.path.dirname(out_file_path)):
        os.makedirs(os.path.dirname(out_file_path))
    df.to_csv(out_file_path, index=False, header=True)

    return

if __name__ == '__main__':
    Run()

