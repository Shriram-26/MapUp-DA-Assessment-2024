#Q9
import pandas as pd
import numpy as np

def calculate_distance_matrix(csv_file):
    df = pd.read_csv(csv_file)
    
    required_columns = {'id_start', 'id_end', 'distance'}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV must contain 'id_start', 'id_end', and 'distance' columns")

    locations = set(df['id_start']).union(set(df['id_end']))
    locations = list(locations)
    
    distance_matrix = pd.DataFrame(index=locations, columns=locations)
    distance_matrix = distance_matrix.fillna(np.inf)

    for index, row in df.iterrows():
        loc_a = row['id_start']
        loc_b = row['id_end']
        distance = row['distance']
        
        distance_matrix.loc[loc_a, loc_b] = distance
        distance_matrix.loc[loc_b, loc_a] = distance

    np.fill_diagonal(distance_matrix.values, 0)

    for k in locations:
        for i in locations:
            for j in locations:
                if distance_matrix.loc[i, k] + distance_matrix.loc[k, j] < distance_matrix.loc[i, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]
                    
    distance_matrix.replace(np.inf, np.nan, inplace=True)

    print("Distance Matrix:")
    print(distance_matrix)

    return distance_matrix

distance_df = calculate_distance_matrix('dataset-2.csv')
print("Returned Distance DataFrame:")
print(distance_df)


#Q10
import pandas as pd

def unroll_distance_matrix(distance_matrix):
    unrolled_data = []

    locations = distance_matrix.index

    for i in locations:
        for j in locations:
            if i != j:
                distance = distance_matrix.loc[i, j]
                if not pd.isna(distance):
                    unrolled_data.append({'id_start': i, 'id_end': j, 'distance': distance})

    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

unrolled_df = unroll_distance_matrix(distance_df)
print("Unrolled Distance DataFrame:")
print(unrolled_df)


#Q11
def find_ids_within_ten_percentage_threshold(df, reference_value):
    related_distances = df[df['id_start'] == reference_value]['distance']

    if related_distances.empty:
        return []

    average_distance = related_distances.mean()

    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1
 
    ids_within_threshold = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]
   
    return sorted(ids_within_threshold['id_start'].tolist())

reference_value = unrolled_df['id_start']

ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_value)

print("IDs within 10% of average distance for reference ID {}: {}".format(reference_value, ids_within_threshold))


#Q12
import pandas as pd

def calculate_toll_rate(df):
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    return df

result_df = calculate_toll_rate(unrolled_df)
print(result_df)


#Q13
import pandas as pd
from datetime import time

def calculate_time_based_toll_rates(df):
    time_ranges = {
        'weekdays': [
            (time(0, 0), time(10, 0), 0.8),
            (time(10, 0), time(18, 0), 1.2),
            (time(18, 0), time(23, 59, 59), 0.8)
        ],
        'weekends': [
            (time(0, 0), time(23, 59, 59), 0.7)
        ]
    }

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    records = []

    for day in days:
        if day in ['Saturday', 'Sunday']:
            discount = time_ranges['weekends'][0][2] 
        else:
            discount_factors = []
            for start, end, factor in time_ranges['weekdays']:
                discount_factors.append((start, end, factor))

        for hour in range(24):
            start_time = time(hour, 0)
            end_time = time(hour, 59, 59)

            if day not in ['Saturday', 'Sunday']:
                for start, end, factor in discount_factors:
                    if start_time >= start and end_time <= end:
                        discount = factor
                        break

            for _, row in df.iterrows():
                discounted_row = {
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'moto': row['moto'] * discount,
                    'car': row['car'] * discount,
                    'rv': row['rv'] * discount,
                    'bus': row['bus'] * discount,
                    'truck': row['truck'] * discount
                }
                records.append(discounted_row)

    result_df = pd.DataFrame(records)

    return result_df

results_df = calculate_time_based_toll_rates(result_df)
print(results_df)
