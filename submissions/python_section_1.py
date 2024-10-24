from typing import Dict, List

import pandas as pd


#Q1
from typing import List
def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        group = []
        for j in range(min(n, len(lst) - i)):
            group.insert(0, lst[i + j])
        result.extend(group)
    return result

print(reverse_by_n([1, 2, 3, 4, 5, 6, 7, 8], 3)) 
print(reverse_by_n([1, 2, 3, 4, 5], 2))           
print(reverse_by_n([10, 20, 30, 40, 50, 60, 70], 4))  
​
​
​#Q2
from typing import List, Dict
def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    return dict(sorted(length_dict.items()))

lst1 = ["apple", "bat", "car", "elephant", "dog", "bear"]
lst2 = ["one", "two", "three", "four"]

print(group_by_length(lst1))  
print(group_by_length(lst2)) 
​
​
​#Q3
from typing import Dict
def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def flatten_helper(d, parent_key=''):
        flat_dict = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flat_dict.update(flatten_helper(v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    flat_dict.update(flatten_helper({f"{new_key}[{i}]": item}))
            else:
                flat_dict[new_key] = v
        return flat_dict
​
    return flatten_helper(nested_dict)

nested_dict = {
     "road": {
         "name": "Highway 1",
         "length": 350,
         "sections": [
              {
                 "id": 1,
                   "condition": {
                      "pavement": "good",
                      "traffic": "moderate"
                   }
             }
        ]
     }
 }

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)
​
​
#Q4​
from typing import List
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  
            return
        
        used = set()
        for i in range(start, len(nums)):
            if nums[i] not in used: 
                used.add(nums[i])  
                nums[start], nums[i] = nums[i], nums[start]  
                backtrack(start + 1) 
                nums[start], nums[i] = nums[i], nums[start]  
​
    nums.sort()
    result = []
    backtrack(0)
    return result

input_nums = [1, 1, 2]
output = unique_permutations(input_nums)
print(output)
​
​
​#Q5
import re
from typing import List
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.
​
    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  
        r'\b\d{2}/\d{2}/\d{4}\b', 
        r'\b\d{4}\.\d{2}\.\d{2}\b'  
    ]
    
    combined_pattern = '|'.join(date_patterns)
    
    matches = re.findall(combined_pattern, text)
    
    return matches

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))
​
​
#Q6​
import polyline
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the earth (specified in decimal degrees).
    Returns distance in meters.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
​
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371000  
    return r * c
​
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.
​
    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coords = polyline.decode(polyline_str)
​
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
​
    df['distance'] = 0.0
​
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
​
    return df

if __name__ == "__main__":
    polyline_str = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
    df = polyline_to_dataframe(polyline_str)
    print(df)
​
​
#Q7​
from typing import List
def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element
    with the sum of all elements in the same row and column (in the rotated matrix),
    excluding itself.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the square matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    
    n = len(matrix)
    
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n-1-i] = matrix[i][j]
​
    final_matrix = [[0] * n for _ in range(n)]
​
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
​
    return final_matrix
​
matrix = [
     [1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]    ]
    
final_matrix = rotate_and_transform_matrix(matrix)

print(f"final_matrix = {final_matrix}")
​
​#Q8
import pandas as pd
​df = pd.read_csv('dataset-1.csv')

date_format = "%Y-%m-%d" 
time_format = "%H:%M:%S" 
​
df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format=f"{date_format} {time_format}", errors='coerce')
df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format=f"{date_format} {time_format}", errors='coerce')
​
if df['start'].isnull().any() or df['end'].isnull().any():
    print("There were errors in parsing dates. Please check your date/time format and values.")
    print(df[['startDay', 'startTime', 'endDay', 'endTime']][df['start'].isnull() | df['end'].isnull()])
​
grouped = df.groupby(['id', 'id_2'])
​
results = []
​
starting_date = pd.Timestamp('2024-01-01')
​
for (id_val, id_2_val), group in grouped:
    days_covered = set()
​
    for day_offset in range(7):
        day_start = starting_date + pd.Timedelta(days=day_offset)
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
​
        day_data = group[(group['start'] <= day_end) & (group['end'] >= day_start)]
        
        if not day_data.empty and (day_data['start'].min() <= day_start) and (day_data['end'].max() >= day_end):
            days_covered.add(day_start.date())
​
    all_days_covered = len(days_covered) == 7
    full_24_hour_coverage = (group['end'].max() - group['start'].min()).total_seconds() >= 24 * 3600
​
    results.append((id_val, id_2_val, not (all_days_covered and full_24_hour_coverage)))
​
result_series = pd.Series({(id_val, id_2_val): incorrect for id_val, id_2_val, incorrect in results})
result_series.index = pd.MultiIndex.from_tuples(result_series.index, names=['id', 'id_2'])
​
print(result_series)   
