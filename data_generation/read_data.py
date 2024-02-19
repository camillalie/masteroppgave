import pandas as pd
import numpy as np
import math

file_path = 'Parameterdata-sheets.xlsx'
sheet_dictionaries = {}


def get_parameters(filename):

    sheet_names = {'Cost of retrofitting', 'Cost of newbuilding', 'Cost of fuel', 'Time used',
                    'Revenue', 'Demand', 'Compatibility rs', 'Installations in route', 'Max Emissions', 'Initial fleet', 'Emissions', 'Compatibility fs'}
    sheet_dictionaries = {}

    for sheet in sheet_names:
        df = pd.read_excel(filename, sheet_name=sheet)
        data_dict = {}

        for index, row in df.iterrows():
            # Drop the 'Value' column and convert the rest to a tuple
            key_tuple = tuple(row.drop('Value'))
            # Explicitly convert numeric values to integers if they are whole numbers
            key_tuple = tuple(int(item) if isinstance(item, float) and item.is_integer() else item for item in key_tuple)
            
            # Check the length of the tuple and store accordingly
            if len(key_tuple) == 1:
                key = key_tuple[0]
            else:
                key = key_tuple
            
            data_dict[key] = row['Value']
        
        sheet_dictionaries[sheet] = data_dict
    
    sheet_dictionaries['Scrapping age'] = 5
    sheet_dictionaries['Max time'] = 168
    
    return sheet_dictionaries

# parameters = get_parameters(file_path)
# print(parameters['Cost of fuel'])

# print(parameters)

def print_nan_or_inf_in_double_dict(double_dict):
    for outer_key, inner_dict in double_dict.items():
        if isinstance(inner_dict, dict):
            for inner_key, value in inner_dict.items():
                if math.isnan(value) or math.isinf(value):
                    print(f"NaN or Inf found at [{outer_key}][{inner_key}]: {value}")
        else:
            print(f"Warning: Value at {outer_key} is not a dictionary.")


# print_nan_or_inf_in_double_dict(parameters)


def compare_dictionaries(dict1, dict2):
    # Check if both dictionaries have the same keys
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    # Keys present in dict1 but not in dict2
    missing_keys_in_dict2 = keys1 - keys2
    if missing_keys_in_dict2:
        print("Keys present in the first dictionary but not in the second:", missing_keys_in_dict2)

    # Keys present in dict2 but not in dict1
    missing_keys_in_dict1 = keys2 - keys1
    if missing_keys_in_dict1:
        print("Keys present in the second dictionary but not in the first:", missing_keys_in_dict1)

    # Compare values for common keys
    common_keys = keys1 & keys2
    for key in common_keys:
        if dict1[key] != dict2[key]:
            print(f"Values for key '{key}' are different:")
            print(f"First dictionary: {dict1[key]}")
            print(f"Second dictionary: {dict2[key]}")
            print("------")

# Assuming the result from your get_parameters function is stored in a variable named parameters

# compare_dictionaries(parameters, testparams)
# parameters = get_parameters(file_path)

# print(parameters)

def get_sets(filename):
    sheet_names = {'Power systems', 'Ages', 'Routes',
                   'Time periods', 'Installations', 'Fuel types'}
    sheet_dictionaries = {}

    for sheet in sheet_names:
        df = pd.read_excel(filename, sheet_name=sheet)
        set_values = set()  # Initialize an empty set

        for column in df.columns:
            # Add non-null values from the column to the set
            set_values.update(df[column].dropna().tolist())

        sheet_dictionaries[sheet] = set_values

    
    return sheet_dictionaries

setssheet = 'run_model/SetData-sheets.xlsx'

# print(get_sets(setssheet))