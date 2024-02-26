import numpy as np
import pandas as pd

def generate_fuel_prices(num_scenarios, num_periods, fuel_dict):
    num_fuels = len(fuel_dict)
    
    # Generate random scaling factors for each group
    group_scaling_factors = {}
    for fuel_key, (_, _, group) in fuel_dict.items():
        if group not in group_scaling_factors:
            group_scaling_factors[group] = np.random.uniform(0, 1, (num_scenarios, num_periods))
    
    # Generate random prices for each fuel within bounds for each time period in each scenario
    fuel_prices = np.zeros((num_fuels, num_periods, num_scenarios))
    for i, (fuel_key, (lower_bound, upper_bound, group)) in enumerate(fuel_dict.items()):
        for scenario in range(num_scenarios):
            for period in range(num_periods):
                scaling_factor = group_scaling_factors[group][scenario, period]
                fuel_prices[i, period, scenario] = lower_bound + (upper_bound - lower_bound) * scaling_factor
                if period > 0:
                    fuel_prices[i, period, scenario] = max(fuel_prices[i, period, scenario], fuel_prices[i, period-1, scenario])
    
    # Grouping fuels into four groups
    fuel_groups = {}
    for fuel_key, (_, _, group) in fuel_dict.items():
        if group not in fuel_groups:
            fuel_groups[group] = []
        fuel_groups[group].append(fuel_key)
    
    # Generating correlated prices within each group
    for group, fuels_in_group in fuel_groups.items():
        # Generating correlated random numbers for this group
        num_group_fuels = len(fuels_in_group)
        means = np.random.uniform(0, 1, num_group_fuels)
        
        # Define covariance matrix with specified correlation
        if group == 'Group A':
            # For Group A, set 100% correlation
            cov_matrix = np.eye(num_group_fuels)
        else:
            # For other groups, set 70% correlation
            cov_matrix = np.full((num_group_fuels, num_group_fuels), 0.7)
            np.fill_diagonal(cov_matrix, 1.0)
        
        # Generate correlated random numbers using Cholesky decomposition
        correlated_values = np.random.multivariate_normal(means, cov_matrix, num_scenarios)
        
        # Normalizing the correlated values
        correlated_values -= correlated_values.min(axis=0)
        correlated_values /= correlated_values.max(axis=0)
        
        # Scaling the correlated values to the group bounds
        for i, fuel_key in enumerate(fuels_in_group):
            lower_bound, upper_bound, _ = fuel_dict[fuel_key]
            for scenario in range(num_scenarios):
                fuel_prices[list(fuel_dict.keys()).index(fuel_key), :, scenario] = lower_bound + (upper_bound - lower_bound) * correlated_values[scenario, i]
    
    return fuel_prices

# Example usage
num_scenarios = 10
num_periods = 5
fuel_dict = {
    'Fuel_1': (1, 3, 'Group A'),
    'Fuel_2': (2, 4, 'Group B'),
    'Fuel_3': (3, 5, 'Group A'),
    'Fuel_4': (4, 6, 'Group B')
}  # Example fuel bounds with group information

# Generate fuel prices
# fuel_prices = generate_fuel_prices(num_scenarios, num_periods, fuel_dict)
# print(fuel_prices)


# # Generate fuel prices
# fuel_prices = generate_fuel_prices(num_scenarios, num_periods, fuel_dict)

# # Example distances for routes
# group_four_distances = load_distances_from_csv('group_four_distances.csv')
# default_distances = load_distances_from_csv('default_distances.csv')

# # Create DataFrame to store parameter values
# data = []
# for scenario in range(num_scenarios):
#     for period in range(num_periods):
#         for fuel_type, (_, _, group) in fuel_dict.items():
#             distances = group_four_distances if group == 'Group B' else default_distances
#             for route, distance in distances.items():
#                 fuel_price = fuel_prices[scenario, period, list(fuel_dict.keys()).index(fuel_type)]
#                 data.append([fuel_type, route, period, scenario, fuel_price * distance])

# parameters_df = pd.DataFrame(data, columns=['fuel type', 'route', 'time period', 'scenario', 'fuel price'])
# print(parameters_df)

