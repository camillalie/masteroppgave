import pandas as pd
from itertools import combinations
import math
import os
import csv
import time


start_time = time.time()
#### DYNAMIC PROGRAM WITH LABELLING/DOMINANCE CHECK
def generate_routes(demand, distances):
    platforms_demand = dict(zip(demand['platform'], demand['avg_q'].replace(',', '.').astype(float)))
    platforms_d = ['MON'] + demand['platform'].tolist() + ['MON']  # Add 'Mon' as start and end platforms

    distances_matrix = distances.map(lambda x: float(x.replace(',', '.')))

    shortest_routes_dict = {}
    cargo_capacity = 100
    max_platforms = 9

    def dp(platform, cargo_remaining, route, visited):
        if cargo_remaining < 0:
            return
        if len(route) > max_platforms:
            return
        if platform == 'MON' and len(route) > 2:
            total_demand = sum(platforms_demand[p] for p in route[1:-1])
            if total_demand <= cargo_capacity:
                key = tuple(sorted(set(route)))
                total_distance = sum(distances_matrix.loc[route[i], route[i+1]] for i in range(len(route)-1))
                if key not in shortest_routes_dict or total_distance < shortest_routes_dict[key][1]:
                    duration_sailing = round((total_distance / 10), 2)
                    duration_lossing = round(((total_demand * 1.389) / 10), 2)
                    duration_sailing = round(duration_sailing, 2)
                    duration_lossing = round(duration_lossing, 2)
                    shortest_routes_dict[key] = (route, total_distance, total_demand, duration_sailing, duration_lossing)
            return
        
        for next_platform in platforms_demand.keys():
            if next_platform != platform and next_platform not in visited:
                try:
                    distance_to_next = distances_matrix.loc[platform, next_platform]
                    new_cargo_remaining = cargo_remaining - platforms_demand[next_platform]
                    dp(next_platform, new_cargo_remaining, route + [next_platform], visited.union({next_platform}))
                except KeyError:
                    print("KeyError occurred for platform:", next_platform)
                    continue

    for r in range(3, min(len(platforms_demand) + 3, max_platforms + 3)):
        for route_combination in combinations(platforms_demand.keys(), r - 2):
            route = ['MON'] + list(route_combination) + ['MON']
            total_distance = 0
            total_demand = 0
            duration_lossing = 0
            valid_route = True
            
            for i in range(len(route) - 1):
                from_platform = platforms_d.index(route[i])
                to_platform = platforms_d.index(route[i + 1])
                total_distance += distances_matrix.iloc[from_platform, to_platform]
                if route[i] != 'MON':
                    total_demand += platforms_demand[route[i]]
                    if total_demand > cargo_capacity:
                        valid_route = False
                        break
            
            if valid_route:
                if total_demand <= cargo_capacity:  # Adjusted condition
                    dp('MON', cargo_capacity, route, {'MON'})

    return shortest_routes_dict

demand = pd.read_csv('input_data/platforms_demand_25.csv', header=0, delimiter=';')
distances = pd.read_csv('input_data/distance_matrix_25.csv', header=0, delimiter=';', index_col='from/to')

shortest_routes_dict = generate_routes(demand, distances)

for route, distance, demand, duration_sailing, duration_lossing in shortest_routes_dict.values():
    print(f"Shortest Route: {route}, Total Distance: {round(distance,2)}, Total Demand: {demand}, Duration sailing: {duration_sailing}, Duration lossing: {duration_lossing}")
print(len(shortest_routes_dict))


shortest_routes_dict_cap = {}

for key, (route, distance, demand, duration_sailing, duration_lossing) in shortest_routes_dict.items():
    if demand > 70:
        shortest_routes_dict_cap[key] = (route, distance, demand, duration_sailing, duration_lossing)






shortest_routes_dict_cap_idle = {}
routes_to_remove = []

for key, (route, distance, demand, duration_sailing, duration_lossing) in shortest_routes_dict_cap.items():
    total_duration = duration_sailing + duration_lossing
    if 16 < total_duration < 24 or 40 < total_duration < 48:
        routes_to_remove.append(key)
    else:
        shortest_routes_dict_cap_idle[key] = (route, distance, demand, duration_sailing, duration_lossing)

for key in routes_to_remove:
    del shortest_routes_dict_cap[key]




routes_only = [route for route, _, _, _, _ in shortest_routes_dict_cap.values()]
distances_only = [distance for _, distance, _, _, _ in shortest_routes_dict_cap.values()]
demand_only = [demand for _, _, demand, _, _ in shortest_routes_dict_cap.values()]
duration_sailing = [duration_sailing for _, _, _, duration_sailing, _ in shortest_routes_dict_cap.values()]
duration_lossing = [duration_lossing for _, _, _, _, duration_lossing in shortest_routes_dict_cap.values()]

df_distances = pd.DataFrame({'Distance': distances_only})
df_demand = pd.DataFrame({'Demand': demand_only})
df_duration_sailing = pd.DataFrame({'Duration (hours)': duration_sailing})
df_duration_lossing = pd.DataFrame({'Duration (hours)': duration_lossing})

df_distances.to_csv('generated_datafiles/distances25.csv', index=False)
df_demand.to_csv('generated_datafiles/demand25.csv', index=False)
df_duration_sailing.to_csv('generated_datafiles/duration_sailing25.csv', index=False)
df_duration_lossing.to_csv('generated_datafiles/duration_lossing25.csv', index=False)

route_file = "generated_datafiles/routes_strøm.csv"

def write_to_csv(filename, data):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

write_to_csv(route_file, routes_only)
print("Running time", time.time() - start_time)