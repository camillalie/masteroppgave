import pandas as pd
from itertools import permutations
import math
import os
import csv
import time

cargo_capacity_psv = 100
psv_speed = 10
max_platforms_in_one_voyage = 7

mappenavn = 'cluster13_new'

demand = pd.read_csv(f'{mappenavn}/output_platforms_demand.csv', header=0, delimiter=';')
distances = pd.read_csv(f'{mappenavn}/output_distance_matrix_kmeans.csv', header=0, delimiter=';', index_col='from/to')

platforms_demand = dict(zip(demand['platform'], demand['avg_q'].replace(',', '.').astype(float)))
platforms_d = ['MON'] + demand['platform'].tolist() + ['MON']  # Add 'DUS' as start and end platforms

shortest_routes_dict = {}

def generate_routes(demand, distances):
    cargo_capacity = cargo_capacity_psv
    max_platforms = max_platforms_in_one_voyage + 2

    def dp(platform, cargo_remaining, route, visited):
        if cargo_remaining < 0:
            return
        if len(route) > max_platforms:
            return
        if platform == 'MON' and len(route) > 2:
            total_demand = sum(platforms_demand[p] for p in route[1:-1])
            if total_demand <= cargo_capacity:
                key = tuple(sorted(set(route)))
                total_distance = sum(distances.loc[route[i], route[i+1]] for i in range(len(route)-1))
                if key not in shortest_routes_dict or total_distance < shortest_routes_dict[key][1]:
                    duration_sailing = round((total_distance / psv_speed), 2)
                    duration_lossing = round(((total_demand * 1.389) / psv_speed), 2)
                    duration_sailing = round(duration_sailing, 2)
                    duration_lossing = round(duration_lossing, 2)
                    shortest_routes_dict[key] = (route, total_distance, total_demand, duration_sailing, duration_lossing)
            return

        # Check if the current route is dominated
        current_distance = sum(distances.loc[route[i], route[i+1]] for i in range(len(route)-1))
        current_demand = sum(platforms_demand[p] for p in route[1:-1])
        
        # Check for dominance in existing routes
        for key, (existing_route, existing_distance, existing_demand, _, _) in shortest_routes_dict.items():
            if set(existing_route[1:-1]) == set(route[1:-1]) and existing_demand >= current_demand and existing_distance <= current_distance:
                return
        
        # Check for dominance in subsequent routes starting with the same platforms
        for existing_route in dominated_routes:
            if set(existing_route[1:-1]) == set(route[1:-1]) and existing_demand >= current_demand and existing_distance <= current_distance:
                return

        for next_platform in platforms_demand.keys():
            if next_platform != platform and next_platform not in visited:
                try:
                    distance_to_next = distances.loc[platform, next_platform]
                    new_cargo_remaining = cargo_remaining - platforms_demand[next_platform]
                    dp(next_platform, new_cargo_remaining, route + [next_platform], visited.union({next_platform}))
                except KeyError:
                    print("KeyError occurred for platform:", next_platform)
                    continue

    for r in range(3, min(len(platforms_demand) + 3, max_platforms + 3)):
        for route_combination in permutations(platforms_demand.keys(), r - 2):
            route = ['MON'] + list(route_combination) + ['MON']
            dp('MON', cargo_capacity, route, {'MON'})

            # Add the current route to the dominated routes set
            dominated_routes.add(tuple(route))

    return shortest_routes_dict

# Keep track of dominated routes to skip future routes starting with them
dominated_routes = set()
shortest_routes_dict = generate_routes(demand, distances)

for route, distance, demand, duration_sailing, duration_lossing in shortest_routes_dict.values():
    print(f"Shortest Route: {route}, Total Distance: {round(distance,2)}, Total Demand: {demand}, Duration sailing: {duration_sailing}, Duration lossing: {duration_lossing}")

print(len(shortest_routes_dict))

routes_only = [route for route, _, _, _, _ in shortest_routes_dict.values()]
distances_only = [distance for _, distance, _, _, _ in shortest_routes_dict.values()]
demand_only = [demand for _, _, demand, _, _ in shortest_routes_dict.values()]
duration_sailing = [duration_sailing for _, _, _, duration_sailing, _ in shortest_routes_dict.values()]
duration_lossing = [duration_lossing for _, _, _, _, duration_lossing in shortest_routes_dict.values()]

df_distances = pd.DataFrame({'Distance': distances_only})
df_demand = pd.DataFrame({'Demand': demand_only})
df_duration_sailing = pd.DataFrame({'Duration (hours)': duration_sailing})
df_duration_lossing = pd.DataFrame({'Duration (hours)': duration_lossing})

df_distances.to_csv(f'{mappenavn}/distances.csv', index=False)
df_demand.to_csv(f'{mappenavn}/demand.csv', index=False)
df_duration_sailing.to_csv(f'{mappenavn}/duration_sailing.csv', index=False)
df_duration_lossing.to_csv(f'{mappenavn}/duration_lossing.csv', index=False)
route_file = f'{mappenavn}/routes.csv'


def write_to_csv(filename, data):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

write_to_csv(route_file, routes_only)