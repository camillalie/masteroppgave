import pandas as pd
from itertools import permutations
import math
import os
import csv
import time

cargo_capacity_psv = 100
psv_speed = 10
max_platforms_in_one_voyage = 7

non_cluster = 'general_routes'
mappenavn = 'cluster23_new/23_1'

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
route_file = f'{mappenavn}/routes.csv'

def write_to_csv(filename, data):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

write_to_csv(route_file, routes_only)

input_filename = f'{mappenavn}/routes.csv'
output_filename = f'{mappenavn}/processed_routes.csv'

def split_into_threes(word):
    if len(word) > 3:
        return ','.join([word[i:i+3] for i in range(0, len(word), 3)])
    else:
        return word

def process_csv_file(input_filename, output_filename):
    with open(input_filename, 'r', newline='') as file, open(output_filename, 'w', newline='') as outfile:
        reader = csv.reader(file)
        writer = csv.writer(outfile)

        for row in reader:
            processed_row = []
            for column in row:
                processed_segments = [split_into_threes(segment) for segment in column.split(',')]
                processed_row.append(','.join(processed_segments))

            outfile.write(','.join(processed_row) + '\n')

process_csv_file(input_filename, output_filename)

with open(output_filename, 'r') as processed_file:
    print(processed_file.read())



def find_matching_row(csv_file, names):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader, start=1):
            if set(names).issubset(row):
                return row_number, row
    return None, None

def get_platform_visits(platform):
    with open(f'{non_cluster}/output_platforms_visits.csv', 'r', newline='') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            if row['platform'] == platform:
                return int(row['visits'])
    return None

def remove_lowest_demand_platforms(row):
    visits_count = {}
    for platform in row:
        if platform != 'route_number' and platform != 'total_distance':
            visits = get_platform_visits(platform)
            if visits is not None:
                if visits in visits_count:
                    visits_count[visits].append(platform)
                else:
                    visits_count[visits] = [platform]
    
    if visits_count:
        lowest_demand = min(visits_count)
        for platform in visits_count[lowest_demand]:
            row.remove(platform)

def compute_weights(route_data):
    visits = [get_platform_visits(platform) for platform in route_data if platform not in ['route_number', 'total_distance']]
    visits = [v for v in visits if v is not None]  
    if not visits:
        return [0]
    
    min_visit = min(visits)
    max_visit = max(visits)
    if max_visit > 0:
        return min_visit / max_visit
    return 0

def compute_weight_incrementally(current_min, previous_min, max_visit):
    if max_visit > 0:
        return (current_min - previous_min) / max_visit
    return 0

def normalize_weights(weights):
    total = sum(weights)
    return [w / total for w in weights]

def compute_min_visit(row_data):
    visits = [get_platform_visits(platform) for platform in row_data if platform not in ['route_number', 'total_distance']]
    visits = [v for v in visits if v is not None] 
    return min(visits) if visits else float('inf')  

def count_unique_visits(row):
    unique_visits = set()
    for platform in row:
        if platform != 'route_number' and platform != 'total_distance':
            visits = get_platform_visits(platform)
            if visits is not None:
                unique_visits.add(visits)
    return len(unique_visits)

def print_row_and_distance(csv_file, row_number):
    if row_number:
        with open(csv_file, 'r', newline='') as file:
            reader = csv.reader(file)
            for i, row_num in enumerate(reader, start=1):
                if i == row_number:
                    print(f"Row number: {row_number}")
                    print("Row:", row_num)
                    break
        with open(f'{non_cluster}/distances.csv', 'r', newline='') as distance_file:
            reader = csv.reader(distance_file)
            for i, row_dist in enumerate(reader, start=1):
                if i == row_number + 1:  
                    print("Distance Row:", row_dist)
                    break
        with open(f'{non_cluster}/demand.csv', 'r', newline='') as distance_file:
            reader = csv.reader(distance_file)
            for i, row_demand in enumerate(reader, start=1):
                if i == row_number + 1:  
                    print("Demand Row:", row_demand)
                    break
    return row_dist, row_demand

def process_routes():
    with open(f'{mappenavn}/distances.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Distances'])
    with open(f'{mappenavn}/demand.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Demand'])

    non_matching_rows = []
    weighted_distances = []
    weighted_demands = []
    with open(f'{mappenavn}/processed_routes.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            total_weighted_distance = 0  
            total_weighted_demand = 0  
            row_number, original_row_data = find_matching_row(f'{non_cluster}/routes.csv', row)
            if row_number is not None:
                print("Processing row:", row)  
                print("Found matching row:", row_number)  
                distance, demand = print_row_and_distance(f'{non_cluster}/routes.csv', row_number)

                visits = [get_platform_visits(platform) for platform in original_row_data if platform not in ['route_number', 'total_distance']]
                visits = [v for v in visits if v is not None]  
                print("Visits:", visits)  

                if visits:
                    max_visit = max(visits)
                    min_visit = min(visits)
                    weights = [min_visit / max_visit if max_visit > 0 else 0]

                    unique_visits_count = count_unique_visits(original_row_data) - 1
                    routes_found = 0
                    row_data = original_row_data[:]
                    previous_min = min_visit

                    weighted_distance = float(distance[0]) * weights[0]  
                    total_weighted_distance += weighted_distance

                    weighted_demand = float(demand[0]) * weights[0]  
                    total_weighted_demand += weighted_demand

                    while len(row_data) > 2 and routes_found < unique_visits_count:
                        remove_lowest_demand_platforms(row_data)
                        row_number, new_row_data = find_matching_row(f'{non_cluster}/routes.csv', row_data)
                        if row_number is not None:
                            print("Found matching row in loop:", row_number)  
                            print_row_and_distance(f'{non_cluster}/routes.csv', row_number)
                            current_min = compute_min_visit(new_row_data)
                            if current_min != float('inf'): 
                                weight = compute_weight_incrementally(current_min, previous_min, max_visit)
                                weights.append(weight)
                                previous_min = current_min
                                with open(f'{non_cluster}/distances.csv', 'r', newline='') as distance_file:
                                    reader = csv.reader(distance_file)
                                    for i, row_distance in enumerate(reader, start=1):
                                        if i == row_number + 1:  
                                            weighted_distance = float(row_distance[0]) * weight  
                                            total_weighted_distance += weighted_distance  
                                            weighted_distances.append(weighted_distance)
                                with open(f'{non_cluster}/demand.csv', 'r', newline='') as demand_file:
                                    reader = csv.reader(demand_file)
                                    for i, row_demand in enumerate(reader, start=1):
                                        if i == row_number + 1:  
                                            weighted_demand = float(row_demand[0]) * weight  
                                            total_weighted_demand += weighted_demand  
                                            weighted_demands.append(weighted_demand)

                            row_data = new_row_data[:]
                            routes_found += 1
                    
                    total_weight = sum(weights)
                    if total_weight > 0:
                        normalized_weights = [weight / total_weight for weight in weights]
                        for route, weight in zip(range(len(normalized_weights)), normalized_weights):
                            print(f"Route {route + 1} Weight: {weight}")
                    
            print(f"Total Weighted Distance for this route: {total_weighted_distance}")
            print(f"Total Weighted Demand for this route: {total_weighted_demand}")

            with open(f'{mappenavn}/distances.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([total_weighted_distance])

            with open(f'{mappenavn}/demand.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([total_weighted_demand])

process_routes()

zero_rows = []
with open(f'{mappenavn}/distances.csv', 'r') as file:
    next(file)  # Skip the header line
    for i, line in enumerate(file, start=1):  # Start enumeration from 1
        if float(line.strip()) == 0:
            zero_rows.append(i)

print("Row numbers with 0 values:", zero_rows)

# Read routes.csv line by line and remove rows based on zero_rows
with open(f'{mappenavn}/routes.csv', 'r') as routes_file:
    routes_data = routes_file.readlines()

# Remove rows from routes_data based on zero_rows
routes_data = [route for i, route in enumerate(routes_data, start=1) if i not in zero_rows]

# Write the updated routes_data back to routes.csv
with open(f'{mappenavn}/routes.csv', 'w') as routes_file:
    routes_file.writelines(routes_data)

demand_df = pd.read_csv(f'{mappenavn}/demand.csv')
demand_df = demand_df[demand_df['Demand'] != 0]
demand_df.to_csv(f'{mappenavn}/demand.csv', index=False)

# Remove rows from distances.csv
distances_df = pd.read_csv(f'{mappenavn}/distances.csv')
distances_df = distances_df[distances_df['Distances'] != 0]
distances_df.to_csv(f'{mappenavn}/distances.csv', index=False)


demand_df = pd.read_csv(f'{mappenavn}/demand.csv')
duration_lossing = round(((demand_df['Demand'] * 1.389) / psv_speed), 2)
duration_loss_df = pd.DataFrame({'Duration (hours)': duration_lossing})
duration_loss_df.to_csv(f'{mappenavn}/duration_lossing.csv', index=False)

distances_df = pd.read_csv(f'{mappenavn}/distances.csv')
duration_sailing = round((distances_df['Distances'] / psv_speed), 2)
duration_sailing_df = pd.DataFrame({'Duration (hours)': duration_sailing})
duration_sailing_df.to_csv(f'{mappenavn}/duration_sailing.csv', index=False)