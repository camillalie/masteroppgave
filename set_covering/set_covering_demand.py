import pandas as pd
import numpy as np
import math
import csv
import gurobipy as gp

#file_path = '../run_model/Parameterdata-sheets.xlsx'
# setssheet = 'data_generation/SetData-sheets.xlsx'
#file_path = 'Dummy-Parameterdata-sheets.xlsx'
sheet_dictionaries = {}


import csv
import gurobipy as gp

def set_covering_model():
    model = gp.Model()

    # Initialize dictionaries and lists
    routes_dict = {}
    platform_to_number = {}
    mj_route_dict = {}
    demand_dict = {}

    # List of platforms with corresponding numbers
    platforms = [
        "GFAGFBGFC", "APT", "STASTB", "ASL", "DAB", "OSE", 
        "DSA", "DSS", "STC", "KVB", "MID", "NLNVAL", "OSC", "OSOVFB", "OSS", "TENTRB", "TEQTRC", "TRO"
    ]
    

    # Create mapping from platform to number
    for idx, platform in enumerate(platforms, start=1):
        platform_to_number[platform] = idx

    # Read the CSV files and populate the dictionaries
    with open('../route_generation_mongstad/generated_datafiles_setcover100/routes.csv', 'r') as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader, 1):
            platform_numbers = [platform_to_number[platform] for platform in row if platform !=  "MON" ] # endre til MON
            routes_dict[row_number] = platform_numbers
            

    with open('../route_generation_mongstad/generated_datafiles_setcover100/mj_route.csv', 'r') as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader, 1):
            value = float(row[0])
            mj_route_dict[row_number] = value
    
    with open('../route_generation_mongstad/generated_datafiles_setcover100/demand.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row_number, row in enumerate(reader, 1):
            value = float(row[0])
            demand_dict[row_number] = value
    #print('Mj route dict: ', mj_route_dict)
    #print('demand dict: ', demand_dict)


    platform_visits_path = '../route_generation_mongstad/clustering/output_platforms_visits.csv'
    df = pd.read_csv(platform_visits_path, delimiter=';')

    # Create a dictionary from the DataFrame
    platform_dict = {idx + 1: row['visits'] for idx, row in df.iterrows()}

    print(platform_dict)

    

    # Define sailed_routes variable
    routes = range(1, len(routes_dict))
    sailed_routes = model.addVars(routes, vtype=gp.GRB.INTEGER, name="sailed_routes_r")

    # Objective function
    total_distance = gp.quicksum(mj_route_dict[r] * sailed_routes[r] for r in routes)
    total_demand = gp.quicksum(demand_dict[r] * sailed_routes[r] for r in routes)
    model.setObjective(total_distance + total_demand*1500, sense=gp.GRB.MINIMIZE)

    
    for platform_num in platform_to_number.values():
        covering_routes = [r for r in routes if platform_num in routes_dict[r]]
        visited_platform = gp.quicksum(sailed_routes[r] for r in covering_routes)
        model.addConstr(visited_platform >= platform_dict[platform_num], name=f"visit_at_least_once_{platform_num}")
    
    # for route in routes:
    #     model.addConstr(sailed_routes[route]*demand_dict[route] <= 80, name=f'el_demand_{route}')
    
    model.update()
    model.optimize()
    for platform_num in platform_to_number.values():
        visited_platform = gp.quicksum(sailed_routes[r] for r in routes if platform_num in routes_dict[r])
        print(f"Constraint Value for platform {platform_num}: {visited_platform.getValue()}")


  

    return model, sailed_routes



def write_output_file(sailed_routes):
    with open('output_routes_c100.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Route Number', 'Sailed'])
        
        for r, sailed_var in sailed_routes.items():
            if sailed_var.x > 0:  # Check if the sailed value is positive
                writer.writerow([r, sailed_var.x])

# Create and solve the model
model, sailed_routes = set_covering_model()
model.optimize()

# Write output file
write_output_file(sailed_routes)




