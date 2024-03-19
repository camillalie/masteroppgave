
import gurobipy as gp
import pandas as pd
from read_data import get_parameters, get_sets
import sys
import numpy as np

# Get parameter values
# file_parameters = 'data_generation/Parameterdata-sheets.xlsx'
file_parameters = 'Parameterdata-sheets.xlsx'
#file_parameters = 'Dummy-Parameterdata-sheets.xlsx'
# file_parameters = '../run_mastermodell_m_gammel_data/Dummy2-Parameterdata-sheets-1w.xlsx'
parameters = get_parameters(file_parameters)
#print(parameters)

print('Parameter values loaded')
# Get set values
file_sets = 'SetData-sheets.xlsx' # "data_generation/SetData-sheets.xlsx"
#file_sets = 'Dummy-SetData-sheets.xlsx'
#file_sets = '../run_mastermodell_m_gammel_data/Dummy2-SetData-sheets-1w.xlsx'
sets = get_sets(file_sets)
print('Sets values loaded')



total_emissions_per_t = {}

def SVFRRP_model(sets, params):
    """
    Set up the optimization model and run it
    """
    
    model = gp.Model()  # Establish the optimization model

    # Sets
    S = sets['Power systems']
    S2 = sets['Power systems']
    A = sets['Ages']
    N = sets['Installations']
    T = sets['Time periods']
    T1 = sets['Time periods 1']
    T2 = sets['Time periods 2']
    T2scrap = sets['TP2 scrap']
    F_s = sets['Fuel types']
    R_s = sets['Routes']
    T_max = 6
    O = sets['Scenarios']
    # T0 = {0,0}

    # Parameters
    retro_cost = params['Cost of retrofitting']                   # C^R
    aquiring_cost = params['Cost of newbuilding']             # C^N
    selling_revenue = params['Revenue']         # R^S
    fuel_cost1 = params['Cost of fuel 1']                     # C^F_fr
    fuel_cost2 = params['Cost of fuel 2']              # C^F_frw
    compatibility_rs = params['Compatibility rs']                     # T_rs
    max_time = params['Max time']                       # Max T
    demand = params['Demand']                           # D_it
    inst_in_route = params['Installations in route']             # A_ir
    emissions = params['Emissions']                     # E_fr
    max_em_period = params['Max Emissions']             # Max E_t
    initial_fleet_s_a = params['Initial fleet']     # X_sa
    scrapping_age = params['Scrapping age']             # Max A
    compatibility_fs = params['Compatibility fs']
    time_used_rs = params['Time used']
    week_to_t = 260
    newbuild_em_t = 0
    eac = 1.05
    emission_factor = 1.08
    probability = params['Probability']  # P_w
    not_buy_new = [9, 10]
    distance = params['Distance']
    
    


    # Indices for variables
    indices_retro = [(1, 9, a, t) for a in A for t in T1] + [(2, 10, a, t) for a in A for t in T1]  
    indices_retro2 = [(1, 9, a, t) for a in A 
                      for idx, t in enumerate(T2) if idx < len(T2) - 1] + [(2, 10, a, t) for a in A for idx, t in enumerate(T2) if idx < len(T2) - 1]
    indices_scrap_1 = [(s, a, t) for s in S for a in A if (a!=0) for t in T1] 
    indices_scrap_2 = [(s, a, t, w) for s in S for a in A if (a!=0) for t in T2scrap for w in O]
    indices_newpsv_1 = [(s, t) for s in S if (s not in not_buy_new) for t in T1]
    indices_newpsv_2 = [(s, t, w) for s in S if (s not in not_buy_new) for t in T2 for w in O]

    # Variables
    psv1_s_a_t = model.addVars(S, A, T1, vtype=gp.GRB.INTEGER, lb=0, name="xsat1")    # x                      
    retro_psv1_s_s_a_t = model.addVars(indices_retro, vtype=gp.GRB.INTEGER, lb=0, name="retro_psv1_s_s_a_t")                 # y^r
    new_psv1_s_t = model.addVars(indices_newpsv_1, vtype=gp.GRB.INTEGER, lb=0, name="new_psv1_s_t")                          # y^N
    scrap_psv1_s_a_t = model.addVars(indices_scrap_1, vtype=gp.GRB.INTEGER, lb=0, name="scrap_psv1_s_a_t")                   # y^S
    weekly_routes1_s_a_f_r_t = model.addVars(S, A,F_s, R_s, T1, vtype=gp.GRB.INTEGER, name="weekly_routes1_s_a_f_r_t")       #z^T

    # stage 2 variables
    psv2_s_a_t_w = model.addVars(S, A, T2, O, vtype=gp.GRB.INTEGER, lb=0, name="xsat2")                                   # x
    retro_psv2_s_s_a_t_w = model.addVars(indices_retro, vtype=gp.GRB.INTEGER, lb=0, name="retro_psv2_s_s_a_t_w")              # y^r
    new_psv2_s_t_w = model.addVars(indices_newpsv_2, vtype=gp.GRB.INTEGER, lb=0, name="new_psv2_s_t_w")                            # y^N
    scrap_psv2_s_a_t_w = model.addVars(indices_scrap_2, vtype=gp.GRB.INTEGER, lb=0, name="scrap_psv2_s_a_t_w")                   # y^S
    weekly_routes2_s_a_f_r_t_w = model.addVars(S, A,F_s, R_s, T2, O, vtype=gp.GRB.INTEGER, name="weekly_routes2_s_a_f_r_t_w")            #z^T



    ###############
    # Objective 
    ###############

    total_cost_per_t_s1 = {
        t: gp.quicksum(
            retro_cost[s, s1] * retro_psv1_s_s_a_t[s, s1, a, t]/eac**(5*(t-1))
            for s in S for a in A for s1 in S if (s, s1, a, t) in retro_psv1_s_s_a_t
        ) + gp.quicksum(
            aquiring_cost[s] * new_psv1_s_t[s, t]/eac**(5*(t-1)) for s in S if (s,t) in new_psv1_s_t
        ) - gp.quicksum(
            selling_revenue[s, a] * scrap_psv1_s_a_t[s, a, t]/eac**(5*(t-1)) for s in S for a in A if (s,a,t) in scrap_psv1_s_a_t
        ) + gp.quicksum(
            (fuel_cost1[f, r] * 0.4 if t == 1 else fuel_cost1[f, r])/eac**(5*(t-1)) * weekly_routes1_s_a_f_r_t[s,a,f,r,t] * week_to_t
            for s in S for a in A for f in F_s for r in R_s
        ) for t in T1
    }



    
    total_cost_per_t_s2 = {
        (t, w): (probability[w] * (gp.quicksum(
                    retro_cost[s, s1] * retro_psv2_s_s_a_t_w[s, s1, a, t, w] / eac**(5*(t-1))
                    for s in S for a in A for s1 in S if (s, s1, a, t, w) in retro_psv2_s_s_a_t_w
                ) + gp.quicksum(
                    aquiring_cost[s] * new_psv2_s_t_w[s, t, w] / eac**(5*(t-1)) for s in S if (s,t,w) in new_psv2_s_t_w
                ) 
                ) + gp.quicksum(
                    (fuel_cost2[f, t, w]) / eac**(5*(t-1)) * weekly_routes2_s_a_f_r_t_w[s,a,f,r,t, w] * week_to_t * distance[r]
                    for s in S for a in A for f in F_s for r in R_s
                )
        ) for t in T2 for w in O
    }

    scrap_cost_t2 = {
        (t, w): probability[w] * gp.quicksum( -
                    selling_revenue[s, a] * scrap_psv2_s_a_t_w[s, a, t, w] / eac**(5*(t-1))
                    for s in S for a in A if (s,a,t,w) in scrap_psv2_s_a_t_w)
        for t in T2scrap for w in O
    }



    total_cost = gp.quicksum(total_cost_per_t_s1[t] for t in T1) + gp.quicksum(total_cost_per_t_s2[t,w] for t in T2 for w in O) + gp.quicksum(scrap_cost_t2[t,w] for t in T2scrap for w in O)
    model.setObjective(total_cost, sense=gp.GRB.MINIMIZE)

    
    #####################
    # STAGE 1
    #####################

    # # Fuel - system Compatibility constraint
    # Fuel - system Compatibility constraint

    for t in T1:
        for r in R_s:
            for a in A: 
                for s in S: 
                    for f in F_s:
                        if (f,s) in compatibility_fs:
                            model.addConstr(
                                weekly_routes1_s_a_f_r_t[s, a, f, r, t] <= compatibility_fs[f, s] * 100,
                                name=f'compatibility_const_s1_{f}_{s}'
                            )
                            
                        else:
                            model.addConstr(
                                weekly_routes1_s_a_f_r_t[s, a, f, r, t] == 0,
                                name=f'compatibility_const_s1_{f}_{s}'
                            )
    model.update()
                   
    # Route - system compatibility constraint
    for t in T1: 
        for a in A: 
            for s in S: 
                for r in R_s:
                    model.addConstr(weekly_routes1_s_a_f_r_t[s, a,f, r, t]  <= compatibility_rs[r, s] * 60,
                                    name=f'route_system_comp_constr_s1_{s}_{a}_{r}_{t}')
                    

 
    # Constraint 5.2 (initial_fleet_balance)
    # Bruker t=1 her siden settet T har verdiene 1 og 2. Må derfor legge til 0 i settet T for å kunne sette t=0 her. Vi må finne en løsning på å få denne indekseringen helt riktig
    for s in S:
        t=0
        for a in A:    
            retrofitted_psvs_from = gp.LinExpr()
            retrofitted_psvs_to = gp.LinExpr()
            scrap_psv = gp.LinExpr()
            for s1 in S:
                if (s, s1, a, t) in retro_psv1_s_s_a_t:
                    retrofitted_psvs_from.add(retro_psv1_s_s_a_t[s, s1, a, t])  # Sum of psv retrofitted from type s
                if (s1, s, a, t) in retro_psv1_s_s_a_t:
                    retrofitted_psvs_to.add(retro_psv1_s_s_a_t[s1, s, a, t])  # Sum of psv retrofitted to type s
            if (s,a,t) in scrap_psv1_s_a_t:
                scrap_psv.add(scrap_psv1_s_a_t[s, a, t])
            model.addConstr(
                initial_fleet_s_a[s, a] - scrap_psv - retrofitted_psvs_from + retrofitted_psvs_to == psv1_s_a_t[s, a, t],
                name=f'initFleetBal_const_s1_{s}_{a}'
            )
    model.update()

    # Kan stå her uten å være med i rapporten
    # Trenger vi den egentlig??
    for s in S:
        for t in T1:
            if (s,t) in new_psv1_s_t:
                model.addConstr(
                    new_psv1_s_t[s,t] <= 5,
                    name = f'psv_buy_s1_{s}_{t}'
                )
    model.update()

    for s in S:
        for t in T2:
            for w in O:
                if (s,t,w) in new_psv2_s_t_w:
                    model.addConstr(
                        new_psv2_s_t_w[s,t, w] <= 5,
                        name = f'psv_buy_s2_{s}_{t}_{w}'
                    )
    model.update()

    #model.addConstr(new_psv_s_t[1,2]==3)

    for s in S:
        for t in sorted(T1)[1:]:
            a = 0
            scrap_psv = gp.LinExpr()
            if (s,a,t) in scrap_psv1_s_a_t:
                scrap_psv = scrap_psv1_s_a_t[s,a,t]
            if (s,t-1) in new_psv1_s_t:
                model.addConstr(
                    scrap_psv <= new_psv1_s_t[s,t-1],
                    name = f'psv_sell_s1_{s}_{t}'
                )
    model.update()

    # Constraint 5.3 (init_retrosale_fleet_balance)
    # Bruker t=1 her siden settet T har verdiene 1 og 2. Må derfor legge til 0 i settet T for å kunne sette t=0 her. Vi må finne en løsning på å få denne indekseringen helt riktig
    # sjekke retrofits, skraping og endr
    for s in S:
        for a in A:
            t = 0
            retrofitted_psvs_from = gp.LinExpr()
            for s2 in S:
                if (s, s2, a, t) in retro_psv1_s_s_a_t:
                    retrofitted_psvs_from.add(retro_psv1_s_s_a_t[s, s2, a, t])  # Sum of psv retrofitted from type s
            scrap_psv = gp.LinExpr()
            if (s,a,t) in scrap_psv1_s_a_t:
                scrap_psv = scrap_psv1_s_a_t[s,a,t]    
                # print('scrap', retrofitted_psvs_from)
            model.addConstr(
                retrofitted_psvs_from + scrap_psv <= initial_fleet_s_a[s, a],
                name=f'init_retrosale_bal_const_s1_{s}_{a}'
            )
    model.update()
    #print('check')


    # Constraint 5.4 (psv_age_balance)
    # Skal bruke a-1 og t-1. Må her summere fra ikke første index
    for s in S:
        for a in sorted(A)[1:]:
            for t in sorted(T1)[1:]:
                retrofitted_psvs_from = gp.LinExpr()
                retrofitted_psvs_to = gp.LinExpr()
                for s1 in S:
                    if (s, s1, a, t) in retro_psv1_s_s_a_t:
                        retrofitted_psvs_from.add(retro_psv1_s_s_a_t[s, s1, a, t])  # Sum of psv retrofitted from type s
                    if (s1, s, a, t) in retro_psv1_s_s_a_t:
                        retrofitted_psvs_to.add(retro_psv1_s_s_a_t[s1, s, a, t])  # Sum of psv retrofitted to type s
                model.addConstr(
                    psv1_s_a_t[s, a-1, t-1] - scrap_psv1_s_a_t[s, a, t] - retrofitted_psvs_from + retrofitted_psvs_to == psv1_s_a_t[s, a, t],
                    name=f'psv_age_bal_const_s1_{s}_{a}_{t}'
                )
    
    #5.4.2
    for s in S:
        for t in sorted(T1)[1:]:
            a = 0

            retrofitted_psvs_from = gp.LinExpr()
            retrofitted_psvs_to = gp.LinExpr()
            scrap_psv = gp.LinExpr()
            if (s,a,t) in scrap_psv1_s_a_t:
                scrap_psv = scrap_psv1_s_a_t[s,a,t]
            for s1 in S:
                if (s, s1, a, t) in retro_psv1_s_s_a_t:
                    retrofitted_psvs_from.add(retro_psv1_s_s_a_t[s, s1, a, t])  # Sum of psv retrofitted from type s
                if (s1, s, a, t) in retro_psv1_s_s_a_t:
                    retrofitted_psvs_to.add(retro_psv1_s_s_a_t[s1, s, a, t])  # Sum of psv retrofitted to type sed_psvs_to.add(retro_psv1_s_s_a_t[s2, s, a, t],0)  # Sum of psv retrofitted to type s
            if (s,t-1) in new_psv1_s_t:
                new_psv = new_psv1_s_t[s,t-1]
            model.addConstr(
                new_psv - scrap_psv - retrofitted_psvs_from + retrofitted_psvs_to == psv1_s_a_t[s, a, t], 
                name=f'psv_age_bal_const2_s1_{s}_{a}_{t}'
            )
            new_psv = 0

    # Constraint 5.5 (retrosale_fleet_balance)
    # fiks
    # forklar hvorfor det ikke er a-1 og t-1
    for s in S:
        for a in sorted(A)[1:]:
            for t in sorted(T1)[1:]:
                retrofitted_psvs_from = gp.LinExpr()
                for s2 in S:
                    if (s, s2, a, t) in retro_psv1_s_s_a_t:
                        retrofitted_psvs_from.add(retro_psv1_s_s_a_t[s, s2, a, t])  # Sum of psv retrofitted from type s
                model.addConstr(
                    retrofitted_psvs_from + scrap_psv1_s_a_t[s, a, t] <= psv1_s_a_t[s, a-1, t-1],
                    name=f'retrosale_fleet_bal_const_s1_{s}_{a}_{t}'
                )
                
    # Constraint 5.6 New builds constraint
    for s in S:
        for t in sorted(T1)[1:]:
            # print('t', t)
            a = 0
            new_psv = 0
            if (s, t-1) in new_psv1_s_t:
                new_psv = new_psv1_s_t[s,t-1]
            model.addConstr(
                psv1_s_a_t[s, a, t] == new_psv,
                name = f'psv_period_balance_const_s1_{s}_{t}'
            )
    
    #model.addConstr(new_psv_s_t[1,1]==400)
    #model.addConstr(scrap_psv_s_a_t[1,1,2]==400)

    # Constraint 5.7
    for s in S:
        for t in sorted(T1)[1:]:
            a = scrapping_age
            model.addConstr(
                scrap_psv1_s_a_t[s, a, t] == psv1_s_a_t[s, a-1, t-1],
                name = f'psv_scrapped_period_balance_const_s1_{s}_{a}_{t}'
            )

    # model.addConstr(scrap_psv_s_a_t[1,2,2]==4)
    # model.addConstr(scrap_psv_s_a_t[2,2,2]==4)
    
    # Constraint 5.8
    for t in T1:
        total_emissions = gp.LinExpr()
        for s in S:
            for a in A:
                for f in F_s:
                    for r in R_s:
                        if f == 1 or f == 2: 
                            if s == 5 or s == 6 or s == 7 or s == 8: # 80% utslipp på mindre båter
                                total_emissions.add(emissions[f, r] * weekly_routes1_s_a_f_r_t[s, a, f, r, t] * week_to_t * 0.8)
                            else: 
                                total_emissions.add(emissions[f, r] * weekly_routes1_s_a_f_r_t[s, a, f, r, t] * week_to_t)
                        else: # forgrønningsfaktor på grønne fuels
                            if s == 5 or s == 6 or s == 7 or s == 8 or s == 9: # 80% utslipp på mindre båter
                                total_emissions.add(emissions[f, r] * weekly_routes1_s_a_f_r_t[s, a, f, r, t] * week_to_t/ emission_factor**t * 0.8)
                            else: 
                                total_emissions.add(emissions[f, r] * weekly_routes1_s_a_f_r_t[s, a, f, r, t] * week_to_t/ emission_factor**t)
                           
        model.addConstr(
            total_emissions <= max_em_period[t],
            name= f'max_emissions_period_constraint_s1'
       )
        total_emissions_per_t[t] = total_emissions
        
    
    # Constraint 5.9
    
    for i in N:
        for t in T1:
            number_of_visits = gp.LinExpr()
            for s in S:
                for a in A:
                    for r in R_s:
                        for f in F_s:
                            number_of_visits.add(weekly_routes1_s_a_f_r_t[s,a,f,r,t] * inst_in_route[i,r])
            model.addConstr(
                number_of_visits >= demand[i,t], 
                name=f'demand_installation_constraint_s1_{s}_{a}_{i}_{t}_{f}'
            )
    
    # Constraint 5.10
    for s in S:
        for a in A:
            for t in T1:
                time_used_psv = gp.LinExpr()
                for r in R_s:
                    for f in F_s:
                        time_used_psv.add(time_used_rs[r, s] * weekly_routes1_s_a_f_r_t[s, a, f, r, t])
                model.addConstr(
                    time_used_psv <= max_time * psv1_s_a_t[s,a,t], 
                    name=f'psv_weekly_time_limit_constr_s1_{s}_{a}_{t}'
                )

    
    ####################
    # NON ANTICIPATIVITY 
    #####################




    # Constraint 5.13 (psv_age_balance)
    # Skal bruke a-1 og t-1. Må her summere fra ikke første index
    for w in O:   
        for s in S:
            for a in sorted(A)[1:]:
                t = 2
                retrofitted_psvs_from = gp.LinExpr()
                retrofitted_psvs_to = gp.LinExpr()
                for s2 in S:
                    if (s, s2, a, t, w) in retro_psv2_s_s_a_t_w:
                        retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                    if (s2, s, a, t, w) in retro_psv2_s_s_a_t_w:
                        retrofitted_psvs_to.add(retro_psv2_s_s_a_t_w[s2, s, a, t, w])  # Sum of psv retrofitted to type s
                model.addConstr(
                    psv1_s_a_t[s, a-1, t-1] - scrap_psv2_s_a_t_w[s, a, t, w] - retrofitted_psvs_from + retrofitted_psvs_to == psv2_s_a_t_w.get((s, a, t, w), 0),
                    name=f'psv_age_bal_const_na_{s}_{a}_{t}_{w}'
                )

        
    #5.14
    for w in O:       
        for s in S:
            t = 2
            a = 0
            retrofitted_psvs_from = gp.LinExpr()
            retrofitted_psvs_to = gp.LinExpr()
            for s2 in S:
                    if (s, s2, a, t, w) in retro_psv2_s_s_a_t_w:
                        retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                    if (s2, s, a, t, w) in retro_psv2_s_s_a_t_w:
                        retrofitted_psvs_to.add(retro_psv2_s_s_a_t_w[s2, s, a, t, w])  # Sum of psv retrofitted to type s
            scrap_psv = gp.LinExpr()
            if (s,a,t,w) in scrap_psv2_s_a_t_w:
                scrap_psv = scrap_psv2_s_a_t_w[s,a,t,w]
            new_psv = 0
            if (s, t-1) in new_psv1_s_t:
                new_psv = new_psv1_s_t[s,t-1]
            model.addConstr(
                new_psv - scrap_psv - retrofitted_psvs_from + retrofitted_psvs_to == psv2_s_a_t_w[s, a, t, w], 
                name=f'psv_age_bal_const2_na_{s}_{a}_{t}'
            )

    # Constraint 5.15
    for s in S:
        for a in sorted(A)[1:]:
            for w in O:
                t = 2
                retrofitted_psvs_from = gp.LinExpr()
                for s2 in S:
                    if (s, s2, a, t, w) in retro_psv2_s_s_a_t_w:
                        retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                scrap_psv = gp.LinExpr()
                if (s,a,t,w) in scrap_psv2_s_a_t_w:
                    scrap_psv = scrap_psv2_s_a_t_w[s,a,t,w]

                model.addConstr(
                    retrofitted_psvs_from + scrap_psv <= psv1_s_a_t[s, a-1, t-1],
                    name=f'retrosale_fleet_bal_const_na_{s}_{a}_{t}_{w}'
                )
                
    # Constraint 5.16
    for s in S:
        for w in O:
            t=2
            a = 0
            new_psv = 0
            if (s, t-1) in new_psv1_s_t:
                new_psv = new_psv1_s_t[s,t-1]
            model.addConstr(
                psv2_s_a_t_w[s, a, t, w] == new_psv ,
                name = f'psv_period_balance_const_na_{s}_{t}_{w}'
            )
    

    # Constraint 5.17
    for w in O:
        for s in S:
            t =2
            a = scrapping_age
            scrap_psv = gp.LinExpr()
            if (s,a,t,w) in scrap_psv2_s_a_t_w:
                scrap_psv = scrap_psv2_s_a_t_w[s,a,t,w]
            model.addConstr(
                scrap_psv == psv1_s_a_t[s, a-1, t-1],
                name = f'psv_scrapped_period_balance_const_na_{s}_{t}_{w}'
            )

    

    #####################
    # STAGE 2 
    #####################
                
                
    # Fuel - system Compatibility constraint

    for t in T2:
        for w in O:
            for r in R_s:
                for a in A: 
                    for s in S: 
                        for f in F_s:
                            if (f,s) in compatibility_fs:
                                model.addConstr(
                                    weekly_routes2_s_a_f_r_t_w[s, a, f, r, t, w] <= compatibility_fs[f, s] * 100,
                                    name=f'compatibility_const_s2_{f}_{s}_{t}_{w}'
                                )
                            else:
                                model.addConstr(
                                    weekly_routes2_s_a_f_r_t_w[s, a, f, r, t, w] == 0,
                                    name=f'compatibility_const_s2_{f}_{s}_{t}_{w}'
                                )
                            
    model.update()
                   
    # Route - system compatibility constraint
    for t in T2: 
        for w in O:
            for a in A: 
                for s in S: 
                    for r in R_s:
                        for f in F_s:
                            model.addConstr(weekly_routes2_s_a_f_r_t_w[s, a,f, r, t, w]  <= compatibility_rs[r, s] * 60,
                                        name=f'route_system_comp_constr_s2_{s}_{a}_{r}_{t}_{w}_{f}')
        
   

    # Kan stå her uten å være med i rapporten
    for s in S:
        for w in O:
            for t in T2:
                new_psv = 0
            if (s, t, w) in new_psv2_s_t_w:
                new_psv = new_psv2_s_t_w[s,t, w]
                model.addConstr(
                    new_psv <= 18,
                    name = f'psv_buy_s2_{s}_{t}_{w}'
                )
    model.update()

    #model.addConstr(new_psv_s_t[1,2]==3)
    for w in O:
        for s in S:
            for t in sorted(T2)[1:]:
                a = 0
                scrap_psv = gp.LinExpr()
                if (s,a,t,w) in scrap_psv2_s_a_t_w:
                    scrap_psv = scrap_psv2_s_a_t_w[s,a,t,w]
                new_psv = 0
                if (s, t-1, w) in new_psv2_s_t_w:
                    new_psv = new_psv2_s_t_w[s,t-1, w]
                model.addConstr(
                    scrap_psv <= new_psv,
                    name = f'psv_sell_s2_{s}_{t}_{w}'
                )
    model.update()



    # Constraint 5.4 (psv_age_balance)
    # Skal bruke a-1 og t-1. Må her summere fra ikke første index
    for w in O:
        for s in S:
            for a in sorted(A)[1:]:
                for t in sorted(T2)[1:]:
                    retrofitted_psvs_from = gp.LinExpr()
                    retrofitted_psvs_to = gp.LinExpr()
                    for s2 in S:
                        if (s, s2, a, t, w) in retro_psv2_s_s_a_t_w:
                            retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                        if (s2, s, a, t, w) in retro_psv2_s_s_a_t_w:
                            retrofitted_psvs_to.add(retro_psv2_s_s_a_t_w[s2, s, a, t, w])  # Sum of psv retrofitted to type s
                    scrap_psv = gp.LinExpr()
                    if (s,a,t,w) in scrap_psv2_s_a_t_w:
                        scrap_psv = scrap_psv2_s_a_t_w[s,a,t,w]
                    model.addConstr(
                        psv2_s_a_t_w[s, a-1, t-1, w] - scrap_psv - retrofitted_psvs_from + retrofitted_psvs_to == psv2_s_a_t_w[s, a, t, w],
                        name=f'psv_age_bal_const_s2_{s}_{a}_{t}_{w}'
                    )
                 
    
    #Constraint 19
    for w in O:               
        for s in S:
            for t in sorted(T2)[1:]:
                a = 0
                retrofitted_psvs_from = gp.LinExpr()
                retrofitted_psvs_to = gp.LinExpr()
                for s2 in S:
                        if (s, s2, a, t, w) in retro_psv2_s_s_a_t_w:
                            retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                        if (s2, s, a, t, w) in retro_psv2_s_s_a_t_w:
                            retrofitted_psvs_to.add(retro_psv2_s_s_a_t_w[s2, s, a, t, w])  # Sum of psv retrofitted to type s
                scrap_psv = gp.LinExpr()
                if (s,a,t,w) in scrap_psv2_s_a_t_w:
                    scrap_psv = scrap_psv2_s_a_t_w[s,a,t,w]
                new_psv = 0
                if (s, t-1, w) in new_psv2_s_t_w:
                    new_psv = new_psv2_s_t_w[s,t-1, w]
                model.addConstr(
                    new_psv - scrap_psv - retrofitted_psvs_from + retrofitted_psvs_to == psv2_s_a_t_w[s, a, t, w], 
                    name=f'psv_age_bal_const2_s2_{s}_{a}_{t}_{w}'
                )

    # Constraint 5.20 (retrosale_fleet_balance)
    
    for w in O:            
        for s in S:
            for a in sorted(A)[1:]:
                for t in sorted(T2)[1:]:
                    retrofitted_psvs_from = gp.LinExpr()
                    for s2 in S:
                        if (s, s2, a, t, w) in retro_psv2_s_s_a_t_w:
                            retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                    scrap_psv = gp.LinExpr()
                    if (s,a,t,w) in scrap_psv2_s_a_t_w:
                        scrap_psv = scrap_psv2_s_a_t_w[s,a,t,w]
                    model.addConstr(
                        retrofitted_psvs_from + scrap_psv <= psv2_s_a_t_w[s, a-1, t-1, w],
                        name=f'retrosale_fleet_bal_const_s2_{s}_{a}_{t}_{w}'
                    )
                    
    # Constraint 5.21 New builds constraint
    for w in O:                 
        for s in S:
            for t in sorted(T2)[1:]:
                # print('t', t)
                a = 0
                new_psv = 0
                if (s, t-1, w) in new_psv2_s_t_w:
                    new_psv = new_psv2_s_t_w[s,t-1, w]
                model.addConstr(
                    psv2_s_a_t_w[s, a, t, w] == new_psv,
                    name = f'psv_period_balance_const_s2_{s}_{t}_{w}'
                )
    
   

    # Constraint 5.22
    for w in O: 
        for s in S:
            for t in sorted(T2)[1:]:
                a = scrapping_age
                scrap_psv = gp.LinExpr()
                if (s,a,t,w) in scrap_psv2_s_a_t_w:
                    scrap_psv = scrap_psv2_s_a_t_w[s,a,t,w]
                model.addConstr(
                    scrap_psv == psv2_s_a_t_w[s, a-1, t-1, w],
                    name = f'psv_scrapped_period_balance_const_s2_{s}_{t}_{w}'
                )

    
    # Constraint 5.23
    for w in O:             
        for t in T2:
            if (t) in max_em_period:
                total_emissions = gp.LinExpr()
                for s in S:
                    for a in A:
                        for f in F_s:
                            for r in R_s:
                                if f == 1 or f == 2: 
                                    if s == 5 or s == 6 or s == 7 or s == 8: # 80% utslipp på mindre båter
                                        total_emissions.add(emissions[f, r] * weekly_routes2_s_a_f_r_t_w[s, a, f, r, t, w] * week_to_t * 0.8)
                                    else: 
                                        total_emissions.add(emissions[f, r] * weekly_routes2_s_a_f_r_t_w[s, a, f, r, t, w] * week_to_t)
                                else: # forgrønningsfaktor på grønne fuels
                                    if s == 5 or s == 6 or s == 7 or s == 8: # 80% utslipp på mindre båter
                                        total_emissions.add(emissions[f, r] * weekly_routes2_s_a_f_r_t_w[s, a, f, r, t, w] * week_to_t/ emission_factor**t * 0.8)
                                    else: 
                                        total_emissions.add(emissions[f, r] * weekly_routes2_s_a_f_r_t_w[s, a, f, r, t, w] * week_to_t/ emission_factor**t)
                    
                
                model.addConstr(
                    total_emissions <= max_em_period[t],
                    name= f'max_emissions_period_constraint_s2_{t}_{w}'
            )
            total_emissions_per_t[t] = total_emissions
        
    
    # Constraint 5.24
    for w in O: 
        for i in N:
            for t in T2:
                number_of_visits = gp.LinExpr()
                for s in S:
                    for a in A:
                        for r in R_s:
                            for f in F_s:
                                number_of_visits.add(weekly_routes2_s_a_f_r_t_w[s,a,f,r,t, w] * inst_in_route[i,r])
                model.addConstr(
                    number_of_visits >= demand[i,t], 
                    name=f'demand_installation_constraint_s2_{s}_{a}_{i}_{t}_{w}_{f}'
                )
    
    # Constraint 5.25
    for w in O:  
        for s in S:
            for a in A:
                for t in T2:
                    time_used_psv = gp.LinExpr()
                    for r in R_s:
                        for f in F_s:
                            time_used_psv.add(time_used_rs[r, s] * weekly_routes2_s_a_f_r_t_w[s, a, f, r, t, w])
                    model.addConstr(
                        time_used_psv <= max_time * psv2_s_a_t_w[s,a,t, w], 
                        name=f'psv_weekly_time_limit_constr_s2_{s}_{a}_{t}_{w}_{f}'
                    )
    
    
    # **Constraint: Sunset value - force the model to sell all PSVs in the last time period
    for w in O:
        for s in S:
            for a in A:
                t = T_max
                if (s,a,t,w) in scrap_psv2_s_a_t_w:
                    retrofitted_psvs_from = gp.LinExpr()
                    retrofitted_psvs_to = gp.LinExpr()
                    for s2 in S:
                        if (s, s2, a, t-1, w) in retro_psv2_s_s_a_t_w:
                            retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t-1, w])  # Sum of psv retrofitted from type s
                        if (s2, s, a, t-1, w) in retro_psv2_s_s_a_t_w:
                            retrofitted_psvs_to.add(retro_psv2_s_s_a_t_w[s2, s, a, t-1, w])  # Sum of psv retrofitted to type s
                    model.addConstr(
                             psv2_s_a_t_w[s,a, t-1,w] +  retrofitted_psvs_to - retrofitted_psvs_from - scrap_psv2_s_a_t_w[s,a,t-1,w] == scrap_psv2_s_a_t_w[s,a,t,w], 
                            name=f'sunsetvalue_{s}_{a}_{t}_{w}'
                        )
    
    # ***Constraint: Sikre at modellen ikke kjøper noen nye psver i siste periode
    for w in O:
        for s in S:
            t = sorted(T2)[-1]
            new_psv = 0
            if (s, t, w) in new_psv2_s_t_w:
                model.addConstr(new_psv2_s_t_w[s,t, w] == 0, 
                            name=f'notbuylastperiod_{s}_{t}_{w}'
                        )

    model.update()
    
    return model, T

print('Line before start running model ')
model, T = SVFRRP_model(sets, parameters)
# Set the MIPGap to 1% (0.01)
model.setParam('MIPGap', 0.05)
# Set the TimeLimit to 10 hours (36000 seconds)
model.setParam('TimeLimit', 36000)
model.optimize() 

max_em_param = parameters['Max Emissions'] 

outputfilepath = 'output_file.txt'

# # Check if the model is feasible
# if model.status == gp.GRB.OPTIMAL:
#     print("Optimal solution found!")
#     # Print variable values associated with the last feasible solution found
#     for var in model.getVars():
#         print(f"{var.varName} = {var.Xn}")
# elif model.status == gp.GRB.INFEASIBLE:
#     print("Model is infeasible.")
#     # Print variable values associated with the last feasible solution found
#     for var in model.getVars():
#         print(f"{var.varName} = {var.Xn}")
# else:
#     print("Optimization terminated with status:", model.status)
# Number of variables
#num_variables = model.NumVars

        # Number of parameters
#num_parameters = model.NumParams

if model.status == gp.GRB.OPTIMAL:
    
    print("Optimal solution found!")
    
    # Print the objective value
    print(f"Objective Value: {model.objVal}")

    # # Print the values of all variables
    # for var in model.getVars():
    #     if var.x > 0:
    #         print(f"{var.varName} = {var.x}")

    for t in T:
        print(f"Total Emissions for time period {t}: {total_emissions_per_t[t].getValue()}")
    
    with open(outputfilepath, mode='a', newline='') as file:
        # Write header
        file.write('MaxEmissions: ')
        # Write the value
        file.write(f"{max_em_param}\n")
        file.write(f"Objective Value: {model.objVal}\n")

        for var in model.getVars():
            if var.Xn > 0:
                file.write(f"{var.varName} = {var.Xn}\n")
        
#         # for t in T1:
#         #     file.write(f"Total Emissions for time period {t}: {total_emissions_per_t[t].getValue()}\n")

#         # file.write('Time Period, Total Cost\n')
#         # for t in T1:
#         #     file.write(f"{t}, {total_cost_per_t_s1[t].getValue()}\n")
                
        

        # file.write(f"Number of variables: {num_variables}")
        # file.write(f"Number of parameters: {num_parameters}")
        # print(f"Number of variables: {num_variables}")
        # print(f"Number of parameters: {num_parameters}")
            
elif model.status == gp.GRB.TIME_LIMIT:
    # Check if a feasible solution is found
    if model.SolCount > 0:
        print("A solution is found within the time limit.")
        with open(outputfilepath, mode='a', newline='') as file:
            file.write('MaxEmissions: ')
            # Write the value
            file.write(f"{max_em_param}\n")
            file.write(f"Objective Value: {model.objVal}\n")

            for var in model.getVars():
                if var.Xn > 0:
                    file.write(f"{var.varName} = {var.Xn}\n")
            
            for t in T:
                 file.write(f"Total Emissions for time period {t}: {total_emissions_per_t[t].getValue()}\n")

#             # file.write('Time Period, Total Cost\n')
#             # for t in T1:
#             #     file.write(f"{t}, {total_cost_per_t_s1[t].getValue()}\n")



else:
    print("No optimal solution found.")
    with open(outputfilepath, mode='a', newline='') as file:
        for var in model.getVars():
                if var.Xn > 0:
                    file.write(f"{var.varName} = {var.Xn}\n")