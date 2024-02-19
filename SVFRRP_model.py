
import gurobipy as gp




total_emissions_per_t = {}

def SVFRP_model(sets, params):
    """
    Set up the optimization model and run it
    """
    
    model = gp.Model()  # Establish the optimization model

    # Sets
    S = sets['Power systems']
    S2 = sets['Power systems']
    A = sets['Ages']
    N = sets['Installations']
    T1 = sets['Time periods 1']
    T2 = sets['Time periods 2']
    F_s = sets['Fuel types']
    R_s = sets['Routes']
    T_max = [6]
    O = sets['Scenarios']
    # T0 = {0,0}

    # Parameters
    retro_cost = params['Cost of retrofitting']                   # C^R
    aquiring_cost = params['Cost of newbuilding']             # C^N
    selling_revenue = params['Revenue']         # R^S
    fuel_cost1 = params['Cost of fuel']                     # C^F_fr
    fuel_cost2W = params['Uncertain cost of fuel']              # C^F_frw
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


    # Variables
    psv1_s_a_t = model.addVars(S, A, T1, vtype=gp.GRB.INTEGER, lb=0, name="xsat1")                                   # x
    retro_psv1_s_s_a_t = model.addVars(S, S2, A, T1, vtype=gp.GRB.INTEGER, lb=0, name="retro_psv1_s_s_a_t")              # y^r
    new_psv1_s_t = model.addVars(S, T1, vtype=gp.GRB.INTEGER, lb=0, name="new_psv1_s_t")                            # y^N
    scrap_psv1_s_a_t = model.addVars(S, A, T1, vtype=gp.GRB.INTEGER, lb=0, name="scrap_psv1_s_a_t")                   # y^S
    weekly_routes1_s_a_f_r_t = model.addVars(S, A, R_s, T1, vtype=gp.GRB.INTEGER, name="weekly_routes1_s_a_f_r_t")            #z^T

    # stage 2 variables
    psv2_s_a_t_w = model.addVars(S, A, T2, O, vtype=gp.GRB.INTEGER, lb=0, name="xsat2")                                   # x
    retro_psv2_s_s_a_t_w = model.addVars(S, S2, A, T2, O, vtype=gp.GRB.INTEGER, lb=0, name="retro_psv2_s_s_a_t_w")              # y^r
    new_psv2_s_t_w = model.addVars(S, T2, O, vtype=gp.GRB.INTEGER, lb=0, name="new_psv2_s_t_w")                            # y^N
    scrap_psv2_s_a_t_w = model.addVars(S, A, T2, O, vtype=gp.GRB.INTEGER, lb=0, name="scrap_psv2_s_a_t_w")                   # y^S
    weekly_routes2_s_a_f_r_t_w = model.addVars(S, A, R_s, T2, O, vtype=gp.GRB.INTEGER, name="weekly_routes2_s_a_f_r_t_w")            #z^T



    ############ Objective ###############

    total_cost_per_t_s1 = {
        t: gp.quicksum(
            retro_cost[s, s1, t] * retro_psv1_s_s_a_t[s, s1, a, t]/eac**(t-1) for s in S for s1 in S for a in A
        ) + gp.quicksum(
            aquiring_cost[s, t] * new_psv1_s_t[s, t]/eac**(t-1) for s in S
        ) - gp.quicksum(
            selling_revenue[s, a, t] * scrap_psv1_s_a_t[s, a, t]/eac**(t-1) for s in S for a in A
        ) + gp.quicksum(
            (fuel_cost1[f, r] * 0.4 if t == 1 else fuel_cost1[f, r])/eac**(t-1) * weekly_routes1_s_a_f_r_t[s,a,r,t] * week_to_t
            for s in S for a in A for f in F_s for r in R_s
        ) for t in T1
    }

    total_cost_per_t_s2 = {
            (t, w): probability * (gp.quicksum(
                retro_cost[s, s1, t] * retro_psv2_s_s_a_t_w[s, s1, a, t, w]/eac**(t-1) for s in S for s1 in S for a in A
            ) + gp.quicksum(
                aquiring_cost[s, t] * new_psv2_s_t_w[s, t, w]/eac**(t-1) for s in S
            ) - gp.quicksum(
                selling_revenue[s, a, t] * scrap_psv2_s_a_t_w[s, a, t, w]/eac**(t-1) for s in S for a in A
            ) + gp.quicksum(
                (fuel_cost2[f, r, w] * 0.4 if t == 1 else fuel_cost2[f, r, w])/eac**(t-1) * weekly_routes2_s_a_f_r_t_w[s,a,r,t, w] * week_to_t
                for s in S for a in A for f in F_s for r in R_s )
            ) for t in T2 for w in O
        }

    total_cost = gp.quicksum(total_cost_per_t_s1[t] for t in T1) + gp.quicksum(total_cost_per_t_s2[t,w] for t in T2 for w in O)
    model.setObjective(total_cost, sense=gp.GRB.MINIMIZE)

    
    #####################
    # STAGE 1
    #####################
    # Fuel - system Compatibility constraint

    for t in T1:
        for r in R_s:
            for a in A: 
                for s in S: 
                    for f in F_s:
                        model.addConstr(
                            weekly_routes1_s_a_f_r_t[s, a, f, r, t] <= compatibility_fs[f, s] * 100,
                            name=f'compatibility_const_s1_{f}_{s}'
                        )
    model.update()
                   
    # Route - system compatibility constraint
    for t in T1: 
        for a in A: 
            for s in S: 
                for r in R_s:
                    model.addConstr(weekly_routes1_s_a_f_r_t[s, a, r, t]  <= compatibility_rs[r, s] * 60,
                                    name=f'route_system_comp_constr_s1_{s}_{a}_{r}_{t}')
    
    # Constraint 5.2 (initial_fleet_balance)
    # Bruker t=1 her siden settet T har verdiene 1 og 2. Må derfor legge til 0 i settet T for å kunne sette t=0 her. Vi må finne en løsning på å få denne indekseringen helt riktig
    for s in S:
        t=1
        for a in A:    
            retrofitted_psvs_from = gp.LinExpr()
            retrofitted_psvs_to = gp.LinExpr()
            for s1 in S:
                retrofitted_psvs_from.add(retro_psv1_s_s_a_t[s, s1, a, t])  # Sum of psv retrofitted from type s
                retrofitted_psvs_to.add(retro_psv1_s_s_a_t[s1, s, a, t])  # Sum of psv retrofitted to type s
            model.addConstr(
                initial_fleet_s_a[s, a] - scrap_psv1_s_a_t[s, a, t] - retrofitted_psvs_from + retrofitted_psvs_to == psv1_s_a_t[s, a, t],
                name=f'initFleetBal_const_s1_{s}_{a}'
            )
    model.update()

    # Kan stå her uten å være med i rapporten
    # Trenger vi den egentlig??
    for s in S:
        for t in T1:
            model.addConstr(
                new_psv1_s_t[s,t] <= 18,
                name = f'psv_buy_s1_{s}_{t}'
            )
    model.update()

    #model.addConstr(new_psv_s_t[1,2]==3)

    for s in S:
        for t in sorted(T1)[1:]:
            a = 0
            model.addConstr(
                scrap_psv1_s_a_t[s, a, t] <= new_psv1_s_t[s,t-1],
                name = f'psv_sell_s1_{s}_{t}'
            )
    model.update()

    # Constraint 5.3 (init_retrosale_fleet_balance)
    # Bruker t=1 her siden settet T har verdiene 1 og 2. Må derfor legge til 0 i settet T for å kunne sette t=0 her. Vi må finne en løsning på å få denne indekseringen helt riktig
    # sjekke retrofits, skraping og endr
    for s in S:
        for a in A:
            t = 1
            retrofitted_psvs_from = gp.LinExpr()
            for s2 in S:
                retrofitted_psvs_from.add(retro_psv1_s_s_a_t[s, s2, a, t])  # Sum of psv retrofitted from type s
                # print('scrap', retrofitted_psvs_from)
            model.addConstr(
                retrofitted_psvs_from + scrap_psv1_s_a_t[s, a, t] <= initial_fleet_s_a[s, a],
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
                for s2 in S:
                    retrofitted_psvs_from.add(retro_psv1_s_s_a_t[s, s2, a, t])  # Sum of psv retrofitted from type s
                    retrofitted_psvs_to.add(retro_psv1_s_s_a_t[s2, s, a, t])  # Sum of psv retrofitted to type s
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
            for s2 in S:
                retrofitted_psvs_from.add(retro_psv1_s_s_a_t[s, s2, a, t])  # Sum of psv retrofitted from type s
                retrofitted_psvs_to.add(retro_psv1_s_s_a_t[s2, s, a, t])  # Sum of psv retrofitted to type s
            model.addConstr(
                new_psv1_s_t[s,t-1] - scrap_psv1_s_a_t[s, a, t] - retrofitted_psvs_from + retrofitted_psvs_to == psv1_s_a_t[s, a, t], 
                name=f'psv_age_bal_const2_s1_{s}_{a}_{t}'
            )

    # Constraint 5.5 (retrosale_fleet_balance)
    # fiks
    # forklar hvorfor det ikke er a-1 og t-1
    for s in S:
        for a in sorted(A)[1:]:
            for t in sorted(T1)[1:]:
                retrofitted_psvs_from = gp.LinExpr()
                for s2 in S:
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
            model.addConstr(
                new_psv1_s_t[s,t-1] == psv1_s_a_t[s, a, t],
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
                name = f'psv_scrapped_period_balance_const_s1_{s}_{t}'
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
                        if f == 1 or f==2:
                            total_emissions.add(emissions[f, r] * weekly_routes1_s_a_f_r_t[s, a, f, r, t] * week_to_t)
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
                        number_of_visits.add(weekly_routes1_s_a_f_r_t[s,a,r,t] * inst_in_route[i,r])
            model.addConstr(
                number_of_visits >= demand[i,t], 
                name=f'demand_installation_constraint_s1_{s}_{a}_{i}_{t}'
            )
    
    # Constraint 5.10
    for s in S:
        for a in A:
            for t in T1:
                time_used_psv = gp.LinExpr()
                for r in R_s:
                    time_used_psv.add(time_used_rs[r, s] * weekly_routes1_s_a_f_r_t[s, a, r, t])
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
                t=2
                retrofitted_psvs_from = gp.LinExpr()
                retrofitted_psvs_to = gp.LinExpr()
                for s2 in S:
                    retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                    retrofitted_psvs_to.add(retro_psv2_s_s_a_t_w[s2, s, a, t, w])  # Sum of psv retrofitted to type s
                model.addConstr(
                    psv1_s_a_t[s, a-1, t-1] - scrap_psv2_s_a_t_w[s, a, t, w] - retrofitted_psvs_from + retrofitted_psvs_to == psv2_s_a_t_w[s, a, t, w],
                    name=f'psv_age_bal_const_na_{s}_{a}_{t}'
                )
        
    #5.14
    for w in O:       
        for s in S:
            t = 2
            a = 0
            retrofitted_psvs_from = gp.LinExpr()
            retrofitted_psvs_to = gp.LinExpr()
            for s2 in S:
                retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                retrofitted_psvs_to.add(retro_psv2_s_s_a_t_w[s2, s, a, t, w])  # Sum of psv retrofitted to type s
            model.addConstr(
                new_psv1_s_t[s,t-1] - scrap_psv2_s_a_t_w[s, a, t, w] - retrofitted_psvs_from + retrofitted_psvs_to == psv2_s_a_t_w[s, a, t, w], 
                name=f'psv_age_bal_const2_na_{s}_{a}_{t}'
            )

    # Constraint 5.15
    for s in S:
        for a in sorted(A)[1:]:
            for w in O:
                t = 2
                retrofitted_psvs_from = gp.LinExpr()
                for s2 in S:
                    retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                model.addConstr(
                    retrofitted_psvs_from + scrap_psv2_s_a_t_w[s, a, t, w] <= psv1_s_a_t[s, a-1, t-1],
                    name=f'retrosale_fleet_bal_const_na_{s}_{a}_{t}_{w}'
                )
                
    # Constraint 5.16
    for s in S:
        for w in O:
            t=2
            a = 0
            model.addConstr(
                psv2_s_a_t_w[s, a, t, w] == new_psv1_s_t[s,t-1] ,
                name = f'psv_period_balance_const_na_{s}_{t}_{w}'
            )
    

    # Constraint 5.17
    for w in O:
        for s in S:
            t =2
            a = scrapping_age
            model.addConstr(
                scrap_psv2_s_a_t_w[s, a, t, w] == psv1_s_a_t[s, a-1, t-1],
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
                            model.addConstr(
                                weekly_routes2_s_a_f_r_t_w[s, a, f, r, t, w] <= compatibility_fs[f, s] * 100,
                                name=f'compatibility_const_s2_{f}_{s}_{w}'
                            )
    model.update()
                   
    # Route - system compatibility constraint
    for t in T2: 
        for w in O:
            for a in A: 
                for s in S: 
                    for r in R_s:
                        model.addConstr(weekly_routes2_s_a_f_r_t_w[s, a, r, t, w]  <= compatibility_rs[r, s] * 60,
                                        name=f'route_system_comp_constr_s2_{s}_{a}_{r}_{t}_{w}')
        
   

    # Kan stå her uten å være med i rapporten
    for s in S:
        for w in O:
            for t in T2:
                model.addConstr(
                    new_psv2_s_t_w[s,t, w] <= 18,
                    name = f'psv_buy_s2_{s}_{t}_{w}'
                )
    model.update()

    #model.addConstr(new_psv_s_t[1,2]==3)
    for w in O:
        for s in S:
            for t in sorted(T2)[1:]:
                a = 0
                model.addConstr(
                    scrap_psv2_s_a_t_w[s, a, t, w] <= new_psv2_s_t_w[s,t-1, w],
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
                        retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                        retrofitted_psvs_to.add(retro_psv2_s_s_a_t_w[s2, s, a, t, w])  # Sum of psv retrofitted to type s
                    model.addConstr(
                        psv2_s_a_t_w[s, a-1, t-1, w] - scrap_psv2_s_a_t_w[s, a, t, w] - retrofitted_psvs_from + retrofitted_psvs_to == psv2_s_a_t_w[s, a, t, w],
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
                    retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                    retrofitted_psvs_to.add(retro_psv2_s_s_a_t_w[s2, s, a, t, w])  # Sum of psv retrofitted to type s
                model.addConstr(
                    new_psv2_s_t_w[s,t-1, w] - scrap_psv2_s_a_t_w[s, a, t, w] - retrofitted_psvs_from + retrofitted_psvs_to == psv2_s_a_t_w[s, a, t, w], 
                    name=f'psv_age_bal_const2_s2_{s}_{a}_{t}_{w}'
                )

    # Constraint 5.20 (retrosale_fleet_balance)
    
    for w in O:            
        for s in S:
            for a in sorted(A)[1:]:
                for t in sorted(T2)[1:]:
                    retrofitted_psvs_from = gp.LinExpr()
                    for s2 in S:
                        retrofitted_psvs_from.add(retro_psv2_s_s_a_t_w[s, s2, a, t, w])  # Sum of psv retrofitted from type s
                    model.addConstr(
                        retrofitted_psvs_from + scrap_psv2_s_a_t_w[s, a, t, w] <= psv2_s_a_t_w[s, a-1, t-1, w],
                        name=f'retrosale_fleet_bal_const_s2_{s}_{a}_{t}_{w}'
                    )
                    
    # Constraint 5.21 New builds constraint
    for w in O:                 
        for s in S:
            for t in sorted(T2)[1:]:
                # print('t', t)
                a = 0
                model.addConstr(
                    new_psv2_s_t_w[s,t-1, w] == psv2_s_a_t_w[s, a, t, w],
                    name = f'psv_period_balance_const_s2_{s}_{t}_{w}'
                )
    
   

    # Constraint 5.22
    for w in O: 
        for s in S:
            for t in sorted(T2)[1:]:
                a = scrapping_age
                model.addConstr(
                    scrap_psv2_s_a_t_w[s, a, t, w] == psv2_s_a_t_w[s, a-1, t-1, w],
                    name = f'psv_scrapped_period_balance_const_s2_{s}_{t}_{w}'
                )

    
    # Constraint 5.23
    for w in O:             
        for t in T2:
            total_emissions = gp.LinExpr()
            for s in S:
                for a in A:
                    for f in F_s:
                        for r in R_s:
                            if f == 1 or f==2:
                                total_emissions.add(emissions[f, r] * weekly_routes2_s_a_f_r_t_w[s, a, f, r, t, w] * week_to_t)
                            else:
                                total_emissions.add(emissions[f, r] * weekly_routes2_s_a_f_r_t_w[s, a, f, r, t, w] * week_to_t/ emission_factor**t)
            model.addConstr(
                total_emissions <= max_em_period[t],
                name= f'max_emissions_period_constraint_s2_{w}'
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
                            number_of_visits.add(weekly_routes2_s_a_f_r_t_w[s,a,r,t, w] * inst_in_route[i,r])
                model.addConstr(
                    number_of_visits >= demand[i,t], 
                    name=f'demand_installation_constraint_s2_{s}_{a}_{i}_{t}_{w}'
                )
    
    # Constraint 5.25
    for w in O:  
        for s in S:
            for a in A:
                for t in T2:
                    time_used_psv = gp.LinExpr()
                    for r in R_s:
                        time_used_psv.add(time_used_rs[r, s] * weekly_routes2_s_a_f_r_t_w[s, a, r, t, w])
                    model.addConstr(
                        time_used_psv <= max_time * psv2_s_a_t_w[s,a,t, w], 
                        name=f'psv_weekly_time_limit_constr_s2_{s}_{a}_{t}_{w}'
                    )
        

    model.update()
    
    return model, T1, T2, total_cost_per_t_s1, total_cost_per_t_s2
