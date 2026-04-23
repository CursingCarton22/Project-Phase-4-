#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from subfunctions_Phase4 import *
from define_experiment import *
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
import pickle
import sys

planet = define_planet()
mission_events = define_mission_events()
tmax = 5000

experiment, end_event = experiment1()

# constraints
max_rover_velocity = -1
min_strength = 40000
max_cost = 7.2e6

# ============================================================
# SWEEP: find best motor, battery, chassis combo
# ============================================================
motors = ['base', 'base_he', 'torque', 'torque_he', 'speed', 'speed_he']
# motors = ['speed', 'speed_he']
batteries = ['LiFePO4', 'NiMH']
# chassis_types = ['steel', 'magnesium', 'carbon']
chassis_types = ['steel']
num_modules = 10  # can be adjusted

print('='*75)
print(f'{"Motor":<12} {"Battery":<12} {"Modules":<8} {"Chassis":<12} {"Time(s)":<12} {"Cost($)":<14} {"Dist(m)"}')
print('='*75)

best_time = np.inf
best_combo = None

for motor in motors:
    for batt in batteries:
        for chassis in chassis_types:
            try:
                edl_system = define_edl_system()
                edl_system = define_motor(edl_system, motor)
                edl_system = define_batt_pack(edl_system, batt, num_modules)
                edl_system = define_chassis(edl_system, chassis)

                edl_system['altitude'] = 11000
                edl_system['velocity'] = -587
                edl_system['parachute']['deployed'] = True
                edl_system['parachute']['ejected'] = False
                edl_system['rover']['on_ground'] = False

                max_batt_energy_per_meter = edl_system['rover']['power_subsys']['battery']['capacity'] / 1000

                # run rover sim only (skip EDL for speed)
                edl_system['rover'] = simulate_rover(edl_system['rover'], planet, experiment, end_event)

                time_rover = edl_system['rover']['telemetry']['completion_time']
                dist = edl_system['rover']['telemetry']['distance_traveled']
                cost = get_cost_edl(edl_system)

                feasible = (dist >= 1000) and (cost <= max_cost)
                flag = '  OK' if feasible else '  FAIL'

                print(f'{motor:<12} {batt:<12} {num_modules:<8} {chassis:<12} {time_rover:<12.2f} {cost:<14.2f} {dist:.2f}{flag}')

                if feasible and time_rover < best_time:
                    best_time = time_rover
                    best_combo = (motor, batt, chassis)

            except Exception as e:
                print(f'{motor:<12} {batt:<12} {num_modules:<8} {chassis:<12} ERROR: {e}')

print('='*75)
if best_combo:
    print(f'Best combo: Motor={best_combo[0]}, Battery={best_combo[1]}, Chassis={best_combo[2]}, Time={best_time:.2f}s')
else:
    print('No feasible combo found - try increasing num_modules')
print('='*75)

# ============================================================
# OPTIMIZER: run with best combo found above
# ============================================================
if best_combo is None:
    raise Exception('No feasible combo found, cannot run optimizer')

edl_system = define_edl_system()
edl_system = define_motor(edl_system, best_combo[0])
edl_system = define_batt_pack(edl_system, best_combo[1], num_modules)
edl_system = define_chassis(edl_system, best_combo[2])

edl_system['altitude'] = 11000
edl_system['velocity'] = -587
edl_system['parachute']['deployed'] = True
edl_system['parachute']['ejected'] = False
edl_system['rover']['on_ground'] = False

max_batt_energy_per_meter = edl_system['rover']['power_subsys']['battery']['capacity'] / 1000

# bounds = Bounds([14, 0.2, 350, 0.05, 100], [19, 0.6, 800, 0.12, 290])
bounds = Bounds(
    [16.5, 0.3, 400, 0.06, 200],
    [18.8, 0.6, 650, 0.10, 290]
)
x0 = np.array([18, 0.5, 600, 0.06, 270])

obj_f = lambda x: obj_fun_time(x, edl_system, planet, mission_events, tmax,
                               experiment, end_event)

cons_f = lambda x: constraints_edl_system(x, edl_system, planet, mission_events,
                                          tmax, experiment, end_event, min_strength,
                                          max_rover_velocity, max_cost, max_batt_energy_per_meter)

nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 0)
ineq_cons = {'type': 'ineq',
             'fun': lambda x: -1 * constraints_edl_system(x, edl_system, planet,
                                                          mission_events, tmax, experiment,
                                                          end_event, min_strength, max_rover_velocity,
                                                          max_cost, max_batt_energy_per_meter)}

# iter_count = [0]
# def callbackTC(x, state):
#     iter_count[0] += 1
#     print(f'  Iteration {iter_count[0]:3d} | Objective: {obj_f(x):.2f} s | Constraint violation: {state.constr_violation:.4f}')
#     return False

best_feasible_x = [None]
best_feasible_f = [np.inf]

iter_count = [0]

def callbackTC(x):
    iter_count[0] += 1

    f = obj_f(x)
    c = cons_f(x)
    violation = np.max(c)

    print(f'  Iter {iter_count[0]:2d} | f={f:.2f} | max(c)={violation:.4f}')

    # ✅ Save best feasible solution found so far
    if violation <= 0 and f < best_feasible_f[0]:
        best_feasible_f[0] = f
        best_feasible_x[0] = x.copy()

    return False

print('\nRunning optimizer...')
options = {'maxiter': 5,
           'verbose': 0,
           'disp': False}
# res = minimize(obj_f, x0, method='trust-constr', constraints=nonlinear_constraint,
#                options=options, bounds=bounds, callback=callbackTC)

# res = minimize(obj_f, x0,
#                method='SLSQP',
#                constraints=ineq_cons,
#                bounds=bounds,
#                options={'maxiter':60, 'ftol': 1e-3, 'disp': True})

res = minimize(obj_f, x0,
               method='SLSQP',
               constraints=ineq_cons,
               bounds=bounds,
               callback=callbackTC,   # 👈 ADD THIS BACK
               options={'maxiter':60, 'ftol': 1e-3, 'disp': True})

c = constraints_edl_system(res.x, edl_system, planet, mission_events, tmax, experiment,
                           end_event, min_strength, max_rover_velocity, max_cost,
                           max_batt_energy_per_meter)

print("Initial constraint check:", cons_f(x0))

feasible = np.max(c - np.zeros(len(c))) <= 0
# if feasible:
#     xbest = res.x
# else:
#     xbest = [99999, 99999, 99999, 99999, 99999]
#     raise Exception('Solution not feasible, exiting code...')

# if best_feasible_x[0] is not None:
#     xbest = best_feasible_x[0]
#     print("\nUsing best FEASIBLE solution found during optimization.")
# else:
#     raise Exception("No feasible solution found during optimization.")
#     sys.exit()

c = cons_f(res.x)

# if np.max(c) <= 0:
#     xbest = res.x
#     print("\n✅ Using optimizer result (feasible)")
# else:
#     raise Exception("No feasible solution found — optimizer failed")

if np.max(c) <= 0:
    xbest = res.x
    print("\n✅ Using optimizer result (feasible)")
elif best_feasible_x[0] is not None:
    xbest = best_feasible_x[0]
    print("\n⚠️ Using best FEASIBLE solution found during optimization")
else:
    print("\n⚠️ No feasible solution found — using initial guess")
    xbest = x0

edl_system = redefine_edl_system(edl_system)
edl_system['parachute']['diameter'] = xbest[0]
edl_system['rover']['wheel_assembly']['wheel']['radius'] = xbest[1]
edl_system['rover']['chassis']['mass'] = xbest[2]
edl_system['rover']['wheel_assembly']['speed_reducer']['diam_gear'] = xbest[3]
edl_system['rocket']['initial_fuel_mass'] = xbest[4]
edl_system['rocket']['fuel_mass'] = xbest[4]

edl_system['team_name'] = 'FunTeamName'
edl_system['team_number'] = 99

with open('FA24_501team99.pickle', 'wb') as handle:
    pickle.dump(edl_system, handle, protocol=pickle.HIGHEST_PROTOCOL)

time_edl_run, _, edl_system = simulate_edl(edl_system, planet, mission_events, tmax, True)
time_edl = time_edl_run[-1]

edl_system['rover'] = simulate_rover(edl_system['rover'], planet, experiment, end_event)
time_rover = edl_system['rover']['telemetry']['completion_time']
total_time = time_edl + time_rover
edl_system_total_cost = get_cost_edl(edl_system)


print('\n' + '='*50)
print('FINAL RESULTS')
print('='*50)
print(f'Motor:                         {best_combo[0]}')
print(f'Battery:                       {best_combo[1]} x{num_modules}')
print(f'Chassis:                       {best_combo[2]}')
print(f'Optimized parachute diameter   = {xbest[0]:.4f} [m]')
print(f'Optimized rocket fuel mass     = {xbest[4]:.4f} [kg]')
print(f'Time to complete EDL mission   = {time_edl:.4f} [s]')
print(f'Rover velocity at landing      = {edl_system["rover_touchdown_speed"]:.4f} [m/s]')
print(f'Optimized wheel radius         = {xbest[1]:.4f} [m]')
print(f'Optimized d2                   = {xbest[3]:.4f} [m]')
print(f'Optimized chassis mass         = {xbest[2]:.4f} [kg]')
print(f'Time to complete rover mission = {time_rover:.4f} [s]')
print(f'Time to complete mission       = {total_time:.4f} [s]')
print(f'Average velocity               = {edl_system["rover"]["telemetry"]["average_velocity"]:.4f} [m/s]')
print(f'Distance traveled              = {edl_system["rover"]["telemetry"]["distance_traveled"]:.4f} [m]')
print(f'Battery energy per meter       = {edl_system["rover"]["telemetry"]["energy_per_distance"]:.4f} [J/m]')
print(f'Total cost                     = {edl_system_total_cost:.4f} [$]')
print('='*50)