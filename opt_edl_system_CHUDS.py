#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEEN 357 – Phase 4 Optimization Script
"""
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from subfunctions_Phase4 import *
from define_experiment import *
import pickle
import sys


TEAM_NAME   = 'CHUDS'   
TEAM_NUMBER = 47             


MOTOR_CHOICES   = ['torque_he', 'speed_he']
BATTERY_CHOICES = ['NiMH', 'LiFePO4']
CHASSIS_CHOICE  = 'steel'     
                               
N_BATT_MODULES  = 10


N_STARTS  = 8   
MAX_ITER  = 200   
tmax      = 5000  


max_rover_velocity = -1.0     
min_strength       = 40000    
max_cost           = 7.2e6   


X_LB = np.array([14.0,  0.55, 250.0, 0.05, 100.0])
X_UB = np.array([19.0,  0.70, 800.0, 0.12, 290.0])
bounds_scipy = Bounds(X_LB, X_UB)

def build_edl(motor_type, batt_type, n_modules):
    planet        = define_planet()
    edl_system    = define_edl_system()
    mission_events = define_mission_events()

    edl_system = define_chassis(edl_system, CHASSIS_CHOICE)
    edl_system = define_motor(edl_system, motor_type)
    edl_system = define_batt_pack(edl_system, batt_type, n_modules)

    # initial EDL state
    edl_system['altitude']              = 11000   # [m]
    edl_system['velocity']              = -587    # [m/s]
    edl_system['parachute']['deployed'] = True
    edl_system['parachute']['ejected']  = False
    edl_system['rover']['on_ground']    = False

    return planet, edl_system, mission_events


def make_objective(edl_system, planet, mission_events, experiment, end_event):
    def obj(x):
        return obj_fun_time(x, edl_system, planet, mission_events,
                            tmax, experiment, end_event)
    return obj

def make_constraints(edl_system, planet, mission_events, experiment,
                     end_event, max_batt_e_per_m):
    def cons(x):
        return constraints_edl_system(x, edl_system, planet, mission_events,
                                      tmax, experiment, end_event,
                                      min_strength, max_rover_velocity,
                                      max_cost, max_batt_e_per_m)
    return cons


def optimize_combo(motor_type, batt_type, n_modules, experiment, end_event,
                   verbose=True):
    planet, edl_system, mission_events = build_edl(motor_type, batt_type, n_modules)

    max_batt_e_per_m = (edl_system['rover']['power_subsys']['battery']['capacity']
                        / 1000.0)

    obj_f  = make_objective(edl_system, planet, mission_events, experiment, end_event)
    cons_f = make_constraints(edl_system, planet, mission_events, experiment,
                              end_event, max_batt_e_per_m)

    ineq_cons = {
        'type': 'ineq',
        'fun' : lambda x: -1.0 * cons_f(x)
    }

    options = {'maxiter': MAX_ITER, 'disp': False, 'ftol': 1e-9}

    best_x   = None
    best_f   = np.inf
    rng      = np.random.default_rng(42)


    mid = 0.5 * (X_LB + X_UB)
    x0_list = [mid,
               np.array([17.0, 0.65, 450.0, 0.09, 200.0]),  # hand-tuned guess
               np.array([16.0, 0.60, 350.0, 0.07, 180.0])]
    for _ in range(N_STARTS - len(x0_list)):
        x0_list.append(X_LB + rng.random(5) * (X_UB - X_LB))

    for i, x0 in enumerate(x0_list):
        try:
            res = minimize(obj_f, x0, method='SLSQP',
                           constraints=ineq_cons,
                           bounds=bounds_scipy,
                           options=options)

            c = cons_f(res.x)
            feasible = np.max(c) <= 0

            if verbose:
                status = "FEASIBLE" if feasible else "infeasible"
                print(f"    [{motor_type}/{batt_type}] start {i+1:2d}: "
                      f"f={res.fun:10.2f}  {status}")

            if feasible and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        except Exception as e:
            if verbose:
                print(f"    [{motor_type}/{batt_type}] start {i+1:2d}: ERROR – {e}")

    return best_x, best_f, edl_system, planet, mission_events, max_batt_e_per_m


def main():
    experiment, end_event = experiment1()

    global_best_x       = None
    global_best_f       = np.inf
    global_best_motor   = None
    global_best_battery = None
    global_best_edl     = None
    global_best_planet  = None
    global_best_events  = None
    global_best_batt_em = None

    print("=" * 65)
    print("MEEN 357 Phase 4 – Discrete Sweep + Multi-Start SLSQP")
    print("=" * 65)

    for motor in MOTOR_CHOICES:
        for batt in BATTERY_CHOICES:
            print(f"\n  Motor={motor:12s}  Battery={batt}")
            x, f, edl, planet, events, batt_em = optimize_combo(
                motor, batt, N_BATT_MODULES, experiment, end_event)

            if x is not None and f < global_best_f:
                global_best_f       = f
                global_best_x       = x
                global_best_motor   = motor
                global_best_battery = batt
                global_best_edl     = edl
                global_best_planet  = planet
                global_best_events  = events
                global_best_batt_em = batt_em
                print(f"  *** NEW GLOBAL BEST: f = {f:.4f} ***")


    if global_best_x is None:
        print("\nNo feasible solution found across all discrete combos.")
        print("Try increasing N_STARTS, relaxing bounds, or increasing N_BATT_MODULES.")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("BEST DISCRETE SELECTION")
    print(f"  Motor:        {global_best_motor}")
    print(f"  Battery:      {global_best_battery}")
    print(f"  Chassis:      {CHASSIS_CHOICE}")
    print(f"  Batt modules: {N_BATT_MODULES}")
    print("=" * 65)

    # ── rerun best design for reporting ────────────────────────────────────
    xbest = global_best_x
    edl_system     = redefine_edl_system(global_best_edl)
    planet         = global_best_planet
    mission_events = global_best_events

    edl_system['parachute']['diameter']                               = xbest[0]
    edl_system['rover']['wheel_assembly']['wheel']['radius']          = xbest[1]
    edl_system['rover']['chassis']['mass']                            = xbest[2]
    edl_system['rover']['wheel_assembly']['speed_reducer']['diam_gear'] = xbest[3]
    edl_system['rocket']['initial_fuel_mass']                         = xbest[4]
    edl_system['rocket']['fuel_mass']                                 = xbest[4]

    edl_system['team_name']   = TEAM_NAME
    edl_system['team_number'] = TEAM_NUMBER

    # verify constraints one more time
    cons_f = make_constraints(edl_system, planet, mission_events,
                              experiment, end_event, global_best_batt_em)
    c = cons_f(xbest)
    feasible = np.max(c) <= 0
    if not feasible:
        print("\nWARNING: re-evaluated solution is NOT feasible. Check manually.")

    fname = f'FA25_SecYY_Team{TEAM_NUMBER:02d}.pickle'
    with open(fname, 'wb') as handle:
        pickle.dump(edl_system, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nPickle saved → {fname}")

 
    time_edl_run, _, edl_system = simulate_edl(edl_system, planet, mission_events,
                                               tmax, True)
    time_edl  = time_edl_run[-1]
    edl_system['rover'] = simulate_rover(edl_system['rover'], planet,
                                         experiment, end_event)
    time_rover  = edl_system['rover']['telemetry']['completion_time']
    total_time  = time_edl + time_rover
    total_cost  = get_cost_edl(edl_system)

    print("\n" + "=" * 65)
    print("REQUIRED REPORT VALUES")
    print("=" * 65)
    print(f"Optimized parachute diameter     = {xbest[0]:.6f} [m]")
    print(f"Optimized rocket fuel mass       = {xbest[4]:.6f} [kg]")
    print(f"Time to complete EDL mission     = {time_edl:.6f} [s]")
    print(f"Rover velocity at landing        = {edl_system['rover_touchdown_speed']:.6f} [m/s]")
    print(f"Optimized wheel radius           = {xbest[1]:.6f} [m]")
    print(f"Optimized d2 (gear diameter)     = {xbest[3]:.6f} [m]")
    print(f"Optimized chassis mass           = {xbest[2]:.6f} [kg]")
    print(f"Time to complete rover mission   = {time_rover:.6f} [s]")
    print(f"Time to complete mission         = {total_time:.6f} [s]")
    print(f"Average velocity                 = {edl_system['rover']['telemetry']['average_velocity']:.6f} [m/s]")
    print(f"Distance traveled                = {edl_system['rover']['telemetry']['distance_traveled']:.6f} [m]")
    print(f"Battery energy per meter         = {edl_system['rover']['telemetry']['energy_per_distance']:.6f} [J/m]")
    print(f"Total cost                       = {total_cost:.6f} [$]")
    print("=" * 65)
    print(f"Motor type:           {global_best_motor}")
    print(f"Battery type:         {global_best_battery}")
    print(f"Number of modules:    {N_BATT_MODULES}")
    print(f"Chassis material:     {CHASSIS_CHOICE}")
    print("=" * 65)

    return xbest, edl_system

if __name__ == '__main__':
    main()
