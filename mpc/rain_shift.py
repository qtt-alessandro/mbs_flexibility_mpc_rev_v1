# %%
import numpy as np
import cvxpy as cp
import json 
import json 
import pandas as pd 
import polars as pl 
import os 
from lower_mpc import step_lower_mpc, LMPCDataBuffer
from upper_mpc import step_upper_level, UMPCDataBuffer
from tqdm import tqdm 
import time 
from datetime import timedelta
import matplotlib.pyplot as plt 



# %%
#time = pl.read_parquet(os.path.join(data_path_wsl, "sim_data/sim_data_1h_full.par"))
#upper_mpc_data_pan = pl.read_parquet(os.path.join(data_path_wsl, "sim_data/sim_data_1h_full.par"))
#p1= pl.read_parquet("/home/alqua/data/data_vdfs/pump1_power_siso.par").group_by_dynamic(index_column="time", every="1h").agg(pl.col("pump1_power").mean())
#p4= pl.read_parquet("/home/alqua/data/data_vdfs/pump4_power_siso.par").group_by_dynamic(index_column="time", every="1h").agg(pl.col("pump4_power").mean())
#p3= pl.read_parquet("/home/alqua/data/data_vdfs/pump3_power_siso.par").group_by_dynamic(index_column="time", every="1h").agg(pl.col("pump3_power").mean())
#pumps_df = p1.join(p4, on="time").join(p3, on="time")
#baseline_df = (pumps_df.group_by("hours").agg(
#                    pl.col("energy_cons").mean().alias("mean_energy_cons"),
#                    pl.col("energy_cons").std().alias("std_energy_cons")
#                ).sort("hours"))

# %%
data_path_wsl = "/home/alqua/data/pump_station_data/"
data = pl.read_parquet('/home/alqua/papers/mbs_flexibility_mpc_rev_v1/mpc/input_data/aggregated_data_for_mpc.par')

# %%
start_date = pd.to_datetime("2023-02-10 00:00:00+00:00")
end_date = pd.to_datetime("2023-02-28 14:00:00+00:00")

data = data.filter((pl.col("time") >= start_date) & 
                                       (pl.col("time") <= end_date)
                                       ).with_columns(pl.col("CO2Emission")/1000, pl.col("price")/1000)

data = data.with_columns(pl.col("time").dt.day().alias("day_index"))



# %%
new_cols_map = {
    'time': 'time_utc',
    'inflow': 'inflow_kf', 
    'inflow_0.1': 'inflow_q10', 
    'inflow_0.5': 'inflow_q50', 
    'inflow_0.9': 'inflow_q90', 
    'price': 'da_price',
    'CO2Emission': 'co2_progn'}

data = data.select(new_cols_map).rename(new_cols_map)


# %%
with open('lower_mpc_coefficients.json', 'r') as json_file:
    models_coefficients = json.load(json_file)

# %%
def circular_right_shift(sequence):
    return [sequence[-1]] + sequence[:-1]

# %%
lmpc_data = LMPCDataBuffer()
umpc_data = UMPCDataBuffer()
lmpc_data.initialize()

slice_df = data.select(pl.col(["time_utc", "inflow_kf", "co2_progn","da_price", 'inflow_q10', 'inflow_q50','inflow_q90']))
inflow_kf = slice_df["time_utc", "inflow_kf"].upsample(every="1m", time_column= "time_utc").fill_null(strategy="forward")["inflow_kf"]

zs = 3 
N = 60
horizon = 24
step_size = 1
max_start = len(slice_df) - horizon + 1
start_index = 0
trigger = [1, 0, 0]

with tqdm(total=max_start, desc="Hour Steps") as pbar:
    start_index = 0
    while start_index < max_start:
        prices_values = slice_df["da_price"][start_index : start_index + horizon]
        co2_values = slice_df["co2_progn"][start_index : start_index + horizon]
        inflow_values = slice_df["inflow_kf"][start_index : start_index + horizon]
        
        start_time_umpc =  time.time()
        ############################################### Upper Level Optimization
        
        umpc_opt_results = step_upper_level(
            horizon=horizon,
            prices_values=prices_values,
            co2_progn_values=co2_values,
            inflow_values=inflow_values,
            h_init=lmpc_data.data["height_sys"][-1],
            energy_init=lmpc_data.data["p1_power"][-1] + lmpc_data.data["p3_power"][-1] + lmpc_data.data["p4_power"][-1],
            Qout_init=lmpc_data.data["qout"][-1])

        end_time_umpc = time.time() - start_time_umpc
        umpc_opt_results["opt_time_umpc"] = end_time_umpc
        umpc_opt_results["qin_q10"] = slice_df["inflow_q10"][0]
        umpc_opt_results["qin_q50"] = slice_df["inflow_q50"][0]
        umpc_opt_results["qin_q90"] = slice_df["inflow_q90"][0]
        umpc_opt_results["time_utc"] = slice_df["time_utc"][start_index]
        #if start_index == 58:
        #    umpc_opt_results["height_ref"] = 70
        #if start_index >0:
            #print("=========================")
            #print("Height sys:", res_dict["height_sys"])
            #print("Height ref:", umpc_opt_results["height_ref"])
        umpc_data.update(umpc_opt_results)

        ############################################### Lower Level Optimization

        if start_index % 3 == 0 and start_index != 0:
        
            trigger = circular_right_shift(trigger)


        for k in range(0, 60):
            u_stack = np.vstack([lmpc_data.data["u1"][-zs:],
                                lmpc_data.data["u2"][-zs:], 
                                lmpc_data.data["u3"][-zs:]])

            power_stack = np.vstack([lmpc_data.data["p1_power"][-zs:],
                                    lmpc_data.data["p3_power"][-zs:], 
                                    lmpc_data.data["p4_power"][-zs:]])
            
            start_time_lmpc = time.time()

            start_time_lmpc =  time.time()
            res_dict  = step_lower_mpc(Qin_est = inflow_kf[start_index + k : start_index + k + 63], # da cambiare
                                    Qout_meas = lmpc_data.data["qout"][-zs:],
                                    h_meas = lmpc_data.data["height_sys"][-zs:],
                                    w_meas = u_stack,
                                    E_meas = power_stack,
                                    P_meas = lmpc_data.data["pressure_sys"][-zs:],
                                    h_ref = umpc_opt_results["height_ref"],
                                    trigger = trigger,
                                    N = 60,
                                    zs = zs,
                                    models_coefficients = models_coefficients)
            end_time_lmpc = time.time() - start_time_lmpc
            res_dict["opt_time_lmpc"] = end_time_lmpc
            res_dict['time_utc'] = slice_df["time_utc"][start_index] + timedelta(minutes=k)
            res_dict['height_ref'] = umpc_opt_results["height_ref"]
            res_dict['qin'] = inflow_kf[start_index + k : start_index + k + 1][0] # access value form the list           
            lmpc_data.update(res_dict)

        start_index += step_size
        pbar.update(step_size)
        if start_index == 5:
            break
    opt_lmpc = lmpc_data.to_dataframe(save=True, file_path='/output_data/lmpc_rain_shift_results.par')
    opt_umpc = umpc_data.to_dataframe(save=True, file_path='/output_data/umpc_rain_shift_results.par')



