from scipy.interpolate import interp1d

import numpy as np

def inteporlate_state(state, reftime, kind):
    value_t = state[:,0]
    values = state[:, 1:]  # shape: (N, 3)
    # Interpolate each column to new_times
    interpolated_values = np.stack([
        interp1d(value_t, values[:, i], kind=kind, bounds_error=False, fill_value='extrapolate')(reftime)
        for i in range(values.shape[1])
    ], axis=1)
    return interpolated_values



def inteporlate_thruster(thruster_t, thruster_data, reftime, kind):
    # Interpolate each column
    interpolated_values = np.stack([
        interp1d(thruster_t[:,i], thruster_data[:, i], kind=kind, bounds_error=False, fill_value='extrapolate')(reftime)
        for i in range(thruster_data.shape[1])
    ], axis=1)

    return interpolated_values


