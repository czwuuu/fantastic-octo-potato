import numpy as np

data_path = 'data_generated/wave_flux_data.npz'
data = np.load(data_path)

ind = 0
wave_flux = data['wave_flux_pairs'][ind]

wave = wave_flux[0, :]
flux = wave_flux[1, :]

