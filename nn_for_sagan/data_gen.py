import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import torch

import sys
sys.path.append('../../wuchengzhou')
import sagan

wave_dict = sagan.utils.line_wave_dict
label_dict = sagan.utils.line_label_dict

uniform = np.random.uniform
normal = np.random.normal

def pnormal(mean, stddev):
        while True:
            value = normal(mean, stddev)
            if value >= 0:  # 确保值不为负
                return value

def generate_continuum(wave):
    # Generate random parameters for the power law
    amp1 = 10 * np.random.rand()
    amp2 = np.random.rand()
    alpha = uniform(0, 2)
    stddev = uniform(500, 2500)
    z = uniform(0, 0.01)
    
    # Create the model
    pl_amps = models.PowerLaw1D(amplitude=amp1, x_0=5500, alpha=alpha, fixed={'x_0': True})
    iron = sagan.IronTemplate(amplitude=amp2, stddev=stddev, z=z, name='Fe II')
    model = pl_amps + iron
    flux = model(wave)
    
    # Add noise
    noise = np.random.normal(0, 0.1, wave.size)
    flux += noise
    
    return flux

# narrow Line with 2 components
def generate_spec(wave):

    amp_c0 = pnormal(1, 0.5)
    sigma_c = pnormal(500, 200)
    amp_w0 = uniform(0.1, 0.5)
    dv_w0 = normal(0, 400)
    sigma_w0 = pnormal(1700, 400)
    line_o3 = sagan.Line_MultiGauss_doublet(n_components=2, amp_c0=amp_c0, amp_c1=0.2, dv_c=normal(0, 75), sigma_c=sigma_c, wavec0=wave_dict['OIII_5007'], wavec1=wave_dict['OIII_4959'], name='[O III]', amp_w0=amp_w0, dv_w0=dv_w0, sigma_w0=sigma_w0)
    
    def tie_o3(model):
        return model['[O III]'].amp_c0 / 2.98
    line_o3.amp_c1.tied = tie_o3
    
    n_ha = sagan.Line_MultiGauss(n_components=1, amp_c=pnormal(0.1, 0.05), wavec=wave_dict['Halpha'], name=f'narrow {label_dict["Halpha"]}')
    n_hb = sagan.Line_MultiGauss(n_components=1, amp_c=pnormal(0.1, 0.05), wavec=wave_dict['Hbeta'], name=f'narrow {label_dict["Hbeta"]}')
    n_hg = sagan.Line_MultiGauss(n_components=1, amp_c=pnormal(0.1, 0.05), wavec=wave_dict['Hgamma'], name=f'narrow {label_dict["Hgamma"]}')
    
    b_HeI = sagan.Line_MultiGauss(n_components=1, amp_c=pnormal(0.1, 0.08), dv_c=normal(0, 75), sigma_c=uniform(1400, 1800), wavec=5875.624, name=f'He I 5876')
    
    b_ha = sagan.Line_MultiGauss(n_components=2, amp_c=uniform(1.5, 2.5), dv_c=normal(0, 75), sigma_c=uniform(1200, 1600), wavec=wave_dict['Halpha'], name=label_dict['Halpha'], amp_w0=uniform(0.05, 0.6), sigma_w0=pnormal(5000, 400), dv_w0=normal(0, 400))
    b_hb = sagan.Line_MultiGauss(n_components=1, amp_c=uniform(0.7, 1.7),  dv_c=normal(0, 75), sigma_c=pnormal(1500, 200), wavec=wave_dict['Hbeta'], name=label_dict['Hbeta'])
    b_hg = sagan.Line_MultiGauss(n_components=1, amp_c=uniform(0.4, 0.9), dv_c=normal(0, 75), sigma_c=pnormal(1500, 200), wavec=wave_dict['Hgamma'], name=label_dict['Hgamma'])
    
    def tie_narrow_sigma_c(model):
        return model['[O III]'].sigma_c

    def tie_narrow_dv_c(model):
        return model['[O III]'].dv_c

    for line in [n_ha, n_hb, n_hg]:
        line.sigma_c.tied = tie_narrow_sigma_c
        line.dv_c.tied = tie_narrow_dv_c
    
    line_ha = b_ha + n_ha
    line_hb = b_hb + n_hb
    line_hg = b_hg + n_hg

    # def model
    model = (line_ha + line_hb + line_hg + line_o3 + b_HeI)
    
    flux = model(wave)
    
    return flux

wave = np.linspace(4150, 7000)
flux = generate_spec(wave)
plt.plot(wave, flux)
plt.show()