import numpy as np
import pylab as plt
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap

def main():
    control_dataset   = nc.Dataset('data/climas/xjksc/xjksc_ANN_mean_allvars.nc')
    co2_2x_dataset = nc.Dataset('data/climas/co2_2x/co2_2x_ANN_mean_allvars.nc')
    one_pct_2x_dataset = nc.Dataset('data/climas/xjksd/xjksd_ANN_mean_allvars.nc')

    lons = control_dataset.variables['longitude'][:]
    lats = control_dataset.variables['latitude'][:]
    aavg_weight_edge = control_dataset.variables['aavg_weight_edge'][:, :]

    control = control_dataset.variables['surf_temp'][0, 0]
    co2_2x = co2_2x_dataset.variables['surf_temp'][0, 0]
    one_pct_2x = one_pct_2x_dataset.variables['surf_temp'][0, 0]

    co2_2x_diff = co2_2x - control
    one_pct_2x_diff = one_pct_2x - control

    lons, lats = np.meshgrid(lons, lats)

    # min/maxes
    print('co2-ctrl min max:', co2_2x_diff.min(), co2_2x_diff.max())
    print('1%-ctrl min max:', one_pct_2x_diff.min(), one_pct_2x_diff.max())

    if False:
        m = Basemap(projection='mill', lon_0=180)
        x, y = m(lons, lats)
        m.pcolormesh(x, y, co2_2x_diff, vmin=-4, vmax=12)
        m.drawcoastlines()
        plt.show()

        m.pcolormesh(x, y, one_pct_2x_diff, vmin=-4, vmax=12)
        m.drawcoastlines()
        plt.show()

    # weighted avg. Note how calc is done.
    tcr = (one_pct_2x_diff * aavg_weight_edge).sum() / aavg_weight_edge.sum()
    print('co2-ctrl global mean:', (co2_2x_diff * aavg_weight_edge).sum() / aavg_weight_edge.sum())
    print('1%-ctrl global mean (TCR):', tcr)

    control_toa_net_flux = control_dataset.variables['toa_swdown'][0, 0] -\
                           control_dataset.variables['toa_swup'][0, 0] -\
                           control_dataset.variables['olr'][0, 0]

    co2_2x_toa_net_flux = co2_2x_dataset.variables['toa_swdown'][0, 0] -\
                          co2_2x_dataset.variables['toa_swup'][0, 0] -\
                          co2_2x_dataset.variables['olr'][0, 0]

    one_pct_2x_toa_net_flux = one_pct_2x_dataset.variables['toa_swdown'][0, 0] -\
                          one_pct_2x_dataset.variables['toa_swup'][0, 0] -\
                          one_pct_2x_dataset.variables['olr'][0, 0]

    toa_net_flux_diff = co2_2x_toa_net_flux - control_toa_net_flux

    ctrl_toa_gm = (control_toa_net_flux * aavg_weight_edge).sum() / aavg_weight_edge.sum()
    co2_2x_toa_gm = (co2_2x_toa_net_flux * aavg_weight_edge).sum() / aavg_weight_edge.sum()
    one_pct_2x_toa_gm = (one_pct_2x_toa_net_flux * aavg_weight_edge).sum() / aavg_weight_edge.sum()

    print('ctrl TOA global mean:', ctrl_toa_gm)
    print('co2_2x TOA global mean:', co2_2x_toa_gm)
    print('one_pct_2x TOA global mean:', one_pct_2x_toa_gm)

    alpha = (co2_2x_toa_gm - one_pct_2x_toa_gm) / tcr
    clim_sensitivity_co2 = co2_2x_toa_gm / alpha
    print('alpha, clim sens (CO2):', alpha, clim_sensitivity_co2)

    return co2_2x_diff, one_pct_2x_diff



if __name__ == "__main__":
    main()
