import argparse
import numpy as np
import netCDF4 as nc

def main(args):
    data_dir = args.data_dir

    control_dataset   = nc.Dataset('%s/climas/xjksc/xjksc_ANN_mean_allvars.nc'%(data_dir))
    co2_2x_dataset = nc.Dataset('%s/climas/co2_2x/co2_2x_ANN_mean_allvars.nc'%(data_dir))
    one_pct_2x_dataset = nc.Dataset('%s/climas/xjksd/xjksd_ANN_mean_allvars.nc'%(data_dir))

    aavg_weight_edge = control_dataset.variables['aavg_weight_edge'][:, :]

    control = control_dataset.variables['surf_temp'][0, 0]
    co2_2x = co2_2x_dataset.variables['surf_temp'][0, 0]
    one_pct_2x = one_pct_2x_dataset.variables['surf_temp'][0, 0]

    co2_2x_diff = co2_2x - control
    one_pct_2x_diff = one_pct_2x - control

    if args.plot:
        from plotting import plot_on_earth
        plot_on_earth(control_dataset, co2_2x_diff, one_pct_2x_diff)
        #m = Basemap(projection='mill', lon_0=180)
        #x, y = m(lons, lats)
        #m.pcolormesh(x, y, co2_2x_diff, vmin=-4, vmax=12)
        #m.drawcoastlines()
        #plt.show()

        #m.pcolormesh(x, y, one_pct_2x_diff, vmin=-4, vmax=12)
        #m.drawcoastlines()
        #plt.show()

    # weighted avg. Note how calc is done.
    tcr = (one_pct_2x_diff * aavg_weight_edge).sum() / aavg_weight_edge.sum()

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


    alpha = (co2_2x_toa_gm - one_pct_2x_toa_gm) / tcr
    clim_sensitivity_co2 = co2_2x_toa_gm / alpha

    if args.output:
        # min/maxes
        print('co2-ctrl min max:', co2_2x_diff.min(), co2_2x_diff.max())
        print('1%-ctrl min max:', one_pct_2x_diff.min(), one_pct_2x_diff.max())

        # weighted avg. Note how calc is done.
        print('co2-ctrl global mean:', (co2_2x_diff * aavg_weight_edge).sum() / aavg_weight_edge.sum())
        print('1%-ctrl global mean (TCR):', tcr)

        print('ctrl TOA global mean:', ctrl_toa_gm)
        print('co2_2x TOA global mean:', co2_2x_toa_gm)
        print('one_pct_2x TOA global mean:', one_pct_2x_toa_gm)
        print('alpha, clim sens (CO2):', alpha, clim_sensitivity_co2)

    return co2_2x_diff, one_pct_2x_diff



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Climate Modelling Analysis')
    parser.add_argument('-p','--plot', help='Plot figures', action='store_true')
    parser.add_argument('-o','--output', help='Output results', action='store_true')
    parser.add_argument('-d','--data-dir', help='Data directory', default='data')
    args = parser.parse_args()
    main(args)
