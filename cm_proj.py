import calendar
import datetime as dt
import argparse
from glob import glob
import numpy as np
import netCDF4 as nc

DIRS = {#'mine': '../xjksa/datam/netcdf/*.nc',
        'sstonly': '/home/ucfaohx/DATA/xjksb/datam/netcdf/*.nc',
        'sstonly': '/home/ucfaohx/DATA/xjksb/datam/netcdf/*.nc',
	'control': '/home/ucfanea/DATA/xjksc/datam/netcdf/*.nc',
	'1%':      '/home/ucfahub/DATA/xjksd/datam/netcdf/*.nc',
	'2x_co2':  '/data/geospatial_23/geogg134/um_output/double-co2/netcdf/*.nc'}

def get_sorted_filenames(args):
    data_dir = args.data_dir
    m = {'h': 0, 'i': 1, 'j': 2}
    months = dict((v.lower(),k) for k,v in enumerate(calendar.month_abbr))
    filenames = glob(DIRS[args.scenario])

    date_filenames = []
    for fn in filenames:
	fn_date = fn.split('/')[-1].split('.')[1] 
	if fn_date[2] == 'h':
	    continue
	date = dt.datetime(2000 + m[fn_date[2]] * 10 + int(fn_date[3]), months[fn_date[-3:]], 1)
	date_filenames.append((date, fn))
    date_filenames.sort(key=lambda tup: tup[0])
    return [fn for d, fn in date_filenames]

def weighted_average(data, weights):
    return (data * weights).sum() / weights.sum()

def calc_time_series(args, variables):
    fns = get_sorted_filenames(args)
    ts = []
    for fn in fns:
	print(fn)
	line = []
	for variable in variables:
	    ds = nc.Dataset(fn)
	    wa = weighted_average(ds.variables[variable][0, 0], ds.variables['aavg_weight_edge'][:])
	    line.append(wa)
	    ds.close()
	ts.append(line)
    return np.array(ts)

def all_time_series(args):
    data = {}
    for scenario in DIRS.keys():
	args.scenario = scenario
	vars = ['surf_temp', 'toa_swdown', 'toa_swup', 'olr']
	data[scenario] = calc_time_series(args, vars)
    return data


def main(args):
    data_dir = args.data_dir

    # Load NetCDF datasets.
    control_dataset    = nc.Dataset('%s/climas/xjksc/xjksc_ANN_mean_allvars.nc'%(data_dir))
    co2_2x_dataset     = nc.Dataset('%s/climas/co2_2x/co2_2x_ANN_mean_allvars.nc'%(data_dir))
    one_pct_2x_dataset = nc.Dataset('%s/climas/xjksd/xjksd_ANN_mean_allvars.nc'%(data_dir))

    # Store the average weight (edge).
    aavg_weight_edge = control_dataset.variables['aavg_weight_edge'][:, :]

    # Get the control, 2xCO2 and 1%-2xCO2 surface temperatures.
    control_st = control_dataset.variables['surf_temp'][0, 0]
    co2_2x_st = co2_2x_dataset.variables['surf_temp'][0, 0]
    one_pct_2x_st = one_pct_2x_dataset.variables['surf_temp'][0, 0]

    # Work out some differences.
    co2_2x_diff = co2_2x_st - control_st
    one_pct_2x_diff = one_pct_2x_st - control_st

    # Plot on nice overlays. Note this WILL NOT work on UCL computers.
    if args.plot:
        from plotting import plot_on_earth
        plot_on_earth(control_dataset, co2_2x_diff, one_pct_2x_diff)

    # weighted avg. Note how calc is done.
    tcr = (one_pct_2x_diff * aavg_weight_edge).sum() / aavg_weight_edge.sum()

    # Calc Top Of Atm net fluxes (incoming solar - outgoing solar - outgoing longwave).
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

    # Calc some global means using a weighted average (Note calculation).

    ctrl_toa_gm = weighted_average(control_toa_net_flux, aavg_weight_edge)
    co2_2x_toa_gm = weighted_average(co2_2x_toa_net_flux, aavg_weight_edge)
    one_pct_2x_toa_gm = weighted_average(one_pct_2x_toa_net_flux, aavg_weight_edge)

    G_2xCO2 = co2_2x_toa_gm - ctrl_toa_gm
    G_1pct  = one_pct_2x_toa_gm - ctrl_toa_gm

    # Work out alpha (climate feedback param) and climate sensitivity.
    alpha = (G_2xCO2 - G_1pct) / tcr
    clim_sensitivity_co2 = G_2xCO2 / alpha

    if args.output:
        # min/maxes
        print('ctrl-mean: %f'%(weighted_average(co2_2x_st, aavg_weight_edge)))
        print('co2-mean: %f'%(weighted_average(control_st, aavg_weight_edge)))
        print('1%%-mean: %f'%(weighted_average(one_pct_2x_st, aavg_weight_edge)))

        print('co2-ctrl min max: %f, %f'%(co2_2x_diff.min(), co2_2x_diff.max()))
        print('1%%-ctrl min max: %f, %f'%(one_pct_2x_diff.min(), one_pct_2x_diff.max()))

        print('co2-ctrl global mean: %f'%(weighted_average(co2_2x_diff, aavg_weight_edge)))
        print('TCR: 1%%-ctrl global mean: %f'%(tcr))

        print('ctrl TOA global mean: %f'%(ctrl_toa_gm))
        print('co2_2x TOA global mean: %f'%(co2_2x_toa_gm))
        print('one_pct_2x TOA global mean: %f'%(one_pct_2x_toa_gm))
        print('G_2xCO2: %f'%(G_2xCO2))
        print('G_1%%: %f'%(G_1pct))
        print('alpha, clim sens (CO2): %f, %f'%(alpha, clim_sensitivity_co2))

    return control_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Climate Modelling Analysis')
    parser.add_argument('-p','--plot', help='Plot figures', action='store_true')
    parser.add_argument('-o','--output', help='Output results', action='store_true')
    parser.add_argument('-d','--data-dir', help='Data directory', default='data')
    parser.add_argument('-s','--scenario', help='Scenario', default='all')
    args = parser.parse_args()
    main(args)
