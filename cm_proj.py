import calendar
import datetime as dt
import argparse
from glob import glob
import numpy as np
import netCDF4 as nc

SCENARIOS = {#'mine': '../xjksa/datam/netcdf/*.nc',
             #'ssto': '/home/ucfaohx/DATA/xjksb/datam/netcdf/*.nc',
             'ctrl': '/home/ucfanea/DATA/xjksc/datam/netcdf/*.nc',
             '1pct': '/home/ucfahub/DATA/xjksd/datam/netcdf/*.nc',
             '2co2': '/data/geospatial_23/geogg134/um_output/double-co2/netcdf/*.nc'}

def get_sorted_filenames(args):
    data_dir = args.data_dir
    m = {'h': 0, 'i': 1, 'j': 2}
    months = dict((v.lower(),k) for k,v in enumerate(calendar.month_abbr))
    filenames = glob(SCENARIOS[args.scenario])

    date_filenames = []
    for fn in filenames:
	fn_date = fn.split('/')[-1].split('.')[1] 
	# Ignore all files with 'h' in pos 2 (like create_climatology_nc).
	if fn_date[2] == 'h':
	    continue
	date = dt.datetime(2000 + m[fn_date[2]] * 10 + int(fn_date[3]), months[fn_date[-3:]], 1)
	date_filenames.append((date, fn))
    date_filenames.sort(key=lambda tup: tup[0])
    return [fn for d, fn in date_filenames]

def weighted_average(data, weights):
    return (data * weights).sum() / weights.sum()

def load_data(args, vars):
    fns = get_sorted_filenames(args)
    scenario_data = {}
    for var in vars:
	scenario_data[var] = []

    for fn in fns:
	#print(fn)
	ds = nc.Dataset(fn)
	for var in vars:
	    scenario_data[var].append(ds.variables[var][0, 0])
	ds.close()

    for var in vars:
	scenario_data[var] = np.array(scenario_data[var])

    return scenario_data

def load_all_data(args, vars):
    data = {}
    for scenario in SCENARIOS.keys():
	args.scenario = scenario
	#print('Load scenario: %s'%scenario)
	data[scenario] = load_data(args, vars)
    return data

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
    for scenario in SCENARIOS.keys():
	args.scenario = scenario
	vars = ['surf_temp', 'toa_swdown', 'toa_swup', 'olr']
	data[scenario] = calc_time_series(args, vars)
    return data

def regional_comparison_sa(args):
    Pass

def load_all_vars_datasets(args):
    data_dir = args.data_dir
    ds = {}

    # Load NetCDF datasets.
    ds['ctrl'] = nc.Dataset('%s/climas/xjksc/xjksc_ANN_mean_allvars.nc'%(data_dir))
    ds['2co2'] = nc.Dataset('%s/climas/co2_2x/co2_2x_ANN_mean_allvars.nc'%(data_dir))
    ds['1pct'] = nc.Dataset('%s/climas/xjksd/xjksd_ANN_mean_allvars.nc'%(data_dir))

    # Store the average weight (edge).
    ds['avg'] = ds['ctrl'].variables['aavg_weight_edge'][:, :]

    return ds

def primary_analysis(args):
    ds = load_all_vars_datasets(args)

    sts = {}
    st_diffs = {}
    toa_net_flux = {}
    toa_gm = {}
    G = {}

    for scenario in SCENARIOS.keys():
        # Get the control, 2co2 and 1%-2co2 surface temperatures.
        sts[scenario] = ds[scenario].variables['surf_temp'][0, 0]

        # Calc Top Of Atm net fluxes (incoming solar - outgoing solar - outgoing longwave).
        toa_net_flux[scenario] = ds[scenario].variables['toa_swdown'][0, 0] -\
                                 ds[scenario].variables['toa_swup'][0, 0] -\
                                 ds[scenario].variables['olr'][0, 0]

        # Calc some global means using a weighted average (Note calculation).
        toa_gm[scenario] = weighted_average(toa_net_flux[scenario], ds['avg'])

    # Work out some differences.
    st_diffs['2co2'] = sts['2co2'] - sts['ctrl']
    st_diffs['1pct'] = sts['1pct'] - sts['ctrl']

    tcr = weighted_average(st_diffs['1pct'], ds['avg'])

    G['2co2']  = toa_gm['2co2'] - toa_gm['ctrl']
    G['1pct']  = toa_gm['1pct'] - toa_gm['ctrl']

    # Work out alpha (climate feedback param) and climate sensitivity.
    alpha = (G['2co2'] - G['1pct']) / tcr
    clim_sens = G['2co2'] / alpha

    # Plot on nice overlays. Note this WILL NOT work on UCL computers.
    if args.plot:
        from plotting import plot_all
        plot_all(ds['ctrl'], st_diffs['1pct'], toa_net_flux['2co2'])

    res = {'ds': ds,
           'sts': sts,
           'toa_net_flux': toa_net_flux,
           'toa_gm': toa_gm,
           'st_diffs': st_diffs,
           'tcr': tcr,
           'G': G,
           'alpha': alpha,
           'clim_sens': clim_sens}

    if args.output:
        print_res(res)

    return res

def print_res(res):
    ds = res['ds']
    sts = res['sts']
    st_diffs = res['st_diffs']
    tcr = res['tcr']
    toa_gm = res['toa_gm']
    G = res['G']
    alpha = res['alpha']
    clim_sens = res['clim_sens']

    # min/maxes
    print('ctrl-mean: %f'%(weighted_average(sts['ctrl'], ds['avg'])))
    print('ctrl TOA global mean: %f'%(toa_gm['ctrl']))

    print('1%%-mean: %f'%(weighted_average(sts['1pct'], ds['avg'])))
    print('1%%-ctrl min max: %f, %f'%(st_diffs['1pct'].min(), st_diffs['1pct'].max()))

    print('co2-mean: %f'%(weighted_average(sts['2co2'], ds['avg'])))
    print('co2-ctrl min max: %f, %f'%(st_diffs['2co2'].min(), st_diffs['2co2'].max()))

    print('co2-ctrl global mean: %f'%(weighted_average(st_diffs['2co2'], ds['avg'])))
    print('TCR: 1%%-ctrl global mean: %f'%(tcr))

    print('one_pct_2x TOA global mean: %f'%(toa_gm['1pct']))
    print('co2_2x TOA global mean: %f'%(toa_gm['2co2']))

    print('G_1pct: %f'%(G['1pct']))
    print('G_2co2: %f'%(G['2co2']))
    print('alpha, clim sens (CO2): %f, %f'%(alpha, clim_sens))


def raw_data_analysis(args, aavg_weight_edge):
    all_data = load_all_data(args, ('surf_temp', 'toa_swdown', 'toa_swup', 'olr'))
    res = {'all_data': all_data,
           'st_mean': {},
           'toa': {},
           'toa_mean': {} }
    for scenario in SCENARIOS.keys():
	res['st_mean'][scenario] = weighted_average(all_data[scenario]['surf_temp'].mean(axis=0), aavg_weight_edge)
	res['toa'][scenario] = all_data[scenario]['toa_swdown'] - all_data[scenario]['toa_swup'] - all_data[scenario]['olr']  
	res['toa_mean'][scenario] = weighted_average(res['toa'][scenario].mean(axis=0), aavg_weight_edge)
	print('Scenario: %s'%scenario)
	print('  mean surf temp: %f'%res['st_mean'])
	print('  mean toa: %f'%res['toa_mean'])
    return res

def create_args():
    parser = argparse.ArgumentParser(description='Simple Climate Modelling Analysis')
    parser.add_argument('-p','--plot', help='Plot figures', action='store_true')
    parser.add_argument('-o','--output', help='Output results', action='store_true')
    parser.add_argument('-r','--raw-data', help='Use 2nd method', action='store_true')
    parser.add_argument('-d','--data-dir', help='Data directory', default='data')
    parser.add_argument('-s','--scenario', help='Scenario', default='all')
    args = parser.parse_args()
    return args

def main(args):
    res = primary_analysis(args)

    if args.raw_data:
	res['raw'] = raw_data_analysis(args, res['ds']['avg'])
    return res

if __name__ == "__main__":
    args = create_args()
    main(args)
