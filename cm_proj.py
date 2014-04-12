from __future__ import print_function
import inspect
import calendar
import datetime as dt
import argparse
import time
from glob import glob
from collections import OrderedDict
import numpy as np
import pylab as plt
import netCDF4 as nc

RAW_SCENARIOS = { #'mine': '../xjksa/datam/netcdf/*.nc',
                  #'ssto': '/home/ucfaohx/DATA/xjksb/datam/netcdf/*.nc',
                  'ctrl': '/home/ucfanea/DATA/xjksc/datam/netcdf/*.nc',
                  '1pct': '/home/ucfahub/DATA/xjksd/datam/netcdf/*.nc',
                  '2co2': '/data/geospatial_23/geogg134/um_output/double-co2/netcdf/*.nc'}

MONTHLY_SCENARIOS = { 'ctrl': '%s/climas/xjksc/xjksc_mean_allvars_%s.nc',
                      '1pct': '%s/climas/xjksd/xjksd_mean_allvars_%s.nc',
                      '2co2': '%s/climas/co2_2x/co2_2x_mean_allvars_%s.nc'}

SCENARIOS = { 'ctrl': '%s/climas/xjksc/xjksc_ANN_mean_allvars.nc',
              '1pct': '%s/climas/xjksd/xjksd_ANN_mean_allvars.nc',
              '2co2': '%s/climas/co2_2x/co2_2x_ANN_mean_allvars.nc'}

MONTHS = OrderedDict((v.lower(),k) for k,v in enumerate(calendar.month_abbr))

def get_sorted_filenames(args):
    data_dir = args.data_dir
    m = {'h': 0, 'i': 1, 'j': 2}
    filenames = glob(RAW_SCENARIOS[args.scenario])

    date_filenames = []
    for fn in filenames:
	fn_date = fn.split('/')[-1].split('.')[1] 
	# Ignore all files with 'h' in pos 2 (like create_climatology_nc).
	if fn_date[2] == 'h':
	    continue
	date = dt.datetime(2000 + m[fn_date[2]] * 10 + int(fn_date[3]), MONTHS[fn_date[-3:]], 1)
	date_filenames.append((date, fn))
    date_filenames.sort(key=lambda tup: tup[0])
    return [fn for d, fn in date_filenames]

def weighted_average(data, weights):
    return (data * weights).sum() / weights.sum()

def load_data(args, variables):
    fns = get_sorted_filenames(args)
    scenario_data = {}
    for var in variables:
	scenario_data[var] = []

    for fn in fns:
	#print(fn)
	ds = nc.Dataset(fn)
	for var in variables:
	    scenario_data[var].append(ds.variables[var][0, 0])
	ds.close()

    for var in variables:
	scenario_data[var] = np.array(scenario_data[var])

    return scenario_data

def load_all_data(args, variables):
    data = {}
    for scenario in RAW_SCENARIOS.keys():
	args.scenario = scenario
	#print('Load scenario: %s'%scenario)
	data[scenario] = load_data(args, variables)
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
    for scenario in RAW_SCENARIOS.keys():
	args.scenario = scenario
	variables = ['surf_temp', 'toa_swdown', 'toa_swup', 'olr']
	data[scenario] = calc_time_series(args, variables)
    return data

def create_sa_mask(ds):
    botmelt = ds['ctrl'].variables['botmelt'][:] # contains a sea mask
    seamask = botmelt.mask

    # Create a mask for just South America.
    rect_mask = np.zeros_like(botmelt.data[0, 0]).astype(bool)
    rect_mask[31:61, 74:89] = True

    return ~seamask | ~rect_mask

def regional_comparison_sa(args, ds, edge_aw):
    parser = argparse.ArgumentParser(description='Run interactive plotter')
    parser.add_argument('-g','--plot-global', help='Plot global figures', action='store_true', default=True)
    parser.add_argument('-l','--plot-local', help='Plot local figures', action='store_true')
    parser.add_argument('-d','--plot-diff', help='Plot diff', action='store_true')
    parser.add_argument('-o','--plot-polar', help='Polar', action='store_true')
    parser.add_argument('-s','--scen', help='Scenario')
    parser.add_argument('-t','--sleep', help='Sleep time', default=0.1)
    parser.add_argument('-p','--prev', help='Prev', action='store_true')
    parser.add_argument('-n','--next', help='Next', action='store_true')
    parser.add_argument('-v','--var', help='Variable')
    parser.add_argument('-c','--vec_var', help='Vec variable')
    parser.add_argument('-q','--quit', help='Quit', action='store_true')
    parser.add_argument('-a','--again', help='Again', action='store_true')

    data_dir = args.data_dir
    reg_ds = {}
    reg_data = {}

    sa_mask = create_sa_mask(ds)
    variables = ['surf_temp', 'precip', 'icedepth', 'snow', 'field1389', 'low_cloud', 'med_cloud', 'high_cloud',
                 'surf_u', 'surf_v']
    vec_variables = [('surf_u', 'surf_v')]
    var_index = 0

    for scenario in MONTHLY_SCENARIOS.keys():
        print('Loading data for %s'%scenario)
        reg_ds[scenario] = {}
        reg_data[scenario] = {}
        for var in variables:
            reg_data[scenario][var] = []

        for month in MONTHS.keys():
            if month != '':
                print('  loading %s'%month)
                reg_ds[scenario][month] = nc.Dataset(MONTHLY_SCENARIOS[scenario]%(data_dir, month))
                for var in variables:
                    reg_data[scenario][var].append(reg_ds[scenario][month].variables[var][0, 0])

        for var in variables:
            reg_data[scenario][var] = np.array(reg_data[scenario][var])

    import plotting
    plt.ion()
    r = ''
    var = variables[0]
    pargs = parser.parse_known_args([])[0]
    pargs.scen = '1pct'
    while True:
        if var == '':
            print('Input args: ', end='')
        else:
            print('Input args [%s:%s]: '%(pargs.scen, var), end='')
        r = raw_input()
        try:
            pargs_new = parser.parse_known_args(r.split())[0]
        except SystemExit:
            continue
        except:
            print('Could not parse args (-q to quit).')
            continue

        for field in dir(pargs):
            print(field)
            if not field.startswith('__') and getattr(pargs_new, field) != None:
                # Not working as intended yet, meant to just update things that aren't None.
                #print("setting %s to %s"%(field, str(getattr(pargs_new, field))))
                setattr(pargs, field, getattr(pargs_new, field))

        if pargs.quit:
            break

        if pargs.var != None:
            var = pargs.var

        if pargs.vec_var != None:
            pass
        else:
            if var not in variables:
                print('%s not recognised'%var)
                print(variables)
                continue
            elif pargs.next and var != '':
                var = variables[variables.index(var) + 1 % len(variables)]
            elif pargs.prev and var != '':
                var = variables[variables.index(var) - 1 + len(variables) % len(variables)]

        if pargs.plot_local:
            pargs.plot_global = False
        if pargs.plot_polar:
            pargs.plot_global = False

        print(var)
        
        if pargs.scen not in MONTHLY_SCENARIOS.keys():
            print('%s not recognised'%pargs.scen)
            print(MONTHLY_SCENARIOS.keys())
            continue
        scen = pargs.scen

        if pargs.plot_diff:
            diff = reg_data[scen][var] - reg_data['ctrl'][var]
            vmin, vmax = diff.min(), diff.max()

        for month in MONTHS.keys():
            print(month)
            if month == '':
                continue
            if pargs.plot_diff:
                if var == 'surf_temp':
                    vmin, vmax = 0, 10
                elif var == 'precip':
                    vmin, vmax = -10, 10
            else:
                if var == 'surf_temp':
                    vmin, vmax = 230, 320
                elif var == 'precip':
                    vmin, vmax = 0, 25
                elif var == 'icedepth':
                    vmin, vmax = 0, 3
                elif var == 'snow':
                    vmin, vmax = 0, 3
                elif var == 'field1389':
                    vmin, vmax = -2e-8, 10e-8
                elif var in ('low_cloud', 'med_cloud', 'high_cloud'):
                    vmin, vmax = 0, 1

            if pargs.vec_var != None:
                data_x = reg_data[scen]['surf_u'][MONTHS[month] - 1]
                data_y = reg_data[scen]['surf_v'][MONTHS[month] - 1]
            else:
                if not pargs.plot_diff:
                    av_diff = 'average'
                    data = reg_data[scen][var][MONTHS[month] - 1]
                else:
                    av_diff = 'diff (%s - ctrl)'%(scen)
                    data = diff[MONTHS[month] - 1]

            plt.clf()
            if pargs.vec_var != None:
                if pargs.plot_global:
                    plotting.vec_general_plot(ds['ctrl'], data_x, data_y)
                elif pargs.plot_local:
                    sa_mask = create_sa_mask(ds)
                    plotting.vec_general_plot(ds['ctrl'], data_x, data_y, 'sa', sa_mask)
            else:
                if pargs.plot_global:
                    plt.title('%s Global %s %s for %s'%(scen, month, av_diff, var))
                    #plt.imshow(data, interpolation='nearest', vmin=vmin, vmax=vmax)
                    plotting.general_plot(ds['ctrl'], data, vmin, vmax)
                elif pargs.plot_polar:
                    plt.title('%s SA %s %s for %s'%(scen, month, av_diff, var))
                    plotting.general_plot(ds['ctrl'], data, vmin, vmax, 'polar', sa_mask)
                elif pargs.plot_local:
                    plt.title('%s SA %s %s for %s'%(scen, month, av_diff, var))
                    sa_mask = create_sa_mask(ds)
                    plotting.general_plot(ds['ctrl'], data, vmin, vmax, 'sa', sa_mask)

            plt.pause(float(pargs.sleep))


    plt.close()

    return reg_ds, reg_data

def load_all_vars_datasets(args):
    data_dir = args.data_dir
    ds = {}

    # Load NetCDF datasets.
    for scenario in SCENARIOS.keys():
        ds[scenario] = nc.Dataset(SCENARIOS[scenario]%(data_dir))

    # Store the average weight (edge).
    edge_aw = ds['ctrl'].variables['aavg_weight_edge'][:, :]

    return ds, edge_aw

def primary_analysis(args):
    ds, edge_aw = load_all_vars_datasets(args)

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
        toa_gm[scenario] = weighted_average(toa_net_flux[scenario], edge_aw)

    # Work out some differences.
    st_diffs['2co2'] = sts['2co2'] - sts['ctrl']
    st_diffs['1pct'] = sts['1pct'] - sts['ctrl']

    tcr = weighted_average(st_diffs['1pct'], edge_aw)

    G['2co2']  = toa_gm['2co2'] - toa_gm['ctrl']
    G['1pct']  = toa_gm['1pct'] - toa_gm['ctrl']

    # Work out alpha (climate feedback param) and climate sensitivity.
    alpha = (G['2co2'] - G['1pct']) / tcr
    clim_sens = G['2co2'] / alpha

    # Plot on nice overlays. Note this WILL NOT work on UCL computers.
    if args.plot_local or args.plot_global:
        from plotting import plot_all
        sa_mask = create_sa_mask(ds)
        plot_all(ds['ctrl'], st_diffs['1pct'], toa_net_flux['2co2'], sa_mask, args)

    res = {'ds': ds,
           'edge_aw': edge_aw,
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
    edge_aw = res['edge_aw']
    sts = res['sts']
    st_diffs = res['st_diffs']
    tcr = res['tcr']
    toa_gm = res['toa_gm']
    G = res['G']
    alpha = res['alpha']
    clim_sens = res['clim_sens']

    # min/maxes
    print('1%%-ctrl min max: %f, %f'%(st_diffs['1pct'].min(), st_diffs['1pct'].max()))
    print('co2-ctrl min max: %f, %f'%(st_diffs['2co2'].min(), st_diffs['2co2'].max()))

    for scenario in SCENARIOS.keys():
	print('Scenario: %s'%scenario)
	print('  mean surf temp: %f'%(weighted_average(sts[scenario], edge_aw)))
	print('  mean toa: %f'%(toa_gm[scenario]))

    print('co2-ctrl global mean: %f'%(weighted_average(st_diffs['2co2'], edge_aw)))
    print('TCR: 1%%-ctrl global mean: %f'%(tcr))

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
	res['st_mean'][scenario]  = weighted_average(all_data[scenario]['surf_temp'].mean(axis=0), aavg_weight_edge)
	res['toa'][scenario]      = all_data[scenario]['toa_swdown'] -\
		                    all_data[scenario]['toa_swup'] -\
				    all_data[scenario]['olr']  
	res['toa_mean'][scenario] = weighted_average(res['toa'][scenario].mean(axis=0), aavg_weight_edge)

    for scenario in SCENARIOS.keys():
	print('Scenario: %s'%scenario)
	print('  mean surf temp: %f'%res['st_mean'][scenario])
	print('  mean toa: %f'%res['toa_mean'][scenario])

    return res

def create_args():
    parser = argparse.ArgumentParser(description='Simple Climate Modelling Analysis')
    parser.add_argument('-g','--plot-global', help='Plot global figures', action='store_true')
    parser.add_argument('-l','--plot-local', help='Plot local figures', action='store_true')
    parser.add_argument('-t','--plot-diff', help='Plot diff', action='store_true')
    parser.add_argument('-o','--output', help='Output results', action='store_true')
    parser.add_argument('-r','--raw-data', help='Raw data analysis (UCL only)', action='store_true')
    parser.add_argument('-e','--regional', help='Regional analysis', action='store_true')
    parser.add_argument('-d','--data-dir', help='Data directory', default='data')
    parser.add_argument('-s','--scenario', help='Scenario', default='all')
    parser.add_argument('-p','--sleep', help='Sleep time', default='0.1')
    args = parser.parse_args()
    return args

def main(args):
    res = primary_analysis(args)

    if args.raw_data:
	res['raw'] = raw_data_analysis(args, res['edge_aw'])

    if args.regional:
        res['reg'] = regional_comparison_sa(args, res['ds'], res['edge_aw'])

    return res

if __name__ == "__main__":
    args = create_args()
    main(args)
