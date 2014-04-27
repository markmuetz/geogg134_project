#!/usr/bin/python
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

def get_all_data(args, ds, edge_aw):
    data_dir = args.data_dir
    all_ds = {}
    all_data = {}
    all_ts_data = {}
    all_lts_data = {}

    sa_mask = create_sa_mask(ds)
    if not args.use_all_variables:
        # 1385: soil moisture
        # 1389: NPP
        # q: humidity
        variables = ['surf_temp', 'precip', 'field1389', 'field1385', 'evap', 'q']
        #variables = ['surf_temp', 'precip', 'icedepth', 'snow', 'field1389', 'low_cloud', 'med_cloud', 'high_cloud',
                     #'surf_u', 'surf_v']
    else:
        variables = None

    vec_variables = ['surf']
    var_index = 0

    for scenario in MONTHLY_SCENARIOS.keys():
        print('Loading data for %s'%scenario)
        all_ds[scenario] = {}
        all_data[scenario] = {}
        all_ts_data[scenario] = {}
        all_lts_data[scenario] = {}
        if variables == None:
            first_ds = nc.Dataset(MONTHLY_SCENARIOS[scenario]%(data_dir, 'jan'))
            variables = first_ds.variables.keys()
            print(variables)

        for var in variables:
            all_data[scenario][var] = []
            all_ts_data[scenario][var] = []
            all_lts_data[scenario][var] = []

        for month in MONTHS.keys():
            if month != '':
                print('  loading %s'%month)
                all_ds[scenario][month] = nc.Dataset(MONTHLY_SCENARIOS[scenario]%(data_dir, month))

                for var in variables:
                    try:
                        # These have vertical distributions, take their means over these levels.
                        if var in ('q', 'field1385'):
                            data = all_ds[scenario][month].variables[var][0].mean(axis=0)
                        else:
                            data = all_ds[scenario][month].variables[var][0, 0]

                        all_data[scenario][var].append(data)
                        all_ts_data[scenario][var].append(weighted_average(data, edge_aw))
                        masked_var = np.ma.masked_array(data, mask=sa_mask[0, 0])
                        all_lts_data[scenario][var].append(weighted_average(masked_var, edge_aw))
                    except Exception, e:
                        print('Could not load %s'%var)
                        print(e.message)
                        #raise
                        variables.remove(var)
                        

        for var in variables:
            all_data[scenario][var] = np.array(all_data[scenario][var])
            all_ts_data[scenario][var] = np.array(all_ts_data[scenario][var])
            all_lts_data[scenario][var] = np.array(all_lts_data[scenario][var])
    return all_data, all_ts_data, all_lts_data

def plot_interactive(ds, inter_data, inter_ts_data, inter_lts_data):
    variables = ['surf_temp', 'precip', 'field1389', 'field1385', 'evap', 'q']
    import plotting
    plt.ion()

    parser = argparse.ArgumentParser(description='Run interactive plotter')
    parser.add_argument('-g','--plot-global', help='Plot global figures', action='store_true', default=True)
    parser.add_argument('-l','--plot-local', help='Plot local figures', action='store_true')
    parser.add_argument('-d','--plot-diff', help='Plot diff', action='store_true')
    parser.add_argument('-o','--plot-polar', help='Polar')
    parser.add_argument('-s','--scen', help='Scenario')
    parser.add_argument('--sleep', help='Sleep time', default=0.1)
    parser.add_argument('-t','--time-series', help='Time series', action='store_true')
    parser.add_argument('-f','--figure', help='Use figure')
    parser.add_argument('-c','--close', help='Close figure', action='store_true')

    parser.add_argument('-v','--var', help='Variable')
    parser.add_argument('--vec_var', help='Vec variable')
    parser.add_argument('-q','--quit', help='Quit', action='store_true')
    parser.add_argument('-a','--average', help='Average', action='store_true')
    parser.add_argument('-i','--ipython', help='Drop into ipython shell', action='store_true')
    parser.add_argument('--vmin', help='vmin')
    parser.add_argument('--vmax', help='vmax')
    parser.add_argument('-m','--months', help='Months')
    parser.add_argument('--ax', help='Axes')

    var = 'surf_temp'
    vec_var = None
    scen = '1pct'
    figure = '1'

    plt.figure(figure)
    pargs = parser.parse_known_args([])[0]

    while True:
        if var == '':
            print('Input args: ', end='')
        else:
            print('Input args [fig-%s:%s:%s]: '%(figure, scen, var), end='')
        r = raw_input()
        try:
            pargs = parser.parse_known_args(r.split())[0]
        except SystemExit:
            continue
        except:
            print('Could not parse args (-q to quit).')
            continue

        if pargs.ipython:
            import ipdb; ipdb.set_trace()

        if pargs.ax:
            ax = plt.subplot(pargs.ax)
            plt.clf()
        else:
            ax = None

        if pargs.quit:
            break

        if pargs.vec_var:
            if pargs.vec_var not in vec_variables:
                print('%s not recognised'%var)
                print(vec_variables)
                continue
            vec_var = pargs.vec_var
            var = vec_var
            continue
        else:
            if pargs.var:
                vec_var = None
                if pargs.var not in variables:
                    print('%s not recognised'%var)
                    print(variables)
                    continue
                var = pargs.var
                continue

        if pargs.scen:
            if pargs.scen not in MONTHLY_SCENARIOS.keys():
                print('%s not recognised'%pargs.scen)
                print(MONTHLY_SCENARIOS.keys())
                continue
            scen = pargs.scen
            continue

        if pargs.figure:
            figure = pargs.figure
            plt.figure(figure)
            continue

        if pargs.close:
            plt.close('all')
            continue

        if pargs.time_series:
            #plt.clf()
            if pargs.plot_local:
                if pargs.plot_diff:
                    plt.title('Difference (%s - ctlr) for %s'%(scen, var))
                    plt.plot(inter_lts_data[scen][var] - inter_lts_data['ctrl'][var])
                else:
                    plt.title('%s for %s'%(scen, var))
                    plt.plot(inter_lts_data[scen][var])
            else:
                if pargs.plot_diff:
                    plt.title('Difference (%s - ctlr) for %s'%(scen, var))
                    plt.plot(inter_ts_data[scen][var] - inter_ts_data['ctrl'][var])
                else:
                    plt.title('%s for %s'%(scen, var))
                    plt.plot(inter_ts_data[scen][var])
            continue

        if pargs.plot_local:
            pargs.plot_global = False
        if pargs.plot_polar:
            pargs.plot_global = False

        if pargs.plot_diff:
            diff = inter_data[scen][var] - inter_data['ctrl'][var]
            vmin, vmax = diff.min(), diff.max()

        if not pargs.plot_diff:
            vmin, vmax = None, None
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

        if pargs.vmin:
            vmin = pargs.vmin
        if pargs.vmax:
            vmax = pargs.vmax

        rolls = {
                'djf':1,
                'son':4,
                'jja':7,
                'jjas':7,
                'mam':10}
        num_months = {
                'djf':3,
                'son':3,
                'jja':3,
                'jjas':4,
                'mam':3}

        plt.set_cmap('RdBu_r')
        #plt.set_cmap('RdYlGn_r')

        if pargs.average:
            if pargs.months:
                roll = rolls[pargs.months]
                nm = num_months[pargs.months]
                if vec_var:
                    data_x = np.roll(inter_data[scen]['%s_u'%var], roll, axis=0)[:nm, :, :].mean(axis=0)
                    data_y = np.roll(inter_data[scen]['%s_v'%var], roll, axis=0)[:nm, :, :].mean(axis=0)
                    av_diff = 'average'
                else:
                    if not pargs.plot_diff:
                        av_diff = 'average'
                        data = np.roll(inter_data[scen][var], roll, axis=0)[:nm, :, :].mean(axis=0)
                    else:
                        av_diff = 'diff (%s - ctrl)'%(scen)
                        data = np.roll(diff, roll, axis=0)[:nm, :, :].mean(axis=0)

            else:
                if vec_var:
                    data_x = inter_data[scen]['%s_u'%var].mean(axis=0)
                    data_y = inter_data[scen]['%s_v'%var].mean(axis=0)
                    av_diff = 'average'
                else:
                    if not pargs.plot_diff:
                        av_diff = 'average'
                        data = inter_data[scen][var].mean(axis=0)
                    else:
                        av_diff = 'diff (%s - ctrl)'%(scen)
                        data = diff.mean(axis=0)

            plt.clf()

            if vmin == None:
                vmin = data.min()
            if vmax == None:
                vmax = data.max()


            if vec_var:
                if pargs.plot_global:
                    plt.title('%s Global %s %s for %s'%(scen, pargs.months, av_diff, var))
                    plotting.vec_general_plot(ds['ctrl'], data_x, data_y, ax=ax)
                elif pargs.plot_polar:
                    plt.title('%s SA %s %s for %s'%(scen, pargs.months, av_diff, var))
                    plotting.vec_general_plot(ds['ctrl'], data_x, data_y, pargs.plot_polar, ax=ax)
                elif pargs.plot_local:
                    plt.title('%s SA %s %s for %s'%(scen, pargs.months, av_diff, var))
                    #sa_mask = create_sa_mask(ds)
                    plotting.vec_general_plot(ds['ctrl'], data_x, data_y, 'sa', None, ax=ax)
            else:
                if pargs.plot_global:
                    plt.title('%s Global %s %s for %s'%(scen, pargs.months, av_diff, var))
                    #plt.imshow(data, interpolation='nearest', vmin=vmin, vmax=vmax)
                    plotting.general_plot(ds['ctrl'], data, vmin, vmax, ax=ax)
                elif pargs.plot_polar:
                    plt.title('%s SA %s %s for %s'%(scen, pargs.months, av_diff, var))
                    plotting.general_plot(ds['ctrl'], data, vmin, vmax, pargs.plot_polar, ax=ax)
                elif pargs.plot_local:
                    plt.title('%s SA %s %s for %s'%(scen, pargs.months, av_diff, var))
                    #sa_mask = create_sa_mask(ds)
                    plotting.general_plot(ds['ctrl'], data, vmin, vmax, 'sa', None, ax=ax)
        else:
            for month in MONTHS.keys():
                if month == '':
                    continue
                if vec_var:
                    data_x = inter_data[scen]['%s_u'%var][MONTHS[month] - 1]
                    data_y = inter_data[scen]['%s_v'%var][MONTHS[month] - 1]
                    av_diff = 'average'
                else:
                    if not pargs.plot_diff:
                        av_diff = 'average'
                        data = inter_data[scen][var][MONTHS[month] - 1]
                    else:
                        av_diff = 'diff (%s - ctrl)'%(scen)
                        data = diff[MONTHS[month] - 1]

                if vmin == None:
                    vmin = data.min()
                if vmax == None:
                    vmax = data.max()
                plt.clf()
                if vec_var:
                    if pargs.plot_global:
                        plt.title('%s Global %s %s for %s'%(scen, month, av_diff, var))
                        plotting.vec_general_plot(ds['ctrl'], data_x, data_y, ax=ax)
                    elif pargs.plot_polar:
                        plt.title('%s SA %s %s for %s'%(scen, month, av_diff, var))
                        plotting.vec_general_plot(ds['ctrl'], data_x, data_y, pargs.plot_polar, ax=ax)
                    elif pargs.plot_local:
                        plt.title('%s SA %s %s for %s'%(scen, month, av_diff, var))
                        sa_mask = create_sa_mask(ds)
                        plotting.vec_general_plot(ds['ctrl'], data_x, data_y, 'sa', sa_mask, ax=ax)
                else:
                    if pargs.plot_global:
                        plt.title('%s Global %s %s for %s'%(scen, month, av_diff, var))
                        #plt.imshow(data, interpolation='nearest', vmin=vmin, vmax=vmax)
                        plotting.general_plot(ds['ctrl'], data, vmin, vmax, ax=ax)
                    elif pargs.plot_polar:
                        plt.title('%s SA %s %s for %s'%(scen, month, av_diff, var))
                        plotting.general_plot(ds['ctrl'], data, vmin, vmax, pargs.plot_polar, ax=ax)
                    elif pargs.plot_local:
                        plt.title('%s SA %s %s for %s'%(scen, month, av_diff, var))
                        sa_mask = create_sa_mask(ds)
                        plotting.general_plot(ds['ctrl'], data, vmin, vmax, 'sa', sa_mask, ax=ax)

                plt.pause(float(pargs.sleep))

    plt.close()

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
    parser.add_argument('-p','--plot', help='Plot figures', action='store_true')
    parser.add_argument('-g','--plot-global', help='Plot global figures', action='store_true')
    parser.add_argument('-l','--plot-local', help='Plot local figures', action='store_true')
    parser.add_argument('-t','--plot-diff', help='Plot diff', action='store_true')
    parser.add_argument('-o','--output', help='Output results', action='store_true')
    parser.add_argument('-r','--raw-data', help='Raw data analysis (UCL only)', action='store_true')
    parser.add_argument('-i','--interactive', help='Interactive mode', action='store_true')
    parser.add_argument('-d','--data-dir', help='Data directory', default='data')
    parser.add_argument('-s','--scenario', help='Scenario', default='all')
    parser.add_argument('-u','--use-all-variables', help='Uses all variables, slow', action='store_true')
    args = parser.parse_args()
    return args

def plot(res):
    import plotting
    sa_mask = create_sa_mask(res['ds'])
    plotting.plot_figures(res['ds']['ctrl'], res['all'], sa_mask)

def main(args):
    res = primary_analysis(args)

    if args.raw_data:
	res['raw'] = raw_data_analysis(args, res['edge_aw'])

    res['all'] = {}
    res['all']['data'], res['all']['ts_data'], res['all']['lts_data'] = get_all_data(args, res['ds'], res['edge_aw'])
    if args.plot:
        plot(res)
    if args.interactive:
        plot_interactive(res['ds'], res['all']['data'], res['all']['ts_data'], res['all']['lts_data'])

    return res

if __name__ == "__main__":
    args = create_args()
    main(args)
