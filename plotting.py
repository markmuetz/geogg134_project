#!/usr/bin/python
# -*- coding: utf-8 -*-
import calendar
import numpy as np
import pylab as plt
from scipy import stats
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

def get_vars_from_control_dataset(control_dataset):
    lons = control_dataset.variables['longitude'][:]
    lats = control_dataset.variables['latitude'][:]
    lons -= 3.75/2 # Corrections to allow for grid sizes. Not sure why these are nec.
    lats += 2.5/2

    return lons, lats

def plot_global_figure(control_dataset, data):
    surf_temp_diff = data['data']['1pct']['surf_temp'] - data['data']['ctrl']['surf_temp']
    precip_diff = data['data']['1pct']['precip'] - data['data']['ctrl']['precip']
    f = plt.figure('global_temp_precip')
    f.subplots_adjust(hspace=0.35, wspace=0.28)
    vmin, vmax = -4., 20.
    zero_point = 1 - (vmax / (vmax - vmin))
    p2 = 0.3
    cdict = {'red':   ((0.0,  0.3, 0.3),
		       (zero_point,  0.9, 0.9),
		       (p2,  0.8, 0.8),
		       (1.0,  0.7, 0.7)),

	     'green': ((0.0,  0.5, 0.5),
		       (zero_point, 0.9, 0.9),
		       (p2,  0.8, 0.8),
		       (1.0,  0.0, 0.0)),

	     'blue':  ((0.0,  0.8, 0.8),
		       (zero_point,  0.9, 0.9),
		       (p2,  0.2, 0.2),
		       (1.0,  0.0, 0.0))}

    cust_st_cmap = LinearSegmentedColormap('cust_st', cdict)
    plt.register_cmap(cmap=cust_st_cmap)

    plt.set_cmap('cust_st')
    ax1 = plt.subplot(5, 2, 1)
    plt.title('$\Delta$ Surface Temp.(K)\nAnnual')
    general_plot(control_dataset, surf_temp_diff.mean(axis=0), vmin, vmax, cbar_ticks=np.arange(-4, 21, 4))

    ax2 = plt.subplot(5, 2, 3)
    plt.title('DJF')
    general_plot(control_dataset, np.roll(surf_temp_diff, 1, axis=0)[:3, :, :].mean(axis=0), vmin, vmax, cbar_ticks=np.arange(-4, 21, 4))

    ax3 = plt.subplot(5, 2, 5)
    plt.title('JJA')
    general_plot(control_dataset, np.roll(surf_temp_diff, 7, axis=0)[:3, :, :].mean(axis=0), vmin, vmax, cbar_ticks=np.arange(-4, 21, 4))

    ax4 = plt.subplot(5, 2, 2)
    vmin, vmax = -4., 4.
    zero_point = 1 - (vmax / (vmax - vmin))
    cdict = {'blue':   ((0.0,  0.1, 0.1),
		       (zero_point,  0.9, 0.9),
		       (1.0,  0.8, 0.8)),

	     'green': ((0.0,  0.4, 0.4),
		       (zero_point, 0.9, 0.9),
		       (1.0,  0.0, 0.0)),

	     'red':  ((0.0,  0.8, 0.8),
		       (zero_point,  0.9, 0.9),
		       (1.0,  0.0, 0.0))}
    cust_pr_cmap = LinearSegmentedColormap('cust_pr', cdict)
    plt.register_cmap(cmap=cust_pr_cmap)

    plt.title('$\Delta$ Precipitation (mm/day)\nAnnual')
    general_plot(control_dataset, precip_diff.mean(axis=0), vmin, vmax, cbar_loc='right', cbar_ticks=np.arange(-4, 4.1, 2))
    plt.set_cmap('cust_pr')

    ax5 = plt.subplot(5, 2, 4)
    plt.title('DJF')
    general_plot(control_dataset, np.roll(precip_diff, 1, axis=0)[:3, :, :].mean(axis=0), vmin, vmax, cbar_loc='right', cbar_ticks=np.arange(-4, 4.1, 2))

    ax6 = plt.subplot(5, 2, 6)
    plt.title('JJA')
    general_plot(control_dataset, np.roll(precip_diff, 7, axis=0)[:3, :, :].mean(axis=0), vmin, vmax, cbar_loc='right', cbar_ticks=np.arange(-4, 4.1, 2))

    ax7 = plt.subplot(5, 2, 7)
    plt.title('Seasonal mean temperature')
    plt.plot(np.arange(0.5, 11.6, 1), surf_temp_diff.mean(axis=(1, 2)), 'r-')
    months = [calendar.month_name[i + 1][:1] for i in range(12)]
    ax7.set_xticks(np.arange(12) + 1./2)
    ax7.set_xticklabels(months)
    plt.ylim((1, 4))
    ax7.set_yticks(np.arange(1, 4.1, 1))
    ax7.set_ylabel('$\Delta$ Temp. (K)')

    ax8 = plt.subplot(5, 2, 9)
    plt.title('Zonal mean temperature')
    plt.plot(np.arange(-90, 90, 180./73.), surf_temp_diff.mean(axis=(0, 2)), 'r-')

    ax8.set_xticks(np.arange(-90, 91, 45))
    plt.xlim((-90, 90))
    plt.ylim((0, 8))
    ax8.set_yticks(np.arange(0, 8.1, 2))
    ax8.set_ylabel('$\Delta$ Temp. (K)')

    def north_south_fmt(x, pos):
	if x > 0:
	    return u'%1.0f°S'%abs(x)
	elif x == 0:
	    return u'0°'
	else:
	    return u'%1.0f°N'%abs(x)
	    #return '%1.0fN'%abs(x)

    formatter = FuncFormatter(north_south_fmt)
    ax8.xaxis.set_major_formatter(formatter)

    ax9 = plt.subplot(5, 2, 8)
    plt.title('Seasonal average precip.')
    plt.plot(np.arange(0.5, 11.6, 1), precip_diff.mean(axis=(1, 2)))
    months = [calendar.month_name[i + 1][:1] for i in range(12)]
    ax9.set_xticks(np.arange(12) + 1./2)
    ax9.set_xticklabels(months)
    plt.ylim((0.05, 0.15))
    ax9.set_yticks(np.arange(0.05, 0.16, 0.05))
    ax9.set_ylabel('$\Delta$ Precip. (mm/day)')

    ax10 = plt.subplot(5, 2, 10)
    plt.title('Zonal average precip.')
    plt.plot(np.arange(-90, 90, 180./73.), precip_diff.mean(axis=(0, 2)))

    ax10.set_xticks(np.arange(-90, 91, 45))
    plt.xlim((-90, 90))
    plt.ylim((-1, 1))
    ax10.set_yticks(np.arange(-1, 1.1, 1))
    ax10.set_ylabel('$\Delta$ Precip. (mm/day)')
    ax10.xaxis.set_major_formatter(formatter)




def plot_sa_seasonal_figure(control_dataset, data, sa_mask):
    f = plt.figure('sa_seasonal')
    plt.set_cmap('RdGy_r')
    graph_settings = (
	    ((-4, 4), np.arange(-4, 4.1, 2)),
	    ((-6, 6), np.arange(-6, 6.1, 3)))

    variables = ['precip', 'surf_temp']
    nice_names = {'precip': '$\Delta$Precip (mm/day)', 
		  'surf_temp': '$\Delta$Surf temp (K)'}
    f.subplots_adjust(hspace=0.2, wspace=0.1)

    for j in range(len(variables)):
	for i, roll in enumerate([1, 10, 7, 4]):
	    titles = ('DJF', 'MAM', 'JJA', 'SON')
	    variable = variables[j]
	    ax = plt.subplot(2, 4, i + j * 4 + 1)
	    ax.set_title(titles[i])
	    variable_diff = data['data']['1pct'][variable] - data['data']['ctrl'][variable]
	    lons, lats = get_vars_from_control_dataset(control_dataset)
	    vmin, vmax = graph_settings[j][0]
	    plot_data = np.roll(variable_diff, roll, axis=0)[:3].mean(axis=0)

	    # unmasked.
	    data_masked = plot_data
	    plot_lons, plot_data = extend_data(lons, lats, data_masked)

	    lons, lats = np.meshgrid(plot_lons, lats)

	    m = Basemap(projection='cyl', resolution='c', llcrnrlat=-60, urcrnrlat=15, llcrnrlon=-85, urcrnrlon=-32)
	    x, y = m(lons, lats)

	    m.pcolormesh(x, y, plot_data, vmin=vmin, vmax=vmax)

	    m.drawcoastlines()
	    if i == 0:
		m.drawparallels(np.arange(-60.,15.,10.), labels=[1, 0, 0, 0], fontsize=10)
		ax.set_ylabel(nice_names[variable])
		ax.get_yaxis().set_label_coords(-0.25, 0.5)
	    elif i == 3:
		m.drawparallels(np.arange(-60.,15.,10.), labels=[0, 1, 0, 0], fontsize=10)
	    else:
		m.drawparallels(np.arange(-60.,15.,10.))

	    m.drawmeridians(np.arange(-90.,-30.,10.), labels=[0, 0, 0, 1], fontsize=10)

	    cbar = m.colorbar(location='bottom', pad='7%', ticks=graph_settings[j][1])
	    #m.colorbar.make_axes(cbar, fraction=0.8)
	    #cbar.ax.set_xticklabels(graph_settings[i][1])

def plot_sa_diff_figure(control_dataset, data, sa_mask):
    f = plt.figure('sa_diff')
    plt.set_cmap('RdGy_r')
    graph_settings = (
	    ((-4, 4), np.arange(-4, 4.1, 2)),
	    ((-6, 6), np.arange(-6, 6.1, 3)),
	    ((-0.7, 0.7), np.arange(-0.6, 0.61, 0.3)),
	    ((-4, 4), np.arange(-4, 4.1, 2)),
	    ((-0.2, 0.2), np.arange(-0.2, 0.21, 0.1)))

    variables = ['precip', 'surf_temp', 'q', 'field1389', 'field1385']
    nice_names = {'precip': '$\Delta$Precip (mm/day)', 
		  'surf_temp': '$\Delta$Surf temp (K)', 
		  'q':'$\Delta$Humidity (g/kg)', 
		  'field1389': '$\Delta$NPP (g/m$^2$/day)', 
		  'field1385': '$\Delta$Soil moisture'}

    f.subplots_adjust(hspace=0.2, wspace=0.1)
    for i in range(len(variables)):
	variable = variables[i]
	ax = plt.subplot(2, 3, i + 1)
	ax.set_title(nice_names[variable])
	variable_diff = data['data']['1pct'][variable] - data['data']['ctrl'][variable]
	if variable == 'field1389':
	    variable_diff *= 24*60*60*1000 # per s to per day, kg to g.
	lons, lats = get_vars_from_control_dataset(control_dataset)
	vmin, vmax = graph_settings[i][0]
	#general_plot(control_dataset, variable_diff.mean(axis=0), vmin=graph_settings[i][0][0], vmax=graph_settings[i][0][1], loc='sa', sa_mask=sa_mask)
	plot_data = variable_diff.mean(axis=0)
	#plot_south_america(lons, lats, sa_mask, plot_data, vmin, vmax)

	if variable in ('surf_temp', 'precip', 'q'):
	    # unmasked.
	    data_masked = plot_data
	    plot_lons, plot_data = extend_data(lons, lats, data_masked)
	else:
	    data_masked = np.ma.array(plot_data, mask=sa_mask)
	    plot_lons, plot_data = extend_data(lons, lats, data_masked)

	lons, lats = np.meshgrid(plot_lons, lats)

	m = Basemap(projection='cyl', resolution='c', llcrnrlat=-60, urcrnrlat=15, llcrnrlon=-85, urcrnrlon=-32)
	x, y = m(lons, lats)

	m.pcolormesh(x, y, plot_data, vmin=vmin, vmax=vmax)

	m.drawcoastlines()
	if i == 0 or i == 3:
	    m.drawparallels(np.arange(-60.,15.,10.), labels=[1, 0, 0, 0], fontsize=10)
	elif i == 2 or i == 4:
	    m.drawparallels(np.arange(-60.,15.,10.), labels=[0, 1, 0, 0], fontsize=10)
	else:
	    m.drawparallels(np.arange(-60.,15.,10.))

	m.drawmeridians(np.arange(-90.,-30.,10.), labels=[0, 0, 0, 1], fontsize=10)

	cbar = m.colorbar(location='bottom', pad='7%', ticks=graph_settings[i][1])
	#m.colorbar.make_axes(cbar, fraction=0.8)
	#cbar.ax.set_xticklabels(graph_settings[i][1])

def plot_corr_figure(control_dataset, data, sa_mask):
    f = plt.figure('corr')
    variables = ['precip', 'surf_temp', 'q', 'field1389', 'field1385']

    graph_settings = (
	    ((-4, 2), np.arange(-4, 2.1, 2)),
	    ((1, 6), np.arange(1, 6.1, 1)),
	    ((-0.1, 0.7), np.arange(0.0, 0.7, 0.2)),
	    ((-4, 3), np.arange(-4, 3, 2)),
	    ((-0.2, 0.1), np.arange(-0.2, 0.11, 0.1)))

    nice_names = {'precip': '$\Delta$Precip (mm/day)', 
		  'surf_temp': '$\Delta$Surf temp (K)', 
		  'q':'$\Delta$Humidity (g/kg)', 
		  'field1389': '$\Delta$NPP (g/m$^2$/day)', 
		  'field1385': '$\Delta$Soil moisture'}

    f.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(len(variables)):
	for j in range(i + 1, len(variables)):
		var1 = variables[i]
		var2 = variables[j]
		ax = plt.subplot(4, 4, i + 4 * (j - 1) + 1)

		if i == 0:
		    ax.set_ylabel(nice_names[var2])
		    ax.get_yaxis().set_label_coords(-0.25, 0.5)
		else:
		    plt.setp(ax.get_yticklabels(), visible=False)

		if j == 4:
		    ax.set_xlabel(nice_names[var1])
		    ax.get_xaxis().set_label_coords(0.5, -0.2)
		else:
		    plt.setp(ax.get_xticklabels(), visible=False)

		plt.xlim(graph_settings[i][0])
		plt.ylim(graph_settings[j][0])
		ax.set_xticks(graph_settings[i][1])
		ax.set_yticks(graph_settings[j][1])

		var1_diff = data['data']['1pct'][var1] - data['data']['ctrl'][var1]
		var2_diff = data['data']['1pct'][var2] - data['data']['ctrl'][var2]

		sa_var1_diff = np.ma.array(var1_diff.mean(axis=0), mask=sa_mask)
		sa_var2_diff = np.ma.array(var2_diff.mean(axis=0), mask=sa_mask)

		x = sa_var1_diff.data[~sa_var1_diff.mask]
		y = sa_var2_diff.data[~sa_var2_diff.mask]

		if var1 == 'field1389':
		    x *= 24*60*60*1000 # per s to per day, kg to g.
		if var2 == 'field1389':
		    y *= 24*60*60*1000 # per s to per day, kg to g.

		slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
		label = 'slope=%1.2f\nintercept=%1.2f\nr=%1.2f'%(slope, intercept, r_value)
		l2 = 'r=%1.2f'%(r_value)
		r2, p2 = stats.pearsonr(x, y)

		#ax.set_title('%s'%(l2))
		plt.plot(x.flatten(), y.flatten(), 'kx', label=l2)
		#print(label)
		#import ipdb; ipdb.set_trace()
		#print(l2)
		#l2 = 'corr=%1.2f'%np.correlate(x, y)
		#print(l2)

		#line = [slope * x.min() + intercept, slope * x.max() + intercept]
		#plt.plot([x.min(), x.max()], line, 'b-', label=l2)
		plt.legend(loc='best', prop={'size':10})


def plot_figures(control_dataset, data, sa_mask):
    #import ipdb; ipdb.set_trace()

    plot_global_figure(control_dataset, data)
    #plot_sa_seasonal_figure(control_dataset, data, sa_mask)
    #plot_sa_diff_figure(control_dataset, data, sa_mask)
    #plot_corr_figure(control_dataset, data, sa_mask)
    plt.show()

def vec_general_plot(control_dataset, data_x, data_y, loc='global', sa_mask=None):
    lons, lats = get_vars_from_control_dataset(control_dataset)
    if loc == 'global':
        vec_plot_on_earth(lons, lats, data_x, data_y)
    elif loc == 'N':
        vec_plot_polar(lons, lats, data_x, data_y, 'N')
    elif loc == 'S':
        vec_plot_polar(lons, lats, data_x, data_y, 'S')
    elif loc == 'sa':
        vec_plot_south_america(lons, lats, sa_mask, data_x, data_y)

def general_plot(control_dataset, data, vmin, vmax, loc='global', sa_mask=None, cbar_loc='right', cbar_ticks=None):
    lons, lats = get_vars_from_control_dataset(control_dataset)
    if loc == 'global':
        plot_on_earth(lons, lats, data, vmin, vmax, cbar_loc, cbar_ticks)
    elif loc == 'N':
        plot_polar(lons, lats, sa_mask, data, vmin, vmax, 'N')
    elif loc == 'S':
        plot_polar(lons, lats, sa_mask, data, vmin, vmax, 'S')
    elif loc == 'sa':
        plot_south_america(lons, lats, sa_mask, data, vmin, vmax)

def plot_all(control_dataset, one_pct_2x_diff, co2_2x_toa_net_flux, sa_mask, args):
    lons, lats = get_vars_from_control_dataset(control_dataset)

    if args.plot_global:
        plot_on_earth(lons, lats, one_pct_2x_diff)
        plot_on_earth(lons, lats, co2_2x_toa_net_flux, -100, 100)

    if args.plot_local:
        plot_south_america(lons, lats, sa_mask, one_pct_2x_diff)
        plot_south_america(lons, lats, sa_mask, co2_2x_toa_net_flux, -100, 100)

def extend_data(lons, lats, data):
    if False:
        # Adds extra data at the end.
        plot_offset = 2
        plot_lons = np.zeros((lons.shape[0] + plot_offset,))
        plot_lons[:-plot_offset] = lons
        plot_lons[-plot_offset:] = lons[-plot_offset:] + 3.75 * plot_offset

        plot_data = np.zeros((data.shape[0], data.shape[1] + plot_offset))
        plot_data[:, :-plot_offset] = data
        plot_data[:, -plot_offset:] = data[:, :plot_offset]
    else:
        # Adds extra data before the start.
        plot_offset = 50
        plot_lons = np.ma.zeros((lons.shape[0] + plot_offset,))
        plot_lons[plot_offset:] = lons
        plot_lons[:plot_offset] = lons[-plot_offset:] - 3.75 * (lons.shape[0])

        plot_data = np.ma.zeros((data.shape[0], data.shape[1] + plot_offset))
        plot_data[:, plot_offset:] = data
        plot_data[:, :plot_offset] = data[:, -plot_offset:]

    return plot_lons, plot_data

def plot_on_earth(lons, lats, data, vmin=-4, vmax=12, cbar_loc='left', cbar_ticks=None):
    #import ipdb; ipdb.set_trace()
    #if ax == None:
	#ax = plt.gca()
    plot_lons, plot_data = extend_data(lons, lats, data)

    lons, lats = np.meshgrid(plot_lons, lats)

    m = Basemap(projection='cyl', resolution='c', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
    x, y = m(lons, lats)

    m.pcolormesh(x, y, plot_data, vmin=vmin, vmax=vmax)
    #m.pcolormesh(x, y, plot_data)

    m.drawcoastlines()
    if cbar_loc == 'left':
	p_labels = [0, 1, 0, 0]
    else:
	p_labels = [1, 0, 0, 0]

    m.drawparallels(np.arange(-90.,90.1,45.), labels=p_labels, fontsize=10)
    m.drawmeridians(np.arange(-180.,180.,60.), labels=[0, 0, 0, 1], fontsize=10)

    #import ipdb; ipdb.set_trace()
    if cbar_ticks == None:
	cbar = m.colorbar(location=cbar_loc, pad='7%')
    else:
	cbar = m.colorbar(location=cbar_loc, pad='7%', ticks=cbar_ticks)

    if cbar_loc == 'left':
	cbar.ax.xaxis.get_offset_text().set_position((10,0))
    #plt.show()

def vec_plot_on_earth(lons, lats, x_data, y_data, vmin=-4, vmax=12, ax=None):
    if ax == None:
	ax = plt.gca()
    plot_lons, plot_x_data = extend_data(lons, lats, x_data)
    plot_lons, plot_y_data = extend_data(lons, lats, y_data)

    lons, lats = np.meshgrid(plot_lons, lats)

    m = Basemap(ax=ax, projection='cyl', resolution='c', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
    x, y = m(lons, lats)

    mag = np.sqrt(plot_x_data**2 + plot_y_data**2)
    vmin, vmax = mag.min(), mag.max()
    m.contourf(x[:-1,:], y[:-1,:], mag, ax=ax)
    #m.pcolormesh(x, y, mag, vmin=vmin, vmax=vmax)
    #m.quiver(x, y, plot_x_data, plot_y_data)
    skip = 5
    m.quiver(x[::skip, ::skip], y[::skip, ::skip], plot_x_data[::skip, ::skip], plot_y_data[::skip, ::skip], ax=ax)

    m.drawcoastlines(ax=ax)
    m.drawparallels(np.arange(-90.,90.,45.), labels=[1, 0, 0, 0], fontsize=10, ax=ax)
    m.drawmeridians(np.arange(-180.,180.,60.), labels=[0, 0, 0, 1], fontsize=10, ax=ax)


    m.colorbar(location='bottom', pad='7%', ax=ax)
    plt.show()

def vec_plot_south_america(lons, lats, sa_mask, data_x, data_y, ax=None):
    if ax == None:
	ax = plt.gca()
    #print(sa_mask.shape)
    #print(data_x.shape)
    #print(data_y.shape)
    #data_x_masked = np.ma.array(data_x, mask=sa_mask)
    #data_y_masked = np.ma.array(data_y, mask=sa_mask)
    plot_lons, plot_x_data = extend_data(lons, lats, data_x)
    plot_lons, plot_y_data = extend_data(lons, lats, data_y)

    lons, lats = np.meshgrid(plot_lons, lats)

    m = Basemap(ax=ax, projection='cyl', resolution='c', llcrnrlat=-60, urcrnrlat=15, llcrnrlon=-85, urcrnrlon=-32)
    x, y = m(lons, lats)

    mag = np.sqrt(plot_x_data**2 + plot_y_data**2)
    vmin, vmax = mag.min(), mag.max()
    m.contourf(x[:-1,:], y[:-1,:], mag, ax=ax)
    #m.pcolormesh(x, y, mag, vmin=vmin, vmax=vmax, ax=ax)
    m.quiver(x, y, plot_x_data, plot_y_data, scale=40, ax=ax)

    m.drawcoastlines(ax=ax)
    m.drawparallels(np.arange(-60.,15.,10.), labels=[1, 0, 0, 0], fontsize=10, ax=ax)
    m.drawmeridians(np.arange(-90.,-30.,10.), labels=[0, 0, 0, 1], fontsize=10, ax=ax)

    m.colorbar(location='right', pad='5%', ax=ax)
    plt.show()

def vec_plot_polar(lons, lats, data_x, data_y, pole, ax=None):
    if ax == None:
	ax = plt.gca()
    plot_lons, plot_x_data = extend_data(lons, lats, data_x)
    plot_lons, plot_y_data = extend_data(lons, lats, data_y)

    lons, lats = np.meshgrid(plot_lons, lats)

    if pole == 'N':
	m = Basemap(ax=ax, resolution='c',projection='stere',lat_0=90.,lon_0=0, width=12000000,height=8000000)
    elif pole == 'S':
	m = Basemap(ax=ax, resolution='c',projection='stere',lat_0=-90.,lon_0=0, width=12000000,height=8000000)
    x, y = m(lons, lats)

    mag = np.sqrt(plot_x_data**2 + plot_y_data**2)
    vmin, vmax = mag.min(), mag.max()
    m.contourf(x[:-1,:], y[:-1,:], mag, ax=ax)
    #m.pcolormesh(x, y, mag, vmin=vmin, vmax=vmax, ax=ax)
    skip = 5
    m.quiver(x[::5, ::5], y[::5, ::5], plot_x_data[::5, ::5], plot_y_data[::5, ::5], scale=50, ax=ax)

    m.drawcoastlines(ax=ax)
    m.drawparallels(np.arange(-60.,15.,10.), labels=[1, 0, 0, 0], fontsize=10, ax=ax)
    m.drawmeridians(np.arange(-180.,180.,10.), fontsize=10, ax=ax)

    m.colorbar(location='right', pad='5%', ax=ax)
    plt.show()

def plot_polar(lons, lats, sa_mask, data, vmin=0, vmax=6, pole='N', ax=None):
    if ax == None:
	ax = plt.gca()
    plot_lons, plot_data = extend_data(lons, lats, data)

    lons, lats = np.meshgrid(plot_lons, lats)

    if pole == 'N':
	m = Basemap(ax=ax, resolution='c',projection='stere',lat_0=90.,lon_0=0, width=12000000,height=8000000)
    elif pole == 'S':
	m = Basemap(ax=ax, resolution='c',projection='stere',lat_0=-90.,lon_0=0, width=12000000,height=8000000)
    x, y = m(lons, lats)

    m.pcolormesh(x, y, plot_data, vmin=vmin, vmax=vmax, ax=ax)

    m.drawcoastlines(ax=ax)
    m.drawparallels(np.arange(-60.,15.,10.), labels=[1, 0, 0, 0], fontsize=10, ax=ax)
    m.drawmeridians(np.arange(-180.,180.,10.), fontsize=10, ax=ax)

    m.colorbar(location='right', pad='5%', ax=ax)
    plt.show()


def plot_south_america(lons, lats, sa_mask, data, vmin=0, vmax=6):
    if sa_mask != None:
	data_masked = np.ma.array(data, mask=sa_mask)
	plot_lons, plot_data = extend_data(lons, lats, data_masked)
    else:
	data_masked = data
	plot_lons, plot_data = extend_data(lons, lats, data_masked)

    lons, lats = np.meshgrid(plot_lons, lats)

    m = Basemap(projection='cyl', resolution='c', llcrnrlat=-60, urcrnrlat=15, llcrnrlon=-85, urcrnrlon=-32)
    x, y = m(lons, lats)

    m.pcolormesh(x, y, plot_data, vmin=vmin, vmax=vmax)
    #m.pcolormesh(x, y, plot_data, ax=ax)

    m.drawcoastlines()
    m.drawparallels(np.arange(-60.,15.,10.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-90.,-30.,10.), labels=[0, 0, 0, 1], fontsize=10)

    m.colorbar(location='bottom', pad='5%')

