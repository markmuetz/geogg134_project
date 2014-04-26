import numpy as np
import pylab as plt
from scipy import stats
from mpl_toolkits.basemap import Basemap

def get_vars_from_control_dataset(control_dataset):
    lons = control_dataset.variables['longitude'][:]
    lats = control_dataset.variables['latitude'][:]
    lons -= 3.75/2 # Corrections to allow for grid sizes. Not sure why these are nec.
    lats += 2.5/2

    return lons, lats

def plot_figures(control_dataset, data, sa_mask):
    surf_temp_diff = data['data']['1pct']['surf_temp'] - data['data']['ctrl']['surf_temp']

    if False:
	plt.figure('global_surf_temp')
	plt.set_cmap('coolwarm')
	vmin, vmax = -20, 20
	ax1 = plt.subplot(3, 1, 1)
	general_plot(control_dataset, np.roll(surf_temp_diff, 1, axis=0)[:3, :, :].mean(axis=0), vmin, vmax, ax=ax1)
	ax2 = plt.subplot(3, 1, 2)
	general_plot(control_dataset, np.roll(surf_temp_diff, 7, axis=0)[:3, :, :].mean(axis=0), vmin, vmax, ax=ax2)
	ax3 = plt.subplot(3, 1, 3)
	general_plot(control_dataset, surf_temp_diff.mean(axis=0), vmin, vmax, ax=ax3)

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


    plt.show()



def vec_general_plot(control_dataset, data_x, data_y, loc='global', sa_mask=None, ax=None):
    if ax == None:
	ax = plt.gca()
    lons, lats = get_vars_from_control_dataset(control_dataset)
    if loc == 'global':
        vec_plot_on_earth(lons, lats, data_x, data_y, ax)
    elif loc == 'N':
        vec_plot_polar(lons, lats, data_x, data_y, 'N', ax)
    elif loc == 'S':
        vec_plot_polar(lons, lats, data_x, data_y, 'S', ax)
    elif loc == 'sa':
        vec_plot_south_america(lons, lats, sa_mask, data_x, data_y, ax)

def general_plot(control_dataset, data, vmin, vmax, loc='global', sa_mask=None, ax=None):
    if ax == None:
	ax = plt.gca()
    lons, lats = get_vars_from_control_dataset(control_dataset)
    if loc == 'global':
        plot_on_earth(lons, lats, data, vmin, vmax, ax)
    elif loc == 'N':
        plot_polar(lons, lats, sa_mask, data, vmin, vmax, 'N', ax)
    elif loc == 'S':
        plot_polar(lons, lats, sa_mask, data, vmin, vmax, 'S', ax)
    elif loc == 'sa':
        plot_south_america(lons, lats, sa_mask, data, vmin, vmax, ax)

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

def plot_on_earth(lons, lats, data, vmin=-4, vmax=12, ax=None):
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
    m.drawparallels(np.arange(-90.,90.1,45.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180.,180.,60.), labels=[0, 0, 0, 1], fontsize=10)

    #import ipdb; ipdb.set_trace()
    m.colorbar(location='right', pad='7%')
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


def plot_south_america(lons, lats, sa_mask, data, vmin=0, vmax=6, ax=None):
    if ax == None:
	ax = plt.gca()
    if sa_mask != None:
	data_masked = np.ma.array(data, mask=sa_mask)
	plot_lons, plot_data = extend_data(lons, lats, data_masked)
    else:
	data_masked = data
	plot_lons, plot_data = extend_data(lons, lats, data_masked)

    lons, lats = np.meshgrid(plot_lons, lats)

    m = Basemap(ax=ax, projection='cyl', resolution='c', llcrnrlat=-60, urcrnrlat=15, llcrnrlon=-85, urcrnrlon=-32)
    x, y = m(lons, lats)

    m.pcolormesh(x, y, plot_data, vmin=vmin, vmax=vmax, ax=ax)
    #m.pcolormesh(x, y, plot_data, ax=ax)

    m.drawcoastlines(ax=ax)
    m.drawparallels(np.arange(-60.,15.,10.), labels=[1, 0, 0, 0], fontsize=10, ax=ax)
    m.drawmeridians(np.arange(-90.,-30.,10.), labels=[0, 0, 0, 1], fontsize=10, ax=ax)

    m.colorbar(location='right', pad='5%', ax=ax)
    plt.show()

