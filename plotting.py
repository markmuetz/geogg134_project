import numpy as np
import pylab as plt
from mpl_toolkits.basemap import Basemap

def get_vars_from_control_dataset(control_dataset):
    lons = control_dataset.variables['longitude'][:]
    lats = control_dataset.variables['latitude'][:]
    lons -= 3.75/2 # Corrections to allow for grid sizes. Not sure why these are nec.
    lats += 2.5/2

    return lons, lats

def general_plot(control_dataset, data, vmin, vmax, loc='global', sa_mask=None):
    lons, lats = get_vars_from_control_dataset(control_dataset)
    if loc == 'global':
        plot_on_earth(lons, lats, data, vmin, vmax)
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

def plot_on_earth(lons, lats, data, vmin=-4, vmax=12):
    plot_lons, plot_data = extend_data(lons, lats, data)

    lons, lats = np.meshgrid(plot_lons, lats)

    m = Basemap(projection='cyl', resolution='c', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
    x, y = m(lons, lats)

    m.pcolormesh(x, y, plot_data, vmin=vmin, vmax=vmax)
    #m.pcolormesh(x, y, plot_data)

    m.drawcoastlines()
    m.drawparallels(np.arange(-90.,90.,45.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180.,180.,60.), labels=[0, 0, 0, 1], fontsize=10)

    m.colorbar(location='bottom', pad='7%')
    plt.show()

def plot_south_america(lons, lats, sa_mask, data, vmin=0, vmax=6):
    data_masked = np.ma.array(data, mask=sa_mask)
    plot_lons, plot_data = extend_data(lons, lats, data_masked)

    lons, lats = np.meshgrid(plot_lons, lats)

    m = Basemap(projection='cyl', resolution='c', llcrnrlat=-60, urcrnrlat=15, llcrnrlon=-85, urcrnrlon=-32)
    x, y = m(lons, lats)

    m.pcolormesh(x, y, plot_data, vmin=vmin, vmax=vmax)
    #m.pcolormesh(x, y, plot_data)

    m.drawcoastlines()
    m.drawparallels(np.arange(-60.,15.,10.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-90.,-30.,10.), labels=[0, 0, 0, 1], fontsize=10)

    m.colorbar(location='right', pad='5%')
    plt.show()

