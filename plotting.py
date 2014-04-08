import numpy as np
import pylab as plt
from mpl_toolkits.basemap import Basemap

def plot_on_earth(control_dataset, co2_2x_diff, one_pct_2x_diff):
    lons = control_dataset.variables['longitude'][:]
    lats = control_dataset.variables['latitude'][:]

    lons, lats = np.meshgrid(lons, lats)

    m = Basemap(projection='mill', lon_0=180)
    x, y = m(lons, lats)
    m.pcolormesh(x, y, co2_2x_diff, vmin=-4, vmax=12)
    m.drawcoastlines()
    plt.show()

    m.pcolormesh(x, y, one_pct_2x_diff, vmin=-4, vmax=12)
    m.drawcoastlines()
    plt.show()
