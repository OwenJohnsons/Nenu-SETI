'''
Author: Ruaidhrí Campion and Owen A. Johnson (Minor Revisions)
Code Purpose: 
Last Major Update: 2023-11-07
'''
#%%
from astroquery.gaia import Gaia
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from matplotlib.projections.geo import AitoffAxes
import matplotlib.projections as mprojections
from matplotlib.axes import Axes
from matplotlib.patches import Wedge
import matplotlib.spines as mspines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scienceplots
plt.style.use('science')


class UpperAitoffAxes(AitoffAxes):
    name = "upper_aitoff"

    def cla(self):
        AitoffAxes.cla(self)
        Axes.set_xlim(self, -np.pi, np.pi)
        Axes.set_ylim(self, 0., .5*np.pi)

    def _gen_axes_patch(self):
        return Wedge((0.5, .5), .5, 0, 180)

    def _gen_axes_spines(self):
        path = Wedge((0, 0), 1., 0, 180).get_path()
        spine = mspines.Spine(self, 'circle', path)
        spine.set_patch_circle((0.5, .5), 0.5)
        return {'wedge': spine}


def sep_to_freq(sep):
    return .5 * 43. / sep

def freq_to_sep(freq):
    return .5 * 43. / freq

def deg_range(array):
    result = []
    for i in array:
        if i <= 180.:
            result.append(i)
        else:
            result.append(i - 360.)
    return np.array(result)


mprojections.register_projection(UpperAitoffAxes)
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Select Data Release 3
Gaia.ROW_LIMIT = -1

freqs = np.array([39.8, 67.7])
min_freq, max_freq = min(freqs), max(freqs)
# colours = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
HWHMs = freq_to_sep(freqs)
min_rad, max_rad = min(HWHMs), max(HWHMs)

pointings = pd.read_csv("coo_targets.csv").set_index("index")
pointings["removed_sources"], pointings["total_sources"] = (np.int32(0) for _
                                                            in range(2))
candidates = pd.DataFrame()

fig, ax = plt.subplots(figsize=(6., 4.), constrained_layout=True)
ax.set_box_aspect(1)
ax.set_xlabel("Right ascension")
ax.set_ylabel("Declination")
cmap = plt.cm.viridis_r
cbar = fig.colorbar(plt.cm.ScalarMappable(
    norm=mpl.colors.Normalize(vmin=.0, vmax=max_rad), cmap=cmap), aspect=7.5)
deg_ticks = np.array([0., .1, .2, .3, .4, .5])
cbar.set_ticks(deg_ticks)
cbar.set_label("Off-centre separation, °")
cax = cbar.ax
cax.set_aspect('auto')
cax.hlines(min_rad, 0., 1., color="r", lw=1., ls="--")
cbar2 = cax.secondary_yaxis("left")
freq_labels = np.array([70, 60, 50, 40], dtype=int)
freq_ticks = freq_to_sep(freq_labels)
cbar2.set_ylim(0., max_rad)
cbar2.set_yticks(freq_ticks)
cbar2.set_yticklabels(freq_labels)
cbar2.set_ylabel("Max frequency, MHz")

largest_num = 0; smallest_num = 1e6

for i in pointings.index:
    r = Gaia.launch_job_async(
        "SELECT source_id, ra, ra_error, dec, dec_error, distance_gspphot, \
            distance_gspphot_upper, distance_gspphot_lower, phot_g_mean_mag, \
            bp_g, g_rp, \
            DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', %s, %s)) AS sep \
        FROM gaiadr3.gaia_source \
        WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), \
                           CIRCLE('ICRS', %s, %s, %s));"
        % (pointings.ra[i], pointings.dec[i], pointings.ra[i],
           pointings.dec[i], max_rad)).get_results().to_pandas()

    remove = []
    for j in r.index:
        if (r.sep[j] + (r.ra_error[j]**2. + r.dec_error[j]**2.)**.5 >=
            max_rad or not
            (r.distance_gspphot_upper[j] - r.distance_gspphot_lower[j]) /
                r.distance_gspphot[j] < .2):
            remove.append(j)
    r = r.drop(index=remove,
               columns=["ra_error", "dec_error", "distance_gspphot_upper",
                        "distance_gspphot_lower"])
    pointings.removed_sources[i] = len(remove)
    pointings.total_sources[i] = len(r)

    if len(r) > largest_num:
        largest_num = len(r)
        print('New largest number of sources: ', largest_num, ' around targert ', pointings['names'][i])
    if len(r) < smallest_num:
        smallest_num = len(r)
        print('New smallest number of sources: ', smallest_num, ' around targert ', pointings['names'][i])
    

    scat = ax.scatter(r.ra, r.dec, c=r.sep, cmap=cmap, s=1., vmin=0.,
                      vmax=max_rad)
    ax.ignore_existing_data_limits = True
    ax.update_datalim(scat.get_datalim(ax.transData))
    ax.scatter(pointings.ra[i], pointings.dec[i], marker="x", color="r")
    ax.autoscale_view()
    ax.set_title("%s" % (pointings['names'][i]))
    fig.savefig("plots/%s-index-%s.png" % (pointings['names'][i], i),
                bbox_inches="tight", dpi = 200, transparent=True)
    scat.remove()

    r.insert(0, "pointing_index", np.int16(i))
    candidates = pd.concat([candidates, r])
    print(i)
del r
plt.clf()

candidates["upper_freq"] = np.minimum(sep_to_freq(candidates.sep), max_freq)
candidates.to_csv("total_candidates.csv", index=False)

candidates = candidates.sort_values(by="sep").groupby("source_id").first(
    ).sort_values(by="pointing_index")
candidates.to_csv("unique_candidates.csv")
pointings["unique_sources"] = np.int32(0)
for i in pointings.index:
    pointings.unique_sources[i] = len(candidates[candidates.pointing_index == i
                                                 ])
pointings.to_csv("pointings.csv")

#%%
fig, ax = plt.subplots(2, 1, constrained_layout=True, sharex=True)
nbins = 100
freq_labels = np.array([60, 50, 40], dtype=int)
freq_ticks = freq_to_sep(freq_labels)
first_hist = np.linspace(0., min_rad, int(.5 + nbins*min_rad/max_rad))
bin_size = first_hist[1]
second_hist = np.arange(min_rad+bin_size, max_rad+bin_size, bin_size)
Y, X = np.histogram(candidates.sep, np.concatenate((first_hist, second_hist)))
X = X[:-1]
colours = [cmap(x/max_rad) for x in X]
ax[0].bar(X, Y, color=colours, width=X[1] - X[0], align="edge")
ax[0].axvline(min_rad, color="r", linestyle="dashed")
ax[0].set_xlim(0., max_rad)
ax[0].set_ylabel("Sources")
ax0 = ax[0].secondary_xaxis("top")
ax0.set_xlim(0., max_rad)
ax0.set_xticks(freq_ticks)
ax0.set_xticklabels(freq_labels)
ax0.set_xlabel("Max frequency, MHz")
ax[1].axvline(min_rad, 0., 1., color="r", linestyle="dashed")
r = candidates.sort_values(by="sep").reset_index().drop(
    columns="index").sep
sep_num = np.empty((2, len(r)-1))
for i in range(len(r)-1):
    sep_num[:, i] = [r[i], i+1]
extra = int(.5 + (max_rad-max(r))/(max(r) / len(r)))
sep_num = np.concatenate((sep_num,
                          np.array([np.linspace(max(r), max_rad, extra),
                                    np.full(extra, len(r))])), axis=1)
s = (len(r)+extra) // 1000
for i in range(0, len(r)+extra, s):
    ax[1].plot(sep_num[0, i:min(i+s+1, len(r)+extra-1)],
               sep_num[1, i:min(i+s+1, len(r)+extra-1)],
               color=cmap(sep_num[0, i]/max_rad))
ax[1].set_xlabel("Off-centre separation, °")
ax[1].set_ylim(bottom=0.)
ax[1].set_ylabel("Sources")
ax1 = ax[1].twinx()
x = np.linspace(0., max_rad, 1000)
ax1.plot(x, np.minimum(sep_to_freq(x), max_freq), color="k")
ax1.set_ylim(bottom=min_freq)
ax1.set_ylabel("Max frequency, MHz")
fig.savefig("Separation distribution.pdf", bbox_inches="tight")
plt.clf()


Y, X = np.histogram(candidates[candidates.upper_freq < max_freq].upper_freq,
                    nbins)
X = X[:-1]
colours = [cmap(freq_to_sep(x)/max_rad) for x in X]
plt.bar(X, Y, color=colours, width=X[1] - X[0], align="edge")
plt.xlim(min_freq, max_freq)
plt.xlabel("Max frequency, MHz")
ax = plt.gca().secondary_xaxis("top")
ax.set_xlim(0., max_rad)
sep_labels = np.array([.35, .4, .45, .5])
ax.set_xticks(sep_to_freq(sep_labels))
ax.set_xticklabels(sep_labels)
ax.set_xlabel("Off-centre separation, °")
plt.ylabel("Sources")
plt.savefig("Max frequency distribution.pdf", bbox_inches="tight")
plt.clf()


fig, ax = plt.subplots(2, 2, constrained_layout=True, sharex="row",
                       sharey="row")
ax[0, 0].set_ylabel("Sources")
ax[1, 0].set_ylabel(r"$M_G$")
ax[1, 0].invert_yaxis()
nbins = 250
dist_hist_colours = ["r", cmap(1.)]
titles = ["Entire frequency range", "All sources"]
H = np.empty((2, nbins, nbins))
xedges, yedges = (np.empty((2, nbins+1)) for _ in range(2))
for i, freq in enumerate([max_freq, min_freq]):
    r = candidates[candidates.upper_freq >= freq]
    ax[0, i].hist(r.distance_gspphot,
                  bins=np.logspace(np.log10(min(r.distance_gspphot)),
                                   np.log10(max(r.distance_gspphot)), nbins),
                  color=dist_hist_colours[i])
    ax[0, i].set_xlim(min(r.distance_gspphot), max(r.distance_gspphot))
    ax[0, i].set_xscale("log", base=10)
    ax[0, i].set_xlabel("Distance, pc")
    ax[0, i].set_title(titles[i])

    H[i], yedges[i], xedges[i] = np.histogram2d(r.phot_g_mean_mag,
                                                r.bp_g - r.g_rp, bins=nbins)
    H[i] = np.ma.masked_where(H[i] == 0, H[i])
del r
cmap2 = plt.cm.plasma.copy()
cmap2.set_bad(color="white")
for i in range(2):
    ax[1, i].pcolormesh(xedges[i], yedges[i], H[i], cmap=cmap2,
                        norm=colors.LogNorm(vmin=1., vmax=np.amax(H)))
    ax[1, i].set_xlabel(r"$G_{BP} - G_{RP}$")
fig.colorbar(plt.cm.ScalarMappable(
    norm=colors.LogNorm(vmin=1., vmax=np.amax(H)), cmap=cmap2))
fig.savefig("Distance distributions and HR diagrams.pdf")
plt.clf()


fig = plt.figure(constrained_layout=True)
gs = GridSpec(2, 2, figure=fig)
phi = np.linspace(0., 2.*np.pi, 50)
cmap2 = plt.cm.autumn
axs = []
for i, rad in enumerate([min_rad, max_rad]):
    axs.append(fig.add_subplot(gs[0, i], projection="upper_aitoff"))
    axs[i].grid(True)
    axs[i].tick_params(axis='both', which='major', labelsize=5)
    for j in pointings.index:
        axs[i].plot(
            (deg_range([pointings.ra[j]]) + rad*np.cos(phi))*np.pi/180.,
            (pointings.dec[j] + rad*np.sin(phi))*np.pi/180., color="b", lw=.1)
axs.append(fig.add_subplot(gs[1, :], projection="upper_aitoff"))
axs[2].grid(True)
axs[2].scatter(deg_range(pointings.ra)*np.pi/180.,
               deg_range(pointings.dec)*np.pi/180., c=pointings.total_sources,
               cmap=cmap2, norm=colors.LogNorm(
                   vmin=min(pointings.total_sources),
                   vmax=max(pointings.total_sources)), s=5.)
cbaxes = inset_axes(axs[2], width="100%", height="25%", loc=8)
cbar = fig.colorbar(plt.cm.ScalarMappable(
    norm=colors.LogNorm(vmin=min(pointings.total_sources),
                        vmax=max(pointings.total_sources)), cmap=cmap2),
                    cax=cbaxes, orientation="horizontal", label="Sources")
cbar2 = cbar.ax.secondary_xaxis("bottom")
labels = np.array([min(pointings.total_sources), max(pointings.total_sources)],
                  dtype=int)
cbar2.set_xticks(labels)
cbar2.set_xticklabels(labels)
fig.savefig("Aitoff projections.pdf", bbox_inches="tight")
plt.clf()


print("%s candidates filtered" % sum(pointings.removed_sources))
print("%s total candidates" % sum(pointings.total_sources))
print("%s unique candidates" % sum(pointings.unique_sources))

# %%
