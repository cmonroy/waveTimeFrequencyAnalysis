

from droppy.TimeDomain.srs import ShockResponseSpectrum

data = pd.Series( index = np.linspace( 0, 10, 100 ), data = 2*np.sin( np.linspace(0, 3.14, 100) )**2 )


srs = ShockResponseSpectrum(data)

fn_array = np.logspace(-2,2,50)

res = srs.run_srs_analysis( fn_array )
res_check = srs.run_srs_analysis_ode( fn_array )

fig, ax = plt.subplots()
res.POS.plot(ax=ax, label = "libs")
res_check.POS.plot(ax=ax, label = "manual")
ax.legend()
ax.set_xscale("log")



fig, ax = plt.subplots()
res.NEG.plot(ax=ax, label = "libs")
res_check.NEG.plot(ax=ax, label = "manual")
ax.legend()
ax.set_xscale("log")
