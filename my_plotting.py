import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator
from my_functions import *
import corner
import itertools
from scipy.optimize import minimize
from IPython.display import display, Math

#colors  = ['orange', 'purple', 'yellow', 'pink', 'red','cyan', 'maroon', 'teal', 'green']
colors = sns.color_palette("Paired")
plt.rcdefaults()
plt.rcParams.update({'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'axes.labelsize': 40,
                     'axes.titlesize': 40,
                     'legend.fontsize': 14,
                     'legend.title_fontsize': 14,
                     #'figure.autolayout': True,
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',})

def plot_1corner(theta, param_names, color_ = 'b'):
    fig = corner.corner(
        theta,
        labels = param_names,
        weights= np.ones(len(theta))/len(theta),
        #quantiles=[0.0, 0.5, 1.00],
        #show_titles = True, # 
        #title_fmt = '.3f',
        #title_kwargs={"fontsize": 12}, 
        color = color_,
        bins = 30,
        smooth1d = 1.5,
        smooth = 1.8,
        verbose = False,
        plot_density = False,
        plot_datapoints = False,
        fillcontours = False)
    return fig, corner

def plot_corner_tocompare(theta_tocompare, fig, corner, color_ = 'r'):
    corner.corner(theta_tocompare,
                  weights= np.ones(len(theta_tocompare))/len(theta_tocompare),
                  fig = fig,
                  bins = 30,
                  #show_titles = True, # 
                  #title_fmt = '.3f',
                  #title_kwargs={"fontsize": 12}, 
                  color = color_,
                  #hist_kwargs = {'linestyle': '-.'},
                  #hist2d_kwargs = {'contourf_kwargs': {'linestyle': '--'}},
                  smooth1d = 1.5,
                  smooth = 1.8,
                  verbose = False,
                  plot_density = True,
                  plot_datapoints = False,
                  fillcontours = True,);

def plot_model_vs_exp(q2s, ss, model_values, exp_df, exp_err, title_ = None, splots = 1, correlated = False):
    fig, ax = plt.subplots(1,splots,  figsize = (8,6), sharey = True, sharex = True)
    
    for j in range(len(q2s)):
        q2 = q2s[j]
        Q2_region = (exp_df['Qs2'] == q2) & (exp_df['sqrt(s)'] == ss)
        Q2_indeces = exp_df.index[Q2_region].tolist()
        exp_df_region = exp_df[Q2_region]
        dat = np.array(exp_df_region['sigma_r'])
        dat_err_cons = np.sqrt(exp_err.diagonal()) if correlated == True else exp_err
        dat_err = dat_err_cons[Q2_indeces]
        xb = np.array(exp_df_region['xbj'])
 
        for i in range(len(model_values)):
            model = [ model_values[i,qq2] for qq2 in Q2_indeces]
            if i == 0:
                ax.plot(xb, model, alpha = 0.8, linewidth = 1.0, color = colors[j], label = "${}$".format((q2)))
            else:
                ax.plot(xb, model, alpha = 0.5, linewidth = 0.3, color = colors[j])
                
        if j == 0:
            ax.errorbar(xb, dat, yerr = dat_err, color = 'black', fmt = '.', label = "Data")        
        else:
            ax.errorbar(xb, dat, yerr = dat_err, color = 'black', fmt = '.')


    ax.set_xlabel("$x_{bj}$")
    ax.set_ylabel("$\sigma_r$ (mb)")
    # set y and x label font sizes
    ax.set_xscale('log')
    ax.set_xlim = (10e-6, 10e-1) 
    ax.set_ylim = (0.4, 1.6)
    ax.set_title(str(title_))
    return fig, ax

def plot_exp_vs_map_vs_median(q2s, ss, map_values, exp_df, exp_err, median_values, splots = 1, correlated = False):
    fig, ax = plt.subplots(1,splots, figsize = (8,6), sharey = True, sharex = True)
    for j in range(len(q2s)):
        q2 = q2s[j]
        Q2_region = (exp_df['Qs2'] == q2) & (exp_df['sqrt(s)'] == ss)
        Q2_indeces = exp_df.index[Q2_region].tolist()
        exp_df_region = exp_df[Q2_region]
        dat = np.array(exp_df_region['sigma_r'])
        dat_err_cons = np.sqrt(exp_err.diagonal()) if correlated == True else exp_err
        dat_err = dat_err_cons[Q2_indeces]
        xb = np.array(exp_df_region['xbj'])

        for i in range(len(map_values)):
            model = [map_values[i,qq2] for qq2 in Q2_indeces]
            ax.plot(xb, model, '--', alpha = 0.9, color = colors[j], label = "MAP estimates")

        for i in range(len(median_values)):
            model = [median_values[i,qq2] for qq2 in Q2_indeces]
            ax.plot(xb, model, alpha = 0.5, color = colors[j], label = " Posterior median")

    # for i in range(len(median_values_low)):
    #     modellow = [median_values_low[i,qq2] for qq2 in Q2_indeces]
    #     modelhigh = [median_values_high[i,qq2] for qq2 in Q2_indeces]
    #     plt.fill_between(xb, modellow, modelhigh, alpha = 0.4, label = "{} credible region".format(confidence))
    
        ax.errorbar(xb, dat, yerr = dat_err, color = 'black', fmt = '.', label = "HERA Data")
    
    #plt.title(r"$Q^2 = {} $".format((q2)) + r" GeV$^2$; $\sqrt{s}$" + " = {} GeV".format(ss))   
    ax.set_xlabel("$x_{bj}$")
    ax.set_ylabel("$\sigma_r$")
    ax.set_xscale('log') 
    #plt.legend()
    return plt.show()

def plot_model_vs_exp_wtrain(q2, ss, model_values, exp_df, exp_err, training_set_all, splots = 2, correlated = False):
    fig, ax = plt.subplots(1,splots, figsize = (16, 6), sharey = True, sharex = True)
    Q2_region = (exp_df['Qs2'] == q2) & (exp_df['sqrt(s)'] == ss)
    Q2_indeces = exp_df.index[Q2_region].tolist()
    exp_df_region = exp_df[Q2_region]
    dat = np.array(exp_df_region['sigma_r'])
    dat_err_cons = np.sqrt(exp_err.diagonal()) if correlated == True else exp_err
    dat_err = dat_err_cons[Q2_indeces]
    xb = np.array(exp_df_region['xbj'])
    

    for i in range(len(model_values)):
        model = [model_values[i,qq2] for qq2 in Q2_indeces]
        ax[1].plot(xb, model, alpha = 0.3, color = 'orange',linewidth = 0.5)
        if i == len(model_values)-1:
            ax[1].plot(xb, model, alpha = 0.9, color = 'orange', linewidth = 0.5, label = "Posterior Samples") 
    
    ax[1].plot(np.average(model))
        
    for i in range(len(training_set_all)):
        train = [training_set_all[i,qq2] for qq2 in Q2_indeces]
        ax[0].plot(xb, train, alpha = 0.5, color = 'orange', linewidth = 0.5)
        if i == len(training_set_all)-1:
            ax[0].plot(xb, train, alpha = 0.9, color = 'orange', linewidth = 0.5, label = "Training Data")

    for i in range(splots):
        ax[i].errorbar(xb, dat, yerr = dat_err, color = 'black',  fmt = '.', alpha = 0.7, label = "HERA Data")
        ax[i].set_title("$Q^2$ = {} GeV$^{}$; ".format(q2, 2) + "$\sqrt{s}$ = " + str(ss) + " GeV", fontsize = 20)
        ax[i].set_xlabel("$x_{bj}$")
        ax[i].set_ylabel("$\sigma_r$")
        ax[i].set_xscale('log') 
        ax[i].legend()
    return fig, ax

def scatter_hist(a, b, ax, ax_histx, ax_histy, params_all, param_names):
    x = params_all[:,a]
    y = params_all[:,b]
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.scatter(x, y, color = 'b')
    ax.set_xlabel(param_names[a])
    ax.set_ylabel(param_names[b])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax_histx.hist(x, bins=10, alpha=0.7, color = 'b')
    ax_histy.hist(y, bins=10, orientation='horizontal', alpha=0.7, color = 'b')

def plot_validation_perkp(kp, pred, pred_err, training_set_all, exp, n_params, params_all, param_names):

    fig, ax = plt.subplots(1, n_params, sharey = True, figsize=(16,4))
    for i in range(0,n_params):
        ax[i].errorbar(params_all[:,i], pred[:,kp], fmt = '.', yerr = pred_err[:,kp], label = "GPE Prediction on Train")
        ax[i].plot(params_all[:,i], pred[:,kp], '.', color = 'b', label = "GPE Prediction on Train")
        ax[i].plot(params_all[:,i], training_set_all[:,kp], 'rx', label = 'Test')
        ax[i].axhline(exp[kp],  color = 'r', linestyle = '--', label = 'HERA value')
        ax[i].set_title(param_names[i])
    
    return fig, ax

def plot_diagonal(pred, true):
    fig, ax = plt.subplots(1,1, figsize = (8,6))
    diag = np.linspace(0.0, np.max(pred) + 0.3, 100)
    ax.plot(diag, diag, color = 'black', linestyle = '--', alpha = 0.5)
    # generate 100 random numbers between 0 and 403
    #ind = np.random.randint(0, 403, 100)
    for i in range(403):
        ax.plot(true[:,i], pred[:,i], '.', color = 'g', alpha = 0.7, rasterized=True)

    ax.set_xlabel("Model $\sigma_r$")
    ax.set_ylabel("Emulator $\sigma_r$")
    return fig, ax

def plot_diagonal_1(pred, true, color_ = 'g', label_ = None):
    fig, ax = plt.subplots(1,1, figsize = (8,6))
    diag = np.linspace(0.4, np.max(pred) + 0.1, 100)
    ax.plot(diag, diag, color = 'black', linestyle = '--', alpha = 0.5)
    ax.plot(pred, true, 'x', color = color_, alpha = 0.7, label = label_)
    ax.set_xlabel("Model Calculation")
    ax.set_ylabel("Experimental Data")
    return fig, ax

def fit_gaussian(to_fit):
    from scipy.stats import norm
    mu, std = norm.fit(to_fit)
    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    gauss = norm.pdf(x, mu, std)
    return x, gauss

def plot_zscore(pred, true, sd, bins_ = 30, text_x = 0.05, text_y = 0.95): 
    from scipy.stats import norm
    fig, ax = plt.subplots(1,1, figsize = (8,6))
    z = np.array([(pred[:,kp] - true[:,kp])/sd[:,kp] for kp in range(403)])

    # fit gaussian to z
    x_fit, gauss_fit = fit_gaussian(z.flatten())

    # target gaussian
    mu = 0
    variance = 1
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    gauss = norm.pdf(x, mu, sigma)

    # plot
    ax.hist(z.flatten(), bins = bins_, density = True, color = 'g', alpha = 0.7, label = "Emulator")
    ax.plot(x_fit, gauss_fit, color = 'g', alpha = 0.7, linewidth = 2, linestyle = '--')
    ax.plot(x, gauss, color = 'black', linewidth = 2, linestyle = '--', label = "Target")
    ax.text(text_x, text_y, "Mean = {:.3f}\nSd = {:.3f}".format(np.mean(z.flatten()), np.std(z.flatten())), transform=ax.transAxes, fontsize = 22)
    ax.set_xlabel("z-score")
    ax.legend()
    return fig, ax

def display_median(paramsamples, param_names):

    for i in range(len(param_names)):
        mcmc = np.percentile(paramsamples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], param_names[i])
        return display(Math(txt))

def display_MAP(paramsamples, param_names, l_bounds, u_bounds, emulators, exp, exp_err, correlated = False):
    posterior_median = np.median(paramsamples, axis = 0)
    for i in range(len(param_names)):
        MAP = minimize(lambda theta: -log_posterior(theta, l_bounds, u_bounds, emulators, exp, exp_err, correlated = correlated), posterior_median);
        txt = "\mathrm{{{1}}} = {0:.3f}"
        txt = txt.format(str(MAP.x), param_names[i])
        return display(Math(txt))

# symmetric treatment of uncertainty band
def get_posterior_mean_and_std(xb, q2, ss, model_values, exp_df):
    xb_region  = (exp_df['xbj'] == xb) & (exp_df['Qs2'] == q2) & (exp_df['sqrt(s)'] == ss)
    xb_index = exp_df.index[xb_region].tolist()
    model_values_for_each_xb = model_values[:,xb_index]
    return np.mean(model_values_for_each_xb, axis = 0), np.std(model_values_for_each_xb, axis = 0)

def plot_posterior_mean_and_ub(q2s, ss, model_values, exp_df, exp_err, title_ = None, legend1_loc = "upper right", correlated = False):
    fig, ax = plt.subplots(1,1, figsize = (8,6))
    mean_line = Line2D([0], [0], color='black')
    mean_patch = mpatches.Patch(color='gray', alpha = 0.8)
    #map_line = Line2D([0], [0], color='black', linestyle = ':')
    handles_ = [(mean_patch, mean_line),]

    for j in range(len(q2s)):
        q2 = q2s[j]
        Q2_region = (exp_df['Qs2'] == q2) & (exp_df['sqrt(s)'] == ss)
        Q2_indeces = exp_df.index[Q2_region].tolist()
        exp_df_region = exp_df[Q2_region]
        dat = np.array(exp_df_region['sigma_r'])
        dat_err_cons = np.sqrt(exp_err.diagonal()) if correlated == True else exp_err
        dat_err = dat_err_cons[Q2_indeces]
        xb = np.array(exp_df_region['xbj'])

        sr_mean = []
        sr_std = []
        for xbj in xb:
            sr_mean_, sr_std_ = get_posterior_mean_and_std(xbj, q2, ss, model_values, exp_df)
            sr_mean.append(sr_mean_)
            sr_std.append(sr_std_)
        
        sr_mean = np.array(sr_mean)
        sr_std = np.array(sr_std)
        up_std = np.array(sr_mean + 2*sr_std).reshape((-1,))
        down_std = (sr_mean - 2*sr_std).reshape((-1,))
        plt_dat = ax.errorbar(xb, dat, yerr = dat_err, color = 'black', fmt = '.', capsize = 3.0)#, label = "Data")  
        
        if j == 0:
            handles_.append(plt_dat)
        
        plt_mean, = ax.plot(xb, sr_mean, '-', color = colors[j], label  = "{}".format(q2), lw = 1.0)
        plt_std = ax.fill_between(xb, up_std, down_std, alpha = 0.5, color = colors[j])#, label = "2 sigma band")
        
        #ax.legend(handles = [(plt_mean, plt_std)], labels =["mean + uncertainty"], title = "$Q^2$ (GeV$^2$)", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # for i in range(len(map_values)):
        #     model = [map_values[i,qq2] for qq2 in Q2_indeces]
        #     plt.plot(xb, model, ':', alpha = 0.8, color = 'black')
    
    legend_1 = plt.legend(loc = legend1_loc, fontsize = 13)#, loc='lower left', borderaxespad=0.  )
    legend_1.set_title(title = "$Q^2$ (GeV$^2$)", prop = {'size': 13})
    legend_2 = plt.legend(handles= handles_, 
                          labels = ["Posterior Mean$\pm 2\sigma$", "HERA data"], 
                          loc = "upper left",
                          fontsize = 13) #,  bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize = 14)
    ax.add_artist(legend_1)
    ax.add_artist(legend_2)
    ax.set_xlabel("$x_{bj}$")
    ax.set_ylabel("$\sigma_r$")
    ax.set_xscale('log')
    x_major = LogLocator(base = 10.0, numticks = 5)
    ax.xaxis.set_major_locator(x_major)
    ax.set_title(str(title_))
    return fig, ax, 


def plot_corner(mve_samples, mv5_samples):
    hm = [0.06, 18.9, 7.2, 16.36, 1.0]
    param_names = [r"$Q_{s0}^{2}$ [GeV²]",
               r"$e_c$",
               r"$C^{2}$",
               r"$\sigma_0/2$ [mb]",
               r"$\gamma$",] 
    from scipy.ndimage import gaussian_filter
    fig_kw = {'linewidth': 2.0}
    fig, axes = plt.subplots(5,5, 
                             figsize = (18,18),
                             constrained_layout = False,
                             **fig_kw)
    plt.subplots_adjust(wspace = 0.07, hspace = 0.07)

    for i,j in itertools.product(range(5), range(5)):
        axes[i,j].tick_params(axis='both', 
                              which='major',
                              direction = 'out', 
                              labelsize = 18, 
                              size = 8, 
                              width = 2.5, 
                              pad = 1.0,
                              )
        axes[i,j].spines[['left', 'right', 'top', 'bottom']].set_linewidth(2.0)


    # axes limits
    range0 = [0.045, 0.11]
    range1 = [2.0, 60.0]
    range2 = [2.0, 8.5]
    range3 = [11.8, 16.6]
    range4 = [0.95, 1.08]
    xranges = np.array([range0, range1, range2, range3, range4])
    hist_plot_kwargs= {'linewidth': 2.5, 'drawstyle': 'steps-mid'}
    hist_kwargs = {'density': True,
                   'bins': 30}
    hist2d_kwargs = {"bins" : 30, 
                     "smooth" : 1.7,
                     "plot_datapoints" : True,
                     "plot_density":True,}
    
    for i in range(5):
        axes[i,i].tick_params(which='major', labelrotation=35)
        if i == 4:
            n_mv5, bins_mv5 = np.histogram(mv5_samples[:,i], **hist_kwargs)
            n_mv5 = gaussian_filter(n_mv5, sigma = 1.5)
            x0 = np.array(list(zip(bins_mv5[:-1], bins_mv5[1:]))).flatten()
            y0 = np.array(list(zip(n_mv5, n_mv5))).flatten()
            axes[i,i].plot(x0, y0, color = 'b', **hist_plot_kwargs)
            
            axes[i,i].set_xlim(xranges[i])
            axes[i,i].set_ylim([0.0,None])
            axes[i,i].axvline(1.0, color = 'r', linestyle = '-', linewidth = 3.0)

        else:
            n_mv5, bins_mv5 = np.histogram(mv5_samples[:,i], **hist_kwargs)
            n_mv5 = gaussian_filter(n_mv5, sigma = 1.5)
            x0 = np.array(list(zip(bins_mv5[:-1], bins_mv5[1:]))).flatten()
            y0 = np.array(list(zip(n_mv5, n_mv5))).flatten()
            axes[i,i].plot(x0, y0, color = 'b', **hist_plot_kwargs)
            
            n_mve, bins_mve = np.histogram(mve_samples[:,i],**hist_kwargs)
            n_mve = gaussian_filter(n_mve, sigma = 1.5)
            x02 = np.array(list(zip(bins_mve[:-1], bins_mve[1:]))).flatten()
            y02 = np.array(list(zip(n_mve, n_mve))).flatten()
            axes[i,i].plot(x02, y02, color = 'r', **hist_plot_kwargs)
            axes[i,i].set_xlim(xranges[i])
            axes[i,i].set_ylim([0.0 , None])

        for j in range(i):
            corner.hist2d(mve_samples[:,i], mve_samples[:,j], ax = axes[j,i], color = 'r', **hist2d_kwargs, contour_kwargs = {'linewidths':2.0})
            axes[j,i].set_xlim(xranges[i])
            axes[j,i].set_ylim(xranges[j])
            axes[j,i].tick_params(which='major', labelrotation=35)
            corner.hist2d(mv5_samples[:,j], mv5_samples[:,i], ax = axes[i,j], color = 'b', **hist2d_kwargs, contour_kwargs = {'linewidths':2.0})
            axes[i,j].set_xlim(xranges[j])
            axes[i,j].set_ylim(xranges[i])
            axes[i,j].tick_params(which='major', labelrotation=35)
        
    for i in range(5):
        axes[i,i].axvline(hm[i], color = 'g', linestyle = ':', linewidth = 3.0)    
    
    for i in range(1,3):
        for j in range(5):
            axes[j,i].set_yticklabels([])
            axes[j,i].tick_params(which='major', axis = 'y', size = 0)

    axes[4,3].set_yticklabels([])
    axes[4,3].tick_params(which='major', axis = 'y', size = 0)
    axes[4,4].set_yticklabels([])
    axes[4,4].tick_params(which='major', axis = 'y', size = 0)
    axes[0,0].set_yticklabels([])
    axes[0,0].tick_params(which='major', axis = 'y', size = 0)

    for i in range(1,5):
        axes[i,0].set_ylabel(param_names[i], fontsize = 24)
        axes[i-1,3].set_ylabel(param_names[i-1], fontsize = 24)
        #axes[i-1,3].set_yticklabels([])
        axes[i-1,3].yaxis.set_label_position("right")
        axes[i-1,3].yaxis.tick_right()
        #axes[0,3].set_ylabel(param_names[0])
        #axes[0,3].yaxis.set_label_position("right")
        fig.delaxes(axes[i-1,4])
    
    # manually removing and arranging tick positions and axis labels
    for i in range(5):
        axes[4,i].set_xlabel(param_names[i], fontsize = 24)
        axes[0,i].set_xlabel(param_names[i], fontsize = 24)
        axes[0,i].xaxis.set_label_position("top")
        axes[0,i].xaxis.tick_top()
        for j in range(1,4):
            axes[j,i].set_xticklabels([])
            axes[j,i].tick_params(which='major', axis = 'x', size = 0)
    
    axes[3,3].set_ylabel("")
    axes[3,3].set_yticklabels([])
    axes[3,3].tick_params(which='major', axis = 'y', size = 0)
    return fig, axes

def plot_pred_mve_vs_mv5(mve_values, mv5_values, x, ylabel, xlabel, title_ = "", legend_loc = "lower right", xlogscale = False, ylogscale = False, linewidth_ = 2):
    ''' 
    Plot initial dipole shape or 2DFT 
    Input: mve_values, a list of mean, upper sd, and lower sd values for the mve model
           mv5_values, a list of mean, upper sd, and lower sd values for the mv5 model    
    '''
    mve_usd = mve_values[0] + 2*mve_values[1]
    mve_dsd = mve_values[0] - 2*mve_values[2]
    mv5_usd = mv5_values[0] + 2*mv5_values[1]
    mv5_dsd = mv5_values[0] - 2*mv5_values[2]

    fig, ax = plt.subplots(1,1, figsize = (8,6))
    ax.plot(x, mv5_values[0], '--',linewidth = linewidth_, color = "b")
    ax.fill_between(x, mv5_usd, mv5_dsd, alpha = 0.6, color = "b")
    ax.plot(x, mve_values[0], '--',linewidth = linewidth_, color = "r")
    ax.fill_between(x, mve_usd, mve_dsd, alpha = 0.6, color = "r")

    blue_line = Line2D([0], [0], color='b', linestyle='--', linewidth=2)
    blue_patch = mpatches.Patch(color='b', alpha = 0.6)
    red_line = Line2D([0], [0], color='r', linestyle='--', linewidth=2)
    red_patch = mpatches.Patch(color='r', alpha = 0.6)
    handles_ = [(blue_patch, blue_line),(red_patch, red_line), ]
    ax.legend(handles= handles_, 
           labels = ["[$ Q_{s0}^{2}, e_{c}, C^{2}, \sigma_{0}/2, \gamma$]", "[$ Q_{s0}^{2}, e_{c}, C^{2}, \sigma_{0}/2$]"], 
           loc = legend_loc)
    ax.set_title(title_)
    ax.set_xscale("log") if xlogscale else None
    ax.set_yscale("log") if ylogscale else None
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return fig, ax