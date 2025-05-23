import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from emcee import EnsembleSampler
import pandas as pd
import scipy as sp
import pickle
from hankel import HankelTransform
from scipy import interpolate, integrate
from scipy.optimize import newton

def ReadBKDipole(thefile):  

    '''Read the dipole amplitude from the given datafile produced running Heikki's BK code
    
    Returns an interpolator for the dipole: N(Y, r), where r is in GeV^-1, and x = x_0*exp(Y)
    
    Note: as this interpolates in r and not in log r, at very small r there are some small interpolation errors
    '''
    with open(thefile) as f:
        content = f.read().split("###")
    
    content = content[1:]   # gets rid of the stuff at the beginning
    content = [i.split() for i in content] # cleans up lines
    NrY_data = []
    pars = []
    for i in content:
        '''Separates and sorts the lines in the file
        
        Takes values in the beginning of the file to 'pars' list
        and every Y value with associated N(Y,r) values to 'NrY_data' list
        '''

        x = list(map(float, i))
        if len(x) == 1:
            pars.append(x)
        else:
            NrY_data.append(x)

        
    rmYs = np.array(NrY_data).T[1:]     # removes Y values
    N_values = rmYs.T
    Y_values = np.array(NrY_data).T[0]

    pars = np.ndarray.flatten(np.array(pars))
    minr = pars[0]
    mult = pars[1]
    n = int(pars[2])
    r_values = np.array([minr*mult**i for i in range(n)])
    
    
    rgrid=[]
    ygrid=[]
    for y in Y_values:
        for r in r_values:
            rgrid.append(r)
            ygrid.append(y)
    
    interpolator = interpolate.CloughTocher2DInterpolator((ygrid, rgrid), N_values.flatten(), fill_value=0)
    
    return interpolator

# def my_chi2(data, obs, obs_err, npts = 403):
#     return np.sum(((data - obs)**2)/obs_err**2)/npts

def my_cut_df(l_limit, u_limit, basis): # for pandas dataframes only
    indeces = basis.index[(basis >= l_limit) & (basis <= u_limit)].to_list()
    return indeces

def my_cut_array(l_limit, u_limit, basis):
    indeces = np.where((basis >= l_limit) & (basis <= u_limit))[0]
    return indeces

# def get_exp_experr_xb(boolean, dataframe):
#     return np.array(dataframe['sigma_r'][boolean]), np.array(dataframe['error'][boolean]), np.array(dataframe['xbj'][boolean])

def get_cor_columns():
    
    ''' Simply returns a list of the correlated uncertainties column names'''

    cor_sys = []
    for i in range(1,10):
        cor_sys.append('sysHZComb100{}'.format(i))

    for i in range(10,100):
        cor_sys.append('sysHZComb10{}'.format(i))

    for i in range(100,163):
        cor_sys.append('sysHZComb1{}'.format(i))

    list_proc = ['proc_nrl', 'proc_tb21', 'proc_tb22', 'proc_tb23', 'proc_tb24', 'proc_gp', 'proc_had']
    for i in range(163,170):
        cor_sys.append(list_proc[i-163])

    return cor_sys

# construct error covariance matrix from data uncertainties: systematic + statistical and correlated
def construct_cov(exp_df): # take labelled exp_df

    ''' Function that returns the relative covariance matrix of the experimental data
        Input: exp_df, pandas dataframe of experimental data with columns'''

    uncor_err = exp_df['uncor_tot'] 
    cor_err = np.array(exp_df[get_cor_columns()]) # shape : 403 x 169
    nkp = len(exp_df) # number of kinematical points
    nunc = np.shape(cor_err)[1] # number of uncertainties
    C = np.zeros( (nkp, nkp) ) # initialize
    for i in range(nkp): # ith kinematical point
        C[i][i] = uncor_err[i]**2 + np.sum([cor_err[i][k]**2 for k in range(nunc)]) # add uncorrelated and correlated errors in diagonal
        for j in range(i):
            C[i][j] = np.sum([cor_err[i][k]*cor_err[j][k] for k in range(nunc)])
            C[j][i] = C[i][j] 
    return C 

def load_exp(filename, Q2_llimit = 2.0, Q2_ulimit = 50.0, correlated = False):

    ''' Function that returns all necessary info from the experimental data
        Input: filename, string of the filename of the experimental data
        Output: exp_df, pandas dataframe of experimental data
                exp, numpy array of experimental data
                exp_err * 0.01 * exp if correlated = False, numpy array of absolute uncertainties (diagonal of covariance matrix))
                exp_cov * 0.01 * exp if correlated = True, numpy array of covariance matrix with shape nkp x nkp
                xbj, numpy array of xbj values'''

    exp_df = pd.DataFrame(pd.read_csv(filename)) # file should have relative uncertainties in percent not absolute
    Q2_region = (exp_df['Qs2'] >= Q2_llimit) & (exp_df['Qs2'] <= Q2_ulimit) # cuts of Q2 region
    exp_df = exp_df[Q2_region]
    exp = np.array(exp_df['sigma_r'])
    xbj = np.array(exp_df['xbj'])

    # convert all uncertainties to absolute uncertainties
    exp_df['stat'] = exp_df['stat'] * 0.01 * exp
    exp_df['uncor'] = exp_df['uncor'] * 0.01 * exp
    exp_df['ignore'] = exp_df['ignore'] * 0.01 * exp # total correlated uncertainties without procedural
    exp_df[get_cor_columns()] = exp_df[get_cor_columns()] * 0.01 * exp[..., None] 

    # add total uncorrelated and correlated uncertainties column to dataframe
    exp_df['uncor_tot'] = np.sqrt((exp_df['stat'])**2 + (exp_df['uncor'])**2) # total uncorrelated errors
    exp_df['cor_wo_proc'] = np.sqrt((exp_df['ignore'])**2 - (exp_df['uncor_tot'])**2) # total correlated uncertainties without procedural
    cor_err = np.array(exp_df[get_cor_columns()])
    exp_df['cor_tot'] = np.sqrt(np.sum(cor_err**2, axis = 1)) # total correlated uncertainties 
    if correlated == False:
        #exp_err = np.sqrt((exp_df['ignore'].values)**2)
        # to include procedural: 
        exp_err = np.sqrt((exp_df['cor_tot'].values)**2 + (exp_df['uncor_tot'].values)**2)
        return exp_df, exp, exp_err, xbj
    if correlated == True:
        return exp_df, exp, construct_cov(exp_df), xbj # returns covariance


def load_training_data(train_file, theta_file):
    train = np.loadtxt(train_file)
    theta = np.vstack(np.loadtxt(theta_file, unpack = True)).T # makes each row a single array = 1 parameter vector
    return train, theta

#  nuisance parameters
def get_A(exp_df):

    ''' Function that returns the A matrix for nuisance paramater profiling'''

    uncor_err2 = exp_df['uncor_tot'].values**2 # uncorr_err2 = ((exp_df['stat']*0.01*exp)**2 + (exp_df['uncor']*0.01*exp)**2)
    cor_columns = get_cor_columns()
    nunc = len(cor_columns)
    cor_err = exp_df[cor_columns].values
    A = np.zeros((nunc, nunc))
    for k in range(nunc):
        A[k][k] = 1 + ((cor_err[:,k]**2)/ uncor_err2).sum() # sum over kp
        for l in range(k):
            A[k][l] = ((cor_err[:,k] * cor_err[:,l]) / uncor_err2).sum() # sum over kp
            A[l][k] = A[k][l]
    return A

def get_nuisance(exp_df, model):

    ''' Function that returns the nuisance parameters given the data and model
        Input: exp_df, pandas dataframe of experimental data
               model, numpy array of model predictions '''

    D = exp_df['sigma_r'].values
    A = get_A(exp_df)
    uncor_err2 = exp_df['uncor_tot'].values**2 
    cor_err = exp_df[get_cor_columns()].values
    nunc = len(cor_err[0])
    A_inv = np.linalg.inv(A)
    lambdas = np.zeros(nunc)

    for h in range(nunc):
        sumi = []
        for i in range(len(D)):
            DT = D[i] - model[i]
            sumk = np.sum([A_inv[h,k] * cor_err[i][k] / uncor_err2[i] for k in range(nunc)])
            sumi.append(DT * sumk)
        
        lambdas[h] = np.sum(sumi)

    return lambdas

def get_shifted_exp(exp_df, lambdas):

    ''' Shifts data according to D_i^shifted = D_i + sum_h lambda_h * C_ih 
        Input: exp_df, pandas dataframe of experimental data
               lambdas, numpy array of nuisance parameters '''
    
    nunc = len(lambdas)
    exp = exp_df['sigma_r'].values
    corr_err = np.array(exp_df[get_cor_columns()])
    
    exp_shift = np.zeros(len(exp))
    for i in range(len(exp)):
        exp_shift[i] = exp[i] + np.sum([lambdas[h] * corr_err[i][h] for h in range(nunc)])
    
    return exp_shift
    
def evaluate_emu_perpc(training_set_all, params_all, max_pc = 10, correlated = False):
    
    ''' Returns z (mean z score) and rd (mean relative difference) for each number of principal components to evaluate how many principal components to use'''
    
    z = np.zeros(max_pc)
    rd = np.zeros(max_pc)
    ems = []
    prims = []
    for i in range(max_pc):
        prim = i + 2
        prims.append(prim)
        emulators0 = train_PCA_GPE(training_set_all, params_all, primary_components = prim, noise_level_bounds_= (1e-10, 1e1))
        ems.append(emulators0)
        pred_rs, err_rs = return_predictions(ems[i], params_all, correlated = correlated)
        err_rs_diag = np.sqrt(np.diagonal(err_rs, axis1 = 1, axis2 = 2)) if correlated == True else err_rs
        rd_mean = []
        z_mean = []
        for f in range(np.shape(pred_rs)[1]):
            zs = np.mean(np.abs(pred_rs[:,f] - training_set_all[:,f])/err_rs_diag[:,f])
            rds = np.mean(np.abs(pred_rs[:,f] - training_set_all[:,f]))
            z_mean.append(zs)
            rd_mean.append(rds)
        z[i] = np.mean(z_mean)
        rd[i] = np.mean(rd_mean)

    return z, rd, prims


def train_PCA_GPE(training_data, 
                  theta, primary_components = 2, 
                  length_scale_bounds_ = (1e-10, 1e10), 
                  noise_level_bounds_ = (1e-5, 1e5),
                  show_var = False): 
    
    ''' Fit the training data to a PCA and GPE model 
        Input: training_data, numpy array of training data (n_samples x nkp)
               theta, parameter vector (1, n_params) or (n_samples, n_params)
               primary_components, number of principal components to use
               length_scale_bounds_, bounds of the length scale hyperparameter in GPE kernel
               noise_level_bounds_, bounds of the noise level hyperparameter in GPE kernel
               show_var, print explained variance for each principal component'''

    # Scale the data to center around mean and have unit variance
    ss = StandardScaler()
    training_data_scaled = ss.fit_transform(training_data)

    # PCA transform data to new basis (shape = n_samples x nkp to n_samples x npc)
    # Whitening returns a covariance matrix that is unit diagonal
    pca = PCA(n_components = primary_components, whiten = True)
    training_data_PCAd = pca.fit_transform(training_data_scaled)

    if show_var == True:
        #print('Explained variance ratio: {0}'.format(str(pca.explained_variance_ratio_)))
        print('Sum of explained variance ratio for {0} components: {1}'.format(primary_components, str(np.sum(pca.explained_variance_ratio_))))

    # Train and optimize kernel hyperparameter of GPE for each principal component
    gpes = []
    for z in training_data_PCAd.T[:primary_components]:

        # White kernel with very small noise level is added
        kernel = RBF(length_scale = np.ones(theta.shape[1]), 
                         length_scale_bounds = length_scale_bounds_) + WhiteKernel(noise_level_bounds = noise_level_bounds_)
        gpe = GaussianProcessRegressor(kernel = kernel,  n_restarts_optimizer = 5)
        gpes.append(gpe.fit(theta, z))
    
    return gpes, pca, ss 

def savetopickle(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def loadfrompickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def invert_cov(cov, var_trans, nsamples, nkp):
    return np.dot(np.array(cov).T, var_trans).reshape(nsamples, nkp, nkp) # A.T * cov * A 

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# function to check if matrix is hermitian
def is_hermitian(x):
    return np.allclose(x, x.conj())

# function to check is matrix is symmetric
def is_symmetric(x):
    return np.allclose(x, x.T)

def return_predictions(emulators, theta, correlated = False):
    
    '''  Function that returns the predictions of the emulators given parameter vectors 
         Input: emulators, list of TRAINED emulators [gpes, pca, ss]
                theta, parameter vector (1, n_params) or (n_samples, n_params)
                correlated, boolean that determines if the covariance matrix is returned (True) or not (False)'''


    # load the trained emulator, pca and scaler
    gpes = emulators[0]
    pca = emulators[1]
    ss = emulators[2]

    nsamples = len(theta)
    npc = len(gpes) # number of principal components
    nkp = np.shape(ss.scale_)[0] # number of kinematical points
    
    mean_prediction = []
    cov_prediction = []
    for gpe in gpes: # predicts per principal component: len(gpes) = npc 
        mean, cov = gpe.predict(theta, return_cov = True)
        gp_var2 = cov.diagonal()[:, None]
        mean_prediction.append(mean)
        cov_prediction.append(gp_var2)

    # Invert Scaling and PCA transform for mean prediction of emulator
    pred_r = ss.inverse_transform(pca.inverse_transform(np.array(mean_prediction).T))
    # dim_pred = pred_r.shape[0]
    
    # if dim_pred == 1: # if dimension is 1, make it a 1d array
    #    pred_r = pred_r[1]
    
    # Set-up transformation matrix for covariance matrix inversion
    trans_matrix = pca.components_ * np.sqrt(pca.explained_variance_[:, None]) # explained variance is multiplied due to whitening
    A = trans_matrix[:npc] 
    var_trans = np.einsum('ki,kj->kij', A, A, optimize = False).reshape(npc, nkp**2)

    # Add truncation errors
    # GPE covariance matrix is theoretically positive semi definite but due to numerical implementation it is not
    # we compute the cov for the remaining principal components 
    # and add small numbers to the diagonal of the covariance matrix

    B = trans_matrix[npc:]
    var_trans_trunc = np.dot(B.T, B)
    var_trans_trunc.flat[::nkp + 1] += 1e-8 * ss.var_

    # inverse transform diagonal covariance matrix
    cov_rpca = invert_cov(cov_prediction, var_trans, nsamples, nkp)
    cov_scaled = cov_rpca * ss.scale_[:, None]

    if correlated == False:
        std_preds = np.sqrt(np.diagonal(cov_scaled, axis1 = 1, axis2 = 2))
        return pred_r, std_preds
    
    elif correlated == True:
        dim_cov = cov_scaled.shape[0]
        if dim_cov == 1:
            cov_scaled = cov_scaled[0]
        return pred_r, cov_scaled + var_trans_trunc
# log formulas

# log likelihood
def log_likelihood(theta, emulators, data, data_err, correlated = False):

    ''' Function that returns the log of the likelihood '''

    theta_reshaped = theta.reshape(1,-1)
    predict, predict_err = return_predictions(emulators, theta_reshaped, correlated = correlated)
    nkp = np.shape(emulators[2].scale_)[0]
    ide = np.identity(nkp)
    
    if correlated == False:
        err2 =  (predict_err**2 + data_err**2)
        #err2 =  data_err**2
        ll = np.log(2*np.pi*err2) + ((data - predict)**2) / err2
        return -.5*np.sum(ll)
    
    if correlated == True:
        err2 = predict_err + data_err

        # if np.allclose(np.linalg.inv(err2) @ err2, ide) == False:
        #     raise ValueError('E^{-1}E is not equal to I')

        delta_y = (data - predict).reshape(403,1)
        B = np.dot(np.linalg.inv(err2), delta_y) 
        err2_det = np.linalg.slogdet(err2)[1].sum() # log(det(err2))
        ll = np.dot(delta_y.T, B) + err2_det # removed log(2*pi)
        
        return -.5*ll
    
def get_chi2(predict, data, data_err, correlated = False, model = 'mve'):

    ''' Function that returns the log of the likelihood '''

    nkp = 403 #np.shape(emulators[2].scale_)[0]
    ide = np.identity(nkp)
    p = 4 if model == 'mve' else 5
    
    if correlated == False:
        err2 = data_err**2 # when uncorrelated, the exp uncertainties are standard deviations
        ll = ((data - predict)**2) / err2
        return (np.sum(ll)/(nkp - p)).reshape(1)[0]
    
    if correlated == True:
        err2 = data_err
        if np.allclose(np.linalg.inv(err2) @ err2, ide) == False:
            raise ValueError('E^{-1}E is not equal to I')

        delta_y = (data - predict).reshape(nkp,1)
        B = np.dot(np.linalg.inv(err2), delta_y) 
        ll = np.dot(delta_y.T, B) 
        return (ll/(nkp - p)).reshape(1)[0]


def log_flat_prior(theta, l_bounds, u_bounds):
    
    ''' Function that returns the log of the flat prior and is set to infinite if outside bounds'''

    for i in range(np.size(theta)): 
        if theta[i] < l_bounds[i] or theta[i] > u_bounds[i]:
            return -np.inf
       
    return 0

# posterior function
def log_posterior(theta, l_bounds, u_bounds, emulators, data, data_err, correlated = False):

    ''' Function that returns the log of the posterior '''

    return log_likelihood(theta, emulators, data, data_err, correlated = correlated) + log_flat_prior(theta, l_bounds, u_bounds)

# emcee sampler function
def emcee_sampler(n_walkers, n_params, log_posterior, l_bounds, u_bounds, emulators, data, data_err, moves = None, correlated = False):

    ''' Function that initializes the emcee sampler '''

    sampler = EnsembleSampler(n_walkers, 
                              n_params, 
                              log_posterior,
                              moves = moves, #moves = [(DEMove(), 0.8), (DESnookerMove(), 0.2)], 
                              args = [l_bounds, u_bounds, emulators, data, data_err, correlated], 
                              threads = 4) # default is stretchmove
    return sampler

def return_samples(p0, n_samples, n_burn, sampler):

    ''' Function that returns the samples after running the emcee sampler over a number of burn-in steps '''

    burn = sampler.run_mcmc(p0, n_burn, progress = True) 
    sampler.reset() # to remove burn samples
    run = sampler.run_mcmc(burn, n_samples, progress = True)
    return sampler.get_chain(flat = True)


# for 2DFT calculation
# define the function to transform

def dipp(r, Qs02, gamma, e_c): # dipole proton
    lambda_qcd = 0.241 #GeV
    B = ((r**2)*(Qs02))**gamma
    C = (1/(r*lambda_qcd) + e_c * np.exp(1))
    N = 1 - np.exp(-B/4 * np.log(C))
    return N

def rho(z, bt, n, A): # Woods-Saxon distribution
    RA = 1.12*A**(1/3) - 0.86*A**(-1/3) # in fm
    d = 0.54 # in fm (5.068 fm = 1 GeV^-1) 
    rhoA = n / (1 + np.exp((np.sqrt(bt**2 + z**2) + RA)/d))
    return rhoA

def woods_saxon(bt, n, A): # Woods-Saxon distribution T_A, integrated over z
    TA = integrate.quad(rho, -np.inf(), np.inf(), args = (bt, n, A))
    return TA[0]

def woods_saxon_4norm(bt, A): # just the previous function but times 2 pi bt
    TA = integrate.quad(rho, 0.0, 1.0, args = (bt, A))
    return 2 * np.pi * bt * TA[0]

def woods_saxon_norm(A): # getting normalization constant
    ta_norm = integrate.quad(woods_saxon_4norm, 0.0, 10.0, args=(A))
    n = 1  / ta_norm[0]    
    return n

def dipA(r, A, Qs02, sigma0_2, e_c): # dipole nucleus
    lambda_qcd = 0.241 #GeV
    bt = 0.5 
    n = woods_saxon_norm(A)
    TA = woods_saxon(bt, n, A) 
    B = A * TA * (sigma0_2)*(r**2)*(Qs02)
    C = 1/(r*lambda_qcd) + e_c * np.exp(1)
    N = 1 - np.exp(-B/4 * np.log(C))
    return N

def get_2DFT_pA(Qs02s, e_cs, k, A): 
    
    ''' Function that returns 2D Fourier transform of dipole amplitude for pA '''

    ht = HankelTransform(nu = 0, N = 1000, h = 0.001)
    sp = []
    sp_median = []
    sp_sd = []
    for i in k:
        sp_per_k = [ht.transform(lambda r: 1 - dipp(r, A,  Qs02s[j], e_cs[j]), i, ret_err = False) for j in range(len(Qs02s))]
        sp.append(sp_per_k)
        sp_median.append(np.median(sp_per_k))
        sp_sd.append(np.std(sp_per_k))

    return np.array(sp), np.array(sp_median), np.array(sp_sd)

def get_2DFT(Qs02s, gammas, e_cs, k): 

    ''' Function that returns 2D Fourier transform of dipole amplitude for pp '''

    ht = HankelTransform(nu = 0, N = 1000, h = 0.001)
    sp = []
    sp_median = []
    sp_sd = []
    for i in k:
        sp_per_k = [ht.transform(lambda r: 1 - dipp(r, Qs02s[j], gammas[j], e_cs[j]), i, ret_err=False) for j in range(len(Qs02s))]
        sp.append(sp_per_k)
        sp_median.append(np.median(sp_per_k))
        sp_sd.append(np.std(sp_per_k))

    return np.array(sp), np.array(sp_median), np.array(sp_sd)

def get_2DFT_sigma02(sigma02s, sp, k):
    
    ''' 2DFT * sigma0/2'''

    ss = []
    ss_median = []
    ss_sd = [] 
    for i in range(len(k)):
        ss_per_k = [sigma02s[j] * sp[i][j] for j in range(len(sigma02s))]
        ss.append(ss_per_k)
        ss_median.append(np.median(ss_per_k))
        ss_sd.append(np.std(ss_per_k))

    return np.array(ss), np.array(ss_median), np.array(ss_sd)

def get_sd(values, mean, which = 'upper'):
    
    ''' Returns standard deviation of values above or below mean '''

    if which == 'upper':
        region = values > mean
    elif which == 'lower':
        region = values < mean
    
    region_indeces = np.where(region)[0]
    return np.std(values[region_indeces])

def get_2DFT_upsd_downsd(Qs02s, gammas, e_cs, sigma02s, k):
    
    ''' Function that returns 2DFT (pp) mean along with the upper and lower standard deviation '''
    
    ht = HankelTransform(nu = 0, N = 1000, h = 0.001)
    conv_mbGeV2 = 2.56819
    sp_mean = []
    sp_up_sd = []
    sp_down_sd = []
    for i in k:
        sp_per_k = [ht.transform(lambda r: 1 - dipp(r, Qs02s[j], gammas[j], e_cs[j]), i, ret_err=False) for j in range(len(Qs02s))]
        #sp_sigma02_per_k = np.array(sp_per_k) * sigma02s
        sps_per_k = [conv_mbGeV2 * sigma02s[j] * sp_per_k[j] for j in range(len(Qs02s))]
        sps_per_k = np.array(sps_per_k)
        sps_per_k_mean = np.mean(sps_per_k)
        sp_mean.append(sps_per_k_mean)
        sp_up_sd.append(get_sd(sps_per_k, sps_per_k_mean, which = 'upper'))
        sp_down_sd.append(get_sd(sps_per_k, sps_per_k_mean, which = 'lower'))
    
    return np.array(sp_mean), np.array(sp_up_sd), np.array(sp_down_sd)

def get_iBK_upsd_downsd(Qs02s, gammas, e_cs, rs):
    iBK_mean = []
    iBK_up_sd = []
    iBK_down_sd = []
    for r in rs:
        iBK_per_r = [dipp(r, Qs02s[j], gammas[j], e_cs[j]) for j in range(len(Qs02s))]
        iBK_per_r = np.array(iBK_per_r)
        iBK_per_r_mean = np.mean(iBK_per_r)
        iBK_mean.append(iBK_per_r_mean)
        iBK_up_sd.append(get_sd(iBK_per_r, iBK_per_r_mean, which = 'upper'))
        iBK_down_sd.append(get_sd(iBK_per_r, iBK_per_r_mean, which = 'lower'))
    
    return np.array(iBK_mean), np.array(iBK_up_sd), np.array(iBK_down_sd)

def get_eBK_upsd_downsd(where_bk_folder, rs):
    mean = []
    up_sd = []
    down_sd = []

    bk_interpolators = [ReadBKDipole(where_bk_folder + "/{}.dat".format(i)) for i in range(100)]
    x0 = 0.01 # 10⁻2
    xbj = 0.00001 # 10⁻5
    y_int = np.log(x0/xbj)
    for r in rs:
        val_per_r = [bk_interpolators[i](y_int, r) for i in range(100)]
        val_per_r = np.array(val_per_r)
        val_per_r_mean = np.mean(val_per_r)
        mean.append(val_per_r_mean)
        up_sd.append(get_sd(val_per_r, val_per_r_mean, which = 'upper'))
        down_sd.append(get_sd(val_per_r, val_per_r_mean, which = 'lower'))
    
    return np.array(mean), np.array(up_sd), np.array(down_sd)

def get_rpa_upsd_downsd(where_rpa):
    x_q2dep = 0.001 # for info only
    Q2_range = np.geomspace(1, 100, 50) # for info only
    rpa = []
    for i in range(100):
        rpa.append(np.loadtxt(where_rpa + "{}.dat".format(i)))
    
    rpa = np.array(rpa)
    rpa_mean = [] #np.mean(rpa_q2, axis = 0)
    rpa_dsd = np.zeros(50)
    rpa_usd = np.zeros(50)
    for i in range(50):
        mn = np.mean(rpa[:,i])
        rpa_mean.append(np.mean(mn))
        rpa_usd[i] = get_sd(rpa[:,i], mn, which = 'upper')
        rpa_dsd[i] = get_sd(rpa[:,i], mn, which = 'lower')

    return np.array(rpa_mean), np.array(rpa_usd), np.array(rpa_dsd)

def solve_r_qsofr(x, qs02, gamma, e_c):
    lambda_qcd = 0.241
    A = qs02**(gamma)/4
    B = 1/lambda_qcd
    C = np.exp(1)*e_c
    func = A*x**(2*gamma) * np.log(B/x + C) - 1/2
    return func

def solve_r_qsofr_evolved(r, bk_interp, x):
    x0 = 0.01 # 10⁻2
    #xbj = 0.00001 # 10⁻5
    y_int = np.log(x0/x)
    func = bk_interp(y_int, r) - (1 - np.exp(-1/2))
    return func

def get_qs2ofx(where_bk, xs):
    bk_interpolators = [ReadBKDipole(where_bk + "/{}.dat".format(i)) for i in range(100)]
    qs2_mean = []
    qs2_usd = []
    qs2_dsd = []
    for x in xs:
        val_per_x = [newton(solve_r_qsofr_evolved, 1.0, args = (bk_interpolators[i], x)) for i in range(100)]
        val_per_x = np.array(val_per_x)
        qs2 = 2/val_per_x**2
        mean = np.mean(qs2)
        qs2_mean.append(mean)
        qs2_usd.append(get_sd(qs2, mean, which = 'upper'))
        qs2_dsd.append(get_sd(qs2, mean, which = 'lower'))
    return np.array(qs2_mean), np.array(qs2_usd), np.array(qs2_dsd)