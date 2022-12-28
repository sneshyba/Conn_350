import numpy as np
import h5io
import matplotlib.pyplot as plt
from copy import deepcopy as makeacopy

def sigmafloor(T,transitionT,transitionTinterval,floor):
    """Generates a sigmoid (smooth step-down) function with a floor"""
    temp = 1 - 1/(1 + np.exp(-(T-transitionT)*3/transitionTinterval))
    return temp*(1-floor)+floor
    
def sigmaup(t,transitiontime,transitiontimeinterval):
    """Generates a sigmoid (smooth step-up) function"""
    return 1 / (1 + np.exp(-(t-transitiontime)*3/transitiontimeinterval))
  
def sigmadown(t,transitiontime,transitiontimeinterval):
    """Generates a sigmoid (smooth step-down) function"""
    return 1 - sigmaup(t,transitiontime,transitiontimeinterval)

def GetMyScenario(filename,reportflag=True,plotflag=True):
    """Loads the stored scheduled flow dictionary, with options to print and plot"""
    
    # Get the entire dictionary
    epsdictionary_fromfile = h5io.read_hdf5(filename)

    # Show what's in the dictionary (if flagged)
    if reportflag:
        print("Here's the scenario summary:")
        display(epsdictionary_fromfile)

    # This extracts the dataframe from the dictionary
    epsdf = epsdictionary_fromfile['dataframe']

    # This extracts the time and emissions from the dataframe
    time = np.array(epsdf['time'])
    eps = np.array(epsdf['emissions'])

    # Graph the time series (if flagged)
    if plotflag:
        plt.figure()
        plt.plot(time,eps)
        plt.grid(True)
        plt.title('Emissions according to '+filename)
        plt.xlabel('year')
        plt.ylabel('GtC/year')
    
    # Return time and emission as numpy arrays
    return time, eps

def CreateClimateState(ClimateParams):
    """Creates a new climate state with default values (preindustrial)"""

    # Create an empty climate state
    ClimateState = {}

    # Fill in some default (preindustrial) values
    ClimateState['C_atm'] = ClimateParams['preindust_C_atm']
    ClimateState['C_ocean'] = ClimateParams['preindust_C_ocean']
    ClimateState['albedo'] = ClimateParams['preindust_albedo']
    ClimateState['T_anomaly'] = 0
    
    # These are just placeholders (values don't mean anything)
    ClimateState['pH'] = 0
    ClimateState['T_C'] = 0
    ClimateState['T_F'] = 0
    ClimateState['F_ha'] = 0
    ClimateState['F_ao'] = 0
    ClimateState['F_oa'] = 0
    ClimateState['F_al'] = 0
    ClimateState['F_la'] = 0
    
    
    # Return the climate
    return ClimateState

def CollectClimateTimeSeries(ClimateState_list,whatIwant):
    """Collects elements from a list of dictionaries"""
    array = np.empty(0)
    for ClimateState in ClimateState_list:
        array = np.append(array,ClimateState[whatIwant])
    return array


def Diagnose_OceanSurfacepH(C_atm,ClimateParams):
    """Computes ocean pH as a function of atmospheric CO2"""
    # Specify a default output value (will be over-ridden by your algorithm)
    pH = 0 
    
    # Extract needed climate parameters
    preindust_pH = ClimateParams['preindust_pH']
    preindust_C_atm = ClimateParams['preindust_C_atm']
    
    # Calculate the new pH according to our algorithm
    pH = -np.log10(C_atm/preindust_C_atm)+preindust_pH

    # Return our diagnosed pH value
    return(pH)

def Diagnose_T_anomaly(C_atm, ClimateParams):
    """Computes a temperature anomaly from the atmospheric carbon amount"""
    CS = ClimateParams['climate_sensitivity']
    preindust_C_atm = ClimateParams['preindust_C_atm']
    T_anomaly = CS*(C_atm-preindust_C_atm)
    return(T_anomaly)

def Diagnose_actual_temperature(T_anomaly):
    """Computes degrees C from a temperature anomaly"""
    T_C = T_anomaly+14
    return(T_C)


def Diagnose_degreesF(T_C):
    """Converts temperature from C to F"""

    # Do the conversion to F
    T_F = T_C*9/5+32### END SOLUTION

    # Return the diagnosed temperature in F
    return(T_F)

def Diagnose_F_ao(C_atm, ClimateParams):
    """Computes flux of carbon from atm to ocean"""

    # Calculate the F_ao based on k_ao and the amount of carbon in the atmosphere
    k_ao = ClimateParams['k_ao']
    F_ao = k_ao*C_atm
    
    # Return the diagnosed flux
    return F_ao

def Diagnose_F_oa(C_ocean, T_anomaly, ClimateParams):
    """Computes a temperature-dependent degassing flux of carbon from the ocean"""

    DC = ClimateParams['DC']
    k_oa = ClimateParams['k_oa']
    F_oa = k_oa*(1+DC*T_anomaly)*C_ocean
    
    # Return the diagnosed flux
    return F_oa

def Diagnose_F_al(T_anomaly, C_atm, ClimateParams):
    """Computes the terrestrial carbon sink"""
    
    # Extract parameters we need from ClimateParameters, and calculate a new flux
    k_al0 = ClimateParams['k_al0']
    k_al1 = ClimateParams['k_al1']
    F_al_transitionT = ClimateParams['F_al_transitionT']
    F_al_transitionTinterval = ClimateParams['F_al_transitionTinterval']
    floor = ClimateParams['fractional_F_al_floor'] 
    F_al = k_al0 + k_al1*sigmafloor(T_anomaly,F_al_transitionT,F_al_transitionTinterval,floor)*C_atm
    
    # Return the diagnosed flux
    return F_al

def Diagnose_F_la(ClimateParams):
    """Computes the terrestrial carbon source"""
    
    k_la = ClimateParams['k_la']
    F_la = k_la
    return F_la

def Diagnose_albedo_with_constraint(T_anomaly, ClimateParams, previousalbedo=0, dtime=0):
    """
    Returns the albedo as a function of temperature, constrained so the change can't 
    exceed a certain amount per year, if so flagged
    """
        
    # Find the albedo without constraint
    albedo = Diagnose_albedo(T_anomaly, ClimateParams)
    
    # Applying a constraint, if called for
    if (previousalbedo !=0) & (dtime != 0):
        albedo_change = albedo-previousalbedo
        max_albedo_change = ClimateParams['max_albedo_change_rate']*dtime
        if np.abs(albedo_change) > max_albedo_change:
            this_albedo_change = np.sign(albedo_change)*max_albedo_change
#             print('albedo change was ', albedo_change, ' and now is', this_albedo_change)
            albedo = previousalbedo + this_albedo_change

    # Return the albedo
    return albedo

def Diagnose_albedo(T_anomaly, ClimateParams):
    """
    Returns the albedo as a function of temperature anomaly
    """
        
    # Extract parameters we need from ClimateParameters, and calculate a new albedo
    transitionT = ClimateParams['albedo_transition_temperature'] 
    transitionTinterval = ClimateParams['albedo_transition_interval'] 
    floor = ClimateParams['fractional_albedo_floor']
    preindust_albedo = ClimateParams['preindust_albedo']
    albedo = sigmafloor(T_anomaly,transitionT,transitionTinterval,floor)*preindust_albedo
                
    # Return the diagnosed albedo
    return albedo


def Diagnose_Delta_T_from_albedo(albedo,ClimateParams):
    """
    Computes additional planetary temperature increase resulting from a lower albedo
    Based on the idea of radiative balance, ASR = OLR
    """
    
    # Extract parameters we need and make the diagnosis
    AS = ClimateParams['albedo_sensitivity']    
    preindust_albedo = ClimateParams['preindust_albedo']
    Delta_T_from_albedo = (albedo-preindust_albedo)*AS
    return Delta_T_from_albedo

# def Diagnose_indirect_T_anomaly(T_anomaly, albedo,ClimateParams):
#     """Computes a temperature anomaly resulting from a lower albedo"""
#     """Based on the idea of radiative balance, ASR = OLR"""
    
#     # Get the delta-T diagnostic
#     Delta_T_from_albedo = Diagnose_Delta_T_from_albedo(albedo,ClimateParams)
    
#     # Update our temperature anomaly
#     T_anomaly += Delta_T_from_albedo
#     return T_anomaly

def Diagnose_Stochastic_C_atm(C_atm,ClimateParams):
    """Returns a noisy version of T"""
    
    # Extract parameters we need and make the diagnosis
    Stochastic_C_atm_std_dev = ClimateParams['Stochastic_C_atm_std_dev']
    C_atm_new = np.random.normal(C_atm, Stochastic_C_atm_std_dev)
    return C_atm_new 

def MakeEmissionsScenario(t_start,t_stop,nsteps,k,eps_0,t_0,t_trans,delta_t_trans):
    """Returns an emissions scenario"""
    time = np.linspace(t_start,t_stop,nsteps)
    myexp = np.exp(k*time)
    myN = eps_0/(np.exp(k*t_0)*sigmadown(t_0,t_trans,delta_t_trans))
    mysigmadown = sigmadown(time,t_trans,delta_t_trans)
    eps = myN*myexp*mysigmadown
    return time, eps

def MakeEmissionsScenario2(t_start,t_stop,nsteps,k,eps_0,t_0,t_peak,delta_t_trans):
    """Returns an emissions scenario parameterized by the year of peak emissions"""
    t_trans = t_peak + delta_t_trans/3*np.log(3/(k*delta_t_trans)-1)
    return MakeEmissionsScenario(t_start,t_stop,nsteps,k,eps_0,t_0,t_trans,delta_t_trans)

def PostPeakFlattener(time,eps,transitiontimeinterval,epslongterm):
    ipeak = np.where(eps==np.max(eps))[0][0]; print('peak',eps[ipeak],ipeak)
    b = eps[ipeak]
    a = epslongterm
    neweps = makeacopy(eps)
    for i in range(ipeak,len(eps)):
        ipostpeak = i-ipeak
        neweps[i] = a + np.exp(-(time[i]-time[ipeak])**2/transitiontimeinterval**2)*(b-a)
    return neweps
 
def MakeEmissionsScenarioLTE(t_start,t_stop,nsteps,k,eps_0,t_0,t_trans,delta_t_trans,epslongterm):
    time, eps = MakeEmissionsScenario(t_start,t_stop,nsteps,k,eps_0,t_0,t_trans,delta_t_trans)
    neweps = PostPeakFlattener(time,eps,delta_t_trans,epslongterm)
    return time, neweps

def MakeEmissionsScenario2LTE(t_start,t_stop,nsteps,k,eps_0,t_0,t_peak,delta_t_trans,epslongterm):
    time, eps = MakeEmissionsScenario2(t_start,t_stop,nsteps,k,eps_0,t_0,t_peak,delta_t_trans)
    neweps = PostPeakFlattener(time,eps,delta_t_trans,epslongterm)
    return time, neweps
