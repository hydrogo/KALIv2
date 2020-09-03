import numpy as np
from numba import njit
### helpers

### snow module

@njit
def cema_neige(Temp, Prec, params):
    
    '''
    Cema-Neige snow model
    Input:
    1. Data - timeseries:
        'T'- mean daily temperature (Celsium degrees)
        'P'- mean daily precipitation (mm/day)
    2. Params - list of model parameters:
        'CTG' - dimensionless weighting coefficient of the snow pack thermal state
                [0, 1]
        'Kf'  - day-degree rate of melting (mm/(day*celsium degree))
                [0.01, 10]
    Output:
    Total amount of liquid and melting precipitation daily timeseries
    (for coupling with hydrological model)
    '''

    FraqSolidPrecip = np.where(Temp < -0.2, 1, 0)

    CTG, Kf = params[0], params[1]

    ### initialization ###
    ## constants ##
    # melting temperature
    Tmelt = 0
    # Threshold for solid precip
    # function for Mean Annual Solid Precipitation
    #def MeanAnnualSolidPrecip(data):
    #annual_vals = [data[str(i)].loc[df["T"] < -0.2,  "P"].sum()\
    #               for i in np.unique(data.index.year)]
    #return np.mean(annual_vals)

    #MASP = MeanAnnualSolidPrecip(data)
    MASP = np.mean(Prec*FraqSolidPrecip)*365.25*24
    Gthreshold = 0.9*MASP
    MinSpeed = 0.1

    ## model states ##
    #G = 0
    #eTG = 0
    #PliqAndMelt = 0

    ### ouput of snow model
    # snowpack volume
    G = np.zeros(len(Temp))
    # snowpack termal state
    eTG = np.zeros(len(Temp))
    # Liquid precipitation
    Pliq = np.zeros(len(Temp))
    # Solid precipitation
    Psol = np.zeros(len(Temp))
    # Melt water
    Melt = np.zeros(len(Temp))
    # rain + snow water volume
    PliqAndMelt = np.zeros(len(Temp))

    for t in range(len(Temp)):
        ### solid and liquid precipitation accounting
        # liquid precipitation
        Pliq[t] = (1 - FraqSolidPrecip[t]) * Prec[t]
        # solid precipitation
        Psol[t] = FraqSolidPrecip[t] * Prec[t]
        ### Snow pack volume before melt
        G[t] = G[t-1] + Psol[t]
        ### Snow pack thermal state before melt
        eTG[t] = CTG * eTG[t-1] + (1 - CTG) * Temp[t]
        # control eTG
        if eTG[t] > 0: eTG[t] = 0
        ### potential melt
        if (int(eTG[t]) == 0) & (Temp[t] > Tmelt):
            PotMelt = Kf * (Temp[t] - Tmelt)
            if PotMelt > G[t]: PotMelt = G[t]
        else:
            PotMelt = 0
        ### ratio of snow pack cover (Gratio)
        if G[t] < Gthreshold:
            Gratio = G[t]/Gthreshold
        else:
            Gratio = 1
        ### actual melt
        Melt[t] = ((1 - MinSpeed) * Gratio + MinSpeed) * PotMelt
        ### snow pack volume update
        G[t] = G[t] - Melt[t]
        ### Gratio update
        if G[t] < Gthreshold:
            Gratio = G[t]/Gthreshold
        else:
            Gratio = 1

        ### Water volume to pass to the hydrological model
        PliqAndMelt[t] = Pliq[t] + Melt[t]

    return PliqAndMelt, G, eTG, Pliq, Psol, Melt


### computation of UH ordinates

@njit
def _SS1(I,C,D):
    '''
    Values of the S curve (cumulative HU curve) of GR unit hydrograph UH1
    Inputs:
       C: time constant
       D: exponent
       I: time-step
    Outputs:
       SS1: Values of the S curve for I
    '''
    FI = I+1
    if FI <= 0: SS1 = 0
    elif FI < C: SS1 = (FI/C)**D
    else: SS1 = 1
    return SS1

@njit
def _SS2(I,C,D):
    '''
    Values of the S curve (cumulative HU curve) of GR unit hydrograph UH2
    Inputs:
       C: time constant
       D: exponent
       I: time-step
    Outputs:
       SS2: Values of the S curve for I
    '''
    FI = I+1
    if FI <= 0: SS2 = 0
    elif FI <= C: SS2 = 0.5*(FI/C)**D
    elif C < FI <= 2*C: SS2 = 1 - 0.5*(2 - FI/C)**D
    else: SS2 = 1
    return SS2

@njit
def _UH1(C, D, NH):
    '''
    C Computation of ordinates of GR unit hydrograph UH1 using successive differences on the S curve SS1
    C Inputs:
    C    C: time constant
    C    D: exponent
    C Outputs:
    C    OrdUH1: NH ordinates of discrete hydrograph
    '''
    OrdUH1 = np.zeros(NH)
    for i in range(NH):
        OrdUH1[i] = _SS1(i, C, D) - _SS1(i-1, C, D)
    return OrdUH1

@njit
def _UH2(C, D, NH):
    '''
    C Computation of ordinates of GR unit hydrograph UH2 using successive differences on the S curve SS1
    C Inputs:
    C    C: time constant
    C    D: exponent
    C Outputs:
    C    OrdUH2: NH ordinates of discrete hydrograph
    '''
    OrdUH2 = np.zeros(2*NH)
    for i in range(2*NH):
        OrdUH2[i] = _SS2(i, C, D) - _SS2(i-1, C, D)
    return OrdUH2

@njit
def run(T, P, E, params):
    
    """
    Input forcing:
    P : 1D numpy array with hourly precipitation values (mm)
    E : 1D numpy array with potential evapotranspiration values (mm)

    Model parameters:
    x1 : production store capacity (X1 - PROD) [mm]
    x2 : intercatchment exchange coefficient (X2 - CES) [mm/hour]
    x3 : routing store capacity (X3 - ROUT) [mm]
    x4 : time constant of unit hydrograph (X4 - TB) [hour]
    *x5 : the rate of separation runoff for fast and slow (X5 - B) [dimensionless]
    *x6 : initial state for production store (X6 - St[0]) [mm]
    *x7 : initial state for routing store (X7 - St[1]) [mm]

    """
    ### model parameters unpacking
    
    X1, X2, X3, X4, CTG, Kf = params
    
    P, G, eTG, Plig, Psol, Melt = cema_neige(T, P, [CTG, Kf])
    #X1 = params[0]
    #X2 = params[1]
    #X3 = params[2]
    #X4 = params[3]
    X5 = 0.9
    # X6 = params[5] init for production store
    # X7 = params[6] init for routing store
    
    
    ### initial conditions
    
    # initialization of model states to zero
    # states of production St[0] and routing St[1] reservoirs holder
    St = np.array([X1/2, X3/2])
    Q = np.zeros(len(P))

    
    # parameter for unit hydrograph lenght
    NH = 480
    # Unit hydrograph states holders
    StUH1 = np.zeros(NH)
    StUH2 = np.zeros(2*NH)

    # calculate ordinates of unit hydrograph
    OrdUH1 = _UH1(X4, 1.25, NH)
    OrdUH2 = _UH2(X4, 1.25, NH)

    # LOOP
    for t in range(len(P)):
        
        # interception and production store
        # check how connects Evap and Precip
        # case 1. No effective rainfall
        if P[t] <= E[t]:
            # net evapotranspiration capacity
            EN = E[t] - P[t]
            # net rainfall
            PN = 0
            # part of production store capacity that suffers deficit
            WS = EN/X1
            # control WS
            if WS > 13: WS = 13
            # calculate tahn of WS
            TWS = np.tanh(WS)
            # part of production store capacity has an accumulated rainfall
            Sr = St[0]/X1
            # actual evaporation rate (will evaporate from production store)
            ER = St[0]*(2 - Sr)*TWS/(1 + (1 - Sr)*TWS)
            # actual evapotranspiration
            AE = ER + P[t]
            # production store capacity update
            St[0] = St[0] - ER
            # control state of production store
            if St[0] < 0: St[0] = 0
            # water that reaches routing functions
            PR = 0
        # case 2. Effective rainfall produces runoff
        else:
            # net evapotranspiration capacity
            EN = 0
            # actual evapotranspiration
            AE = E[t]
            # net rainfall
            PN = P[t] - E[t]
            # part of production store capacity that holds rainfall
            WS = PN/X1
            # control WS
            if WS > 13: WS = 13
            # calculate tahn of WS
            TWS = np.tanh(WS)
            # active part of production store
            Sr = St[0]/X1
            # amount of net rainfall that goes directly to the production store
            PS = X1*(1 - Sr*Sr)*TWS/(1 + Sr*TWS)
            # water that reaches routing functions
            PR = PN - PS
            # production store capacity update
            St[0] = St[0] + PS
            # control state of production store
            if St[0] < 0: St[0] = 0
        
        # percolation from production store
        Sr = St[0]/X1
        Sr = Sr * Sr
        Sr = Sr * Sr
        # percolation leakage from production store
        PERC = St[0]*(1 - 1/np.sqrt(np.sqrt(1 + Sr/759.69140625)))
        # production store capacity update
        St[0] = St[0] - PERC
        # update amount of water that reaches routing functions
        PR = PR + PERC
        
        # split of effective rainfall into the two routing components
        # could be further parametrized as x5 parameter
        # i think no one watershed behaves similarly in this way
        PRHU1 = PR*X5 # * x5
        PRHU2 = PR*(1-X5) # * (1 - x5)
        
        # convolution of unit hydrograph UH1
        for k in range(int( max(1, min(NH-1, int(X4+1))) )):
            StUH1[k] = StUH1[k+1] + OrdUH1[k] * PRHU1
        StUH1[NH-1] = OrdUH1[NH-1] * PRHU1

        # convolution of unit hydrograph UH2
        for k in range(int( max(1, min(2*NH-1, 2*int(X4+1))) )):
            StUH2[k] = StUH2[k+1] + OrdUH2[k] * PRHU2
        StUH2[2*NH-1] = OrdUH2[2*NH-1] * PRHU2
        
        # potential intercatchment semi-exchange
        # part of routing store
        Rr = St[1]/X3
        EXCH = X2*Rr*Rr*Rr*np.sqrt(Rr)
        
        # routing store
        AEXCH1 = EXCH
        if St[1] + StUH1[0] + EXCH < 0: AEXCH1 = -St[1] - StUH1[0]
        # update state 2 (routing reservoir)
        St[1] = St[1] + StUH1[0] + EXCH
        # control state 2
        if St[1] < 0: St[1] = 0
        Rr = St[1]/X3
        Rr = Rr * Rr
        Rr = Rr * Rr
        # runoff from routing store QR
        QR = St[1] * (1 - 1/np.sqrt(np.sqrt(1+Rr)))
        # update state 2 (routing reservoir)
        St[1] = St[1] - QR
        
        # direct runoff
        AEXCH2 = EXCH
        if StUH2[0] + EXCH < 0: AEXCH2 = -StUH2[0]
        # runoff from direct branch QD
        QD = max(0, StUH2[0] + EXCH)

        # total runoff
        Q[t] = QR + QD
        #print(Q[t])
        
    # control total runoff
    Q = np.where(Q != np.nan , Q, 0)
    Q = np.where(Q > 0, Q, 0)
    
    
    return Q

def bounds():
    '''
    GR4H params:
    X1 : production store capacity (X1 - PROD) [mm]
        [0.1, 1500]
    X2 : intercatchment exchange coefficient (X2 - CES) [mm/day]
        [-10, 10]
    X3 : routing store capacity (X3 - ROUT) [mm]
        [0.1, 500]
    X4 : time constant of unit hydrograph (X4 - TB) [day]
        [0.5, 4]
    *X5 : the rate of separation runoff for fast and slow (X5 - B) [dimensionless]
        [0.01, 0.99]
    *X6 : initial state for production store (X6 - St[0]) [mm]
        [0.1, 1500]
    *X7 : initial state for routing store (X7 - St[1]) [mm]
        [0.1, 500]
    Cema-Neige snow model parameters:
    CTG : dimensionless weighting coefficient of the snow pack thermal state
        [0, 1]
    Kf : day-degree rate of melting (mm/(day*celsium degree))
        [0.01, 10]
    '''
    bnds = ((0.1, 1500), (-10, 10), (0.1, 500), (0.5, 96.0), (0, 1), (0.001, 10))
    return bnds
