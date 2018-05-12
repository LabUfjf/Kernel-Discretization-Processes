
def diffArea(nest, outlier = 0, kinds = 'all', axis = 'probability', ROI = 20 , mu = 0, sigma = 1, weight = False, interpolator = 'linear', distribuition = 'normal', plot = True):
    
    """
    Return an error area between a analitic function and a estimated discretization from a distribuition.

    Parameters
    ----------
    nest: int
        The number of estimation points.
    outlier: int, optional
        Is the point of an outlier event, e.g outlier = 50 will put an event in -50 and +50 if mu = 0.
        Defaut is 0
    kinds: str or array, optional
        specifies the kind of distribuition to analize.
        ('Linspace', 'CDFm', 'PDFm', 'iPDF1', 'iPDF2', 'all').
        Defaut is 'all'.
    axis: str, optional
        specifies the x axis to analize
        ('probability', 'derivative', '2nd_derivative', 'X').
        Defaut is 'probability'.
    ROI: int, optional
        Specifies the number of regions of interest.
        Defaut is 20.
    mu: int, optional
        Specifies the mean of distribuition.
        Defaut is 0.
    sigma: int, optional
        Specifies the standard desviation of a distribuition.
        Defaut is 1.
    weight: bool, optional
        if True, each ROI will have a diferent weight to analyze.
        Defaut is False
    interpolator: str, optional
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of zeroth, first, second or third order) or as an
        integer specifying the order of the spline interpolator to use.
        Default is 'linear'.
    distribuition: str, optional
        Select the distribuition to analyze.
        ('normal', 'lognormal')
        Defaut is 'normal'
    plot: bool, optional
        If True, a plot will be ploted with the analyzes
        Defaut is True

    """    
    import numpy as np
    from scipy.stats import norm, lognorm
    from numpy import pi
    from scipy.interpolate import interp1d
    from numpy import sqrt, pi, log, exp
    import matplotlib.pyplot as plt
    from distAnalyze import pdf, dpdf, ddpdf, PDF, dPDF, ddPDF

    area = []
    n = []

    if distribuition == 'normal': 
        outlier_inf = outlier_sup = outlier
    elif distribuition == 'lognormal': 
        outlier_inf = 0
        outlier_sup = outlier

    ngrid = int(100e3)
    truth = pdf
        
    if axis == 'probability':
        truth1 = pdf
    elif axis == 'derivative':
        truth1 = dpdf
    elif axis == '2nd_derivative':
        truth1 = ddpdf
    elif axis == 'X':
        truth1 = lambda x,mu,sigma,distribuition: x
    else: return 'No valid axis'
            
    probROIord = {}
    areaROIord = {}
    div = {}
        
    if kinds == 'all': 
        kinds = ['Linspace', 'CDFm', 'PDFm', 'iPDF1', 'iPDF2']
    elif type(kinds) == str:
        kinds = [kinds]

    for kind in kinds:
        if distribuition == 'normal':
            inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
        elif distribuition == 'lognormal':
            inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = exp(mu))

        xgrid = np.linspace(inf,sup,ngrid)
        xgridROI = xgrid.reshape([ROI,ngrid//ROI])
        
        dx = np.diff(xgrid)[0]
        
        if kind == 'Linspace':
            xest = np.linspace(inf-outlier_inf,sup+outlier_sup,nest)
            yest = pdf(xest,mu,sigma,distribuition)
            
        elif kind == 'CDFm':
            eps = 5e-5
            yest = np.linspace(0+eps,1-eps,nest)
            if distribuition == 'normal':
                xest = norm.ppf(yest, loc = mu, scale = sigma)
            elif distribuition == 'lognormal':
                xest = lognorm.ppf(yest, sigma, loc = 0, scale = exp(mu))
            yest = pdf(xest,mu,sigma,distribuition)
            
        elif kind == 'PDFm':
            xest, yest = PDF(nest,mu,sigma, distribuition, outlier)
        elif kind == 'iPDF1':
            xest, yest = dPDF(nest,mu,sigma, distribuition, outlier)
        elif kind == 'iPDF2':
            xest, yest = ddPDF(nest,mu,sigma, distribuition, outlier)      
       
        
        fest = interp1d(xest,pdf(xest,mu, sigma,distribuition),kind = interpolator, fill_value = 'extrapolate')
        
        yestGrid = []
        ytruthGrid = []
        ytruthGrid2 = []
        divi = []
        
        for i in range(ROI):
            yestGrid.append([fest(xgridROI[i])])
            ytruthGrid.append([truth(xgridROI[i],mu,sigma,distribuition)])
            ytruthGrid2.append([truth1(xgridROI[i],mu,sigma,distribuition)])
            divi.append(len(np.intersect1d(np.where(xest >= min(xgridROI[i]))[0], np.where(xest < max(xgridROI[i]))[0])))

        diff2 = (abs((np.array(yestGrid) - np.array(ytruthGrid))*dx))
        areaROI = np.sum(np.sum(diff2,1),1)
        
        divi = np.array(divi)   
        divi[divi == 0] = 1
        
        try:
            probROI = np.mean(np.sum(ytruthGrid2,1),1)
        except:
            probROI = np.mean(ytruthGrid2,1)
        
        probROIord[kind] = np.sort(probROI)
        index = np.argsort(probROI)
        
        areaROIord[kind] = areaROI[index]
        deletes = ~np.isnan(areaROIord[kind])
        areaROIord[kind] = areaROIord[kind][deletes]
        probROIord[kind] = probROIord[kind][deletes]
        
        area = np.append(area,np.mean(areaROIord[kind]))
        n = np.append(n,len(probROIord[kind]))
        div[kind] = divi[index]
        if plot:
            if weight:
                plt.logy(probROIord[kind],areaROIord[kind]*div[kind],'-o',label = kind, ms = 3)
            else: plt.plot(probROIord[kind],areaROIord[kind],'-o',label = kind, ms = 3)

            plt.yscale('log')
            plt.legend()
        
        #plt.title('%s - Pontos = %d, div = %s - %s' %(j,nest, divs,interpolator))
        
    return area,n

def PDF(pts,mu,sigma, distribuition, outlier = 0):
    from scipy.stats import norm, lognorm
    import numpy as np
    from scipy.interpolate import interp1d
    
    if distribuition == 'normal':
        inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
        outlier_inf = outlier_sup = outlier

        X1 = np.linspace(inf-outlier,mu,int(1e6))
        Y1 = norm.pdf(X1, loc = mu, scale = sigma)
        interp = interp1d(Y1,X1)
        y1 = np.linspace(Y1[0],Y1[-1],pts//2+1)
        x1 = interp(y1)
        
        X2 = np.linspace(mu,sup+outlier,int(1e6))
        Y2 = norm.pdf(X2, loc = mu, scale = sigma)
        interp = interp1d(Y2,X2)
        y2 = np.flip(y1,0)
        x2 = interp(y2)

    elif distribuition == 'lognormal':
        inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
        outlier_inf = 0
        outlier_sup = outlier
        mode = np.exp(mu - sigma**2)
        
        X1 = np.linspace(inf-outlier_inf,mode,int(1e6))
        Y1 = lognorm.pdf(X1, sigma, loc = 0, scale = np.exp(mu))
        interp = interp1d(Y1,X1)
        y1 = np.linspace(Y1[0],Y1[-1],pts//2+1)
        x1 = interp(y1)
        
        X2 = np.linspace(mode,sup+outlier_sup,int(1e6))
        Y2 = lognorm.pdf(X2, sigma, loc = 0, scale = np.exp(mu))
        interp = interp1d(Y2,X2)
        y2 = np.flip(y1,0)
        x2 = interp(y2)
       
        
    X = np.concatenate([x1[:-1],x2])
    Y = np.concatenate([y1[:-1],y2])
    
    return X,Y

def dPDF(pts,mu,sigma, distribuition, outlier = 0):
    import numpy as np
    from scipy.interpolate import interp1d
    from distAnalyze import dpdf
    from scipy.stats import norm, lognorm
    eps = 5e-5

    if distribuition == 'normal':
        inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
        outlier_inf = outlier_sup = outlier
    elif distribuition == 'lognormal':
        inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
        outlier_inf = 0
        outlier_sup = outlier
    #dy = lambda x,u,s : abs(1/(s**3*sqrt(2*pi))*(u-x)*np.exp(-0.5*((u-x)/s)**2))
    
    x = np.linspace(inf-outlier_inf,sup+outlier_sup,int(10e3))
    y = dpdf(x,mu,sigma,distribuition)
    
    cdf = np.sum(np.tri(len(x))*y,1)
    cdf = cdf/max(cdf)
    
    interp = interp1d(cdf,x, fill_value = 'extrapolate')
    Y = np.linspace(eps,1-eps,pts)
    X = interp(Y)
    
    return X,Y

    
def ddPDF(pts,mu,sigma, distribuition, outlier = 0):
    import numpy as np
    from scipy.interpolate import interp1d
    from distAnalyze import ddpdf
    from scipy.stats import norm, lognorm
    eps = 5e-5 
    #ddy = lambda x,u,s: abs(-(s**2-u**2+2*u*x-x**2)/(s**5*sqrt(2*pi))*np.exp(-0.5*((u-x)/s)**2))
    if distribuition == 'normal':
        inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
        outlier_inf = outlier_sup = outlier
    elif distribuition == 'lognormal':
        inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
        outlier_inf = 0
        outlier_sup = outlier
    
    x = np.linspace(inf-outlier_inf,sup+outlier_sup,int(10e3))
    y = ddpdf(x,mu,sigma,distribuition)
    
    cdf = np.sum(np.tri(len(x))*y,1)
    cdf = cdf/max(cdf)
    
    interp = interp1d(cdf,x, fill_value = 'extrapolate')
    Y = np.linspace(eps,1-eps,pts)
    X = interp(Y)

    return X,Y



def pdf(x, u, s, distribuition):
    import numpy as np
    from numpy import pi, sqrt, log, exp, isnan

    if distribuition == 'normal':
        y = 1/(s*sqrt(2*pi))*exp(-0.5*((x-u)/s)**2)

    elif distribuition == 'lognormal':
        y = 1/(x*s*sqrt(2*pi))*exp(-(log(x)-u)**2/(2*s**2))
        y[isnan(y)] = 0
    return y

def dpdf(x, u, s, distribuition):
    import numpy as np
    from numpy import pi, sqrt, log, exp, isnan
    
    if distribuition == 'normal':
        y = abs(1/(s**3*sqrt(2*pi))*(u-x)*np.exp(-0.5*((u-x)/s)**2))

    elif distribuition == 'lognormal':
        y = abs(-exp(-(u-log(x))**2/(2*s**2))*(s**2-u+log(x))/(s**3*x**2*sqrt(2*pi)))
        y[isnan(y)] = 0
    return y

def ddpdf(x, u, s, distribuition):
    import numpy as np
    from numpy import pi, sqrt, log, exp, isnan
    
    if distribuition == 'normal':
        y = abs(-(s**2-u**2+2*u*x-x**2)/(s**5*sqrt(2*pi))*np.exp(-0.5*((u-x)/s)**2))

    elif distribuition == 'lognormal':
        y = abs(exp(-(log(x)-u)**2/(2*s**2))*(2*s**4-3*s**2*u+3*s**2*log(x)-s**2+u**2-2*u*log(x)+log(x)**2)/(s**5*x**3*sqrt(2*pi)))
        y[isnan(y)] = 0
    return y