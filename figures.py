def linspace(nest, mu = 0, sigma = 1, outlier = 0, distribuition = 'normal', points = True, grid = True):
    """
    Returns a generic plot of a selected distribuition based on linspace discretization.

    Parameters
    ----------
    nest: int
        The number of estimation points.
    mu: int, optional
        Specifies the mean of distribuition.
        Defaut is 0.
    sigma: int, optional
        Specifies the standard desviation of a distribuition.
        Defaut is 1.
    outlier: int, optional
        Is the point of an outlier event, e.g outlier = 50 will put an event in -50 and +50 if mu = 0.
        Defaut is 0
    distribuition: str, optional
        Select the distribuition to analyze.
        ('normal', 'lognormal')
        Defaut is 'normal'
    points: bool, optional
        If True, it will plot with the discratization points on its PDF.
        Defaut is True.
    grid: bool, optional
        If True, a grid of discatization will be show in the plot.
        Defaut is True.
    """
    import numpy as np
    import scipy.stats as sp
    import matplotlib.pyplot as plt

    ngrid = int(100e3)

    if distribuition == 'normal':
        outlier_inf = outlier_sup = outlier
        a,b = sp.norm.interval(0.9999, loc = mu, scale = sigma)
        xgrid = np.linspace(a-outlier_inf,b+outlier_sup,ngrid)
        xest = np.linspace(min(xgrid), max(xgrid), nest)
        ygrid = sp.norm.pdf(xgrid,loc = mu, scale = sigma)
        yest = sp.norm.pdf(xest, loc = mu, scale = sigma)
    
    elif distribuition == 'lognormal':
        outlier_inf = 0
        outlier_sup = outlier
        a,b = sp.lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
        xgrid = np.linspace(a-outlier_inf,b+outlier_sup,ngrid)
        xest = np.linspace(min(xgrid), max(xgrid), nest)
        ygrid = sp.lognorm.pdf(xgrid, sigma, loc = 0, scale = np.exp(mu))
        yest = sp.lognorm.pdf(xest, sigma, loc = 0, scale = np.exp(mu))

    plt.plot(xgrid,ygrid, label = '%s PDF' %distribuition)
    plt.xlabel('X')
    plt.ylabel('Probability')

    if points:
        plt.plot(xest,yest,'ok', label = 'Linspace Points')
    if grid:
          plt.vlines(xest,0,yest,linestyle=':')
          plt.hlines(yest,a-outlier_inf,xest,linestyle = ':')
          plt.plot(np.zeros(nest)+a-outlier_inf,yest,'rx', ms = 5, label = 'Y points')
    plt.legend()

