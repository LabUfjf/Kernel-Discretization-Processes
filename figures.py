def linspace(nest, mu = 0, sigma = 1, outlier = 0, distribuition = 'normal', points = False, grid = False):

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
        Defaut is False.
    grid: bool, optional
        If True, a grid of discatization will be show in the plot.
        Defaut is False.
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

def cdf(nest, mu = 0, sigma = 1, outlier = 0, distribuition = 'normal', points = None, grid = False, PDF = False):
    
    """
    Returns a generic plot from CDF of a selected distribuition based on cdf discretization.

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
    points: str, optional
        Show the estimation points along the follow plots ('PDF' or 'CDF')
        Defaut is None.
    grid: bool, optional
        If True, a grid of discatization will be show in the plot.
        Defaut is False.
    PDF: bool, optional
        If True, the PDF plot will be show.
        Defaut is False
    """
        
    import numpy as np
    import scipy.stats as sp
    import matplotlib.pyplot as plt

    ngrid = int(1e6)
    eps = 5e-5
    blue = '#1f77b4ff'
    orange = '#ff7f0eff'

    if distribuition == 'normal':
        outlier_inf = outlier_sup = outlier
        a,b = sp.norm.interval(0.9999, loc = mu, scale = sigma)
        a,b = a-outlier_inf, b+outlier_sup
        x = np.linspace(a,b,ngrid)
        y = sp.norm.pdf(x,loc = mu, scale = sigma)
        
        ygrid = np.linspace(eps, 1-eps, ngrid)
        xgrid = sp.norm.ppf(ygrid, loc = mu, scale = sigma)

        yest = np.linspace(eps, 1-eps, nest)
        xest = sp.norm.ppf(yest, loc = mu, scale = sigma)

    elif distribuition == 'lognormal':
        outlier_inf = 0
        outlier_sup = outlier
        a,b = sp.lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
        a,b = a-outlier_inf, b+outlier_sup
        x = np.linspace(a,b,ngrid)
        y = sp.lognorm.pdf(x,sigma, loc = 0, scale = np.exp(mu))

        ygrid = np.linspace(eps, 1-eps, ngrid)
        xgrid = sp.lognorm.ppf(ygrid, sigma, loc = 0, scale = np.exp(mu))

        yest = np.linspace(eps, 1-eps, nest)
        xest = sp.lognorm.ppf(yest, sigma, loc = 0, scale = np.exp(mu))

    fig, ax1 = plt.subplots()
    cdf, = ax1.plot(xgrid,ygrid, color = orange)
    ax1.set_ylabel('Cumulative Probability', color = orange)
    ax1.legend([cdf], ['CDF'])
    ax1.set_xlabel('x')

    if points == 'CDF':
        pts, = ax1.plot(xest,yest,'ok')
        ax1.legend([cdf,pts], ['CDF', 'Points'])

    if PDF or points == 'PDF':
        ax2 = ax1.twinx()

        pdf, = ax2.plot(x,y, color = blue)
        ax1.legend([pdf,cdf], ['PDF', 'CDF'])
        ax2.set_ylabel('Probability', color = blue)
        ax1.tick_params(axis='y', labelcolor=orange)
        ax2.tick_params(axis='y', labelcolor=blue)

        if points == 'PDF':
        	if distribuition == 'lognormal':
        		pts, = ax2.plot(xest,sp.lognorm.pdf(xest, sigma, loc = 0, scale = np.exp(mu)), 'ok')
        	elif distribuition == 'normal':
        		pts, = ax2.plot(xest,sp.norm.pdf(xest, loc = mu, scale = sigma), 'ok')
        	ax1.legend([pdf,cdf,pts], ['PDF', 'CDF', 'Points'])
    if grid:
          ax2.vlines(xest,0,yest,linestyle=':')
          ax2.hlines(yest,a,xest,linestyle = ':')
          ypts, = ax1.plot(xest,np.zeros(nest),'rx', ms = 5)
          ax1.legend([pdf,cdf,pts,ypts], ['PDF', 'CDF', 'Points', 'X Points'])