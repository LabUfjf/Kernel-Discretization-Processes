def linspace(nest, mu = 0, sigma = 1, outlier = 0, distribuition = 'normal',data = 0, points = False, grid = False):

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
    data: int, optional
        If data > 0, a randon data will be inserted insted analitcs data.
        Defaut is 0.
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
        if not data:
              a,b = sp.norm.interval(0.9999, loc = mu, scale = sigma)
        else:
              d = np.random.normal(loc = mu, scale = sigma, size = data)
              a,b = min(d),max(d)
              
        xgrid = np.linspace(a-outlier_inf,b+outlier_sup,ngrid)
        xest = np.linspace(min(xgrid), max(xgrid), nest)
        ygrid = sp.norm.pdf(xgrid,loc = mu, scale = sigma)
        yest = sp.norm.pdf(xest, loc = mu, scale = sigma)
              
    
    elif distribuition == 'lognormal':
        outlier_inf = 0
        outlier_sup = outlier
        if not data:
              a,b = sp.lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
        else:
              d = np.random.lognormal(mean = mu, sigma = sigma, size = data)
              a,b = min(d),max(d)
              
        xgrid = np.linspace(a-outlier_inf,b+outlier_sup,ngrid)
        xest = np.linspace(min(xgrid), max(xgrid), nest)
        ygrid = sp.lognorm.pdf(xgrid, sigma, loc = 0, scale = np.exp(mu))
        yest = sp.lognorm.pdf(xest, sigma, loc = 0, scale = np.exp(mu))

    plt.figure(figsize = (12,8))       
    plt.plot(xgrid,ygrid, label = 'PDF')

    plt.xlabel('X',fontsize = 30)
    plt.ylabel('Probability',fontsize = 30)

    if points:
        plt.plot(xest,yest,'ok', label = 'Linspace Points')
    if grid:
          plt.vlines(xest,0,yest,linestyle=':')
          plt.hlines(yest,a-outlier_inf,xest,linestyle = ':')
          plt.plot(np.zeros(nest)+a-outlier_inf,yest,'rx', ms = 5, label = 'Y points')
    plt.legend(prop = {'size':18})
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    #plt.title('$\mu$ = %.1f, $\sigma$ = %.1f - Linspace' %(mu,sigma))
    plt.tight_layout()
    #plt.savefig("/media/rafael/DiscoCompartilhado/Faculdade/Bolsa - Atlas/KernelDensityEstimation-Python/Kernel-Discretization-Processes/Figures_log/linspace.png")


def CDFm(nest, mu = 0, sigma = 1, data = 0, outlier = 0, distribuition = 'normal', points = None, grid = False, PDF = False):
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
    data: int, optional
        If data > 0, a randon data will be inserted insted analitcs data.
        Defaut is 0.
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
    from statsmodels.distributions import ECDF
    from scipy.interpolate import interp1d


    ngrid = int(1e6)
    eps = 5e-5
    blue = '#1f77b4ff'
    orange = '#ff7f0eff'

    if distribuition == 'normal':
        outlier_inf = outlier_sup = outlier
        if not data:
              a,b = sp.norm.interval(0.9999, loc = mu, scale = sigma)
              a,b = a-outlier_inf, b+outlier_sup
              
              yest = np.linspace(eps, 1-eps, nest)
              xest = sp.norm.ppf(yest, loc = mu, scale = sigma)
              ygrid = np.linspace(eps, 1-eps, ngrid)
              xgrid = sp.norm.ppf(ygrid, loc = mu, scale = sigma)
              
              
        else:
              d = np.random.normal(loc = mu, scale = sigma, size = data)
              ecdf = ECDF(d)
              a,b = min(d),max(d)
              xest = np.linspace(a,b,data)
              yest = ecdf(xest)
              interp = interp1d(yest,xest,fill_value = 'extrapolate', kind = 'linear')
              yest = np.linspace(eps,1-eps,nest)
              xest = interp(yest)
              ygrid = np.linspace(eps, 1-eps, ngrid)
              xgrid = interp(ygrid)
              
              #a,b = min(d)-outlier_inf,max(d)+outlier_sup
              
              #yest,xest = np.histogram(d,bins = 'fd',normed = True)
              #xest = np.mean(np.array([xest[:-1],xest[1:]]),0)
              #yest = np.cumsum(yest)/max(np.cumsum(yest))
              #yest = np.linspace(min(yest),max(yest),nest)
              
        
        
        
        

    elif distribuition == 'lognormal':
        outlier_inf = 0
        outlier_sup = outlier
        if not data:
              a,b = sp.lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
              a,b = a-outlier_inf, b+outlier_sup
      
              yest = np.linspace(eps, 1-eps, nest)
              xest = sp.lognorm.ppf(yest, sigma, loc = 0, scale = np.exp(mu))
              ygrid = np.linspace(eps, 1-eps, ngrid)
              xgrid = sp.lognorm.ppf(ygrid, sigma, loc = 0, scale = np.exp(mu))
              
        else:
              d = np.random.lognormal(mean = mu, sigma = sigma, size = data)
              #a,b = min(d),max(d)+outlier_sup
              ecdf = ECDF(d)
              a,b = min(d),max(d)
              xest = np.linspace(a,b,nest)
              yest = ecdf(xest)
              interp = interp1d(yest,xest,fill_value = 'extrapolate', kind = 'linear')
              yest = np.linspace(eps,1-eps,nest)
              xest = interp(yest)
              ygrid = np.linspace(eps, 1-eps, ngrid)
              xgrid = interp(ygrid)#sp.lognorm.ppf(ygrid, sigma, loc = 0, scale = np.exp(mu))
              
              #yest,xest = np.histogram(d,bins = 'fd',normed = True)
              #xest = np.mean(np.array([xest[:-1],xest[1:]]),0)
              #yest = np.cumsum(yest)/max(np.cumsum(yest))
              #yest = np.linspace(min(yest),max(yest),nest)
              
            
              
        
              
        x = np.linspace(a,b,ngrid)
        y = sp.lognorm.pdf(x,sigma, loc = 0, scale = np.exp(mu))

        

    fig, ax1 = plt.subplots(figsize = (12,8), dpi = 200)
    
    #ax1.set_title('$\mu$ = %.1f, $\sigma$ = %.1f - CDFm' %(mu,sigma))
    cdf, = ax1.plot(xgrid,ygrid, color = orange)
    ax1.set_ylabel('Cumulative Probability', color = orange, fontsize = 30)
    ax1.legend([cdf], ['CDF'])
    ax1.set_xlabel('x', fontsize = 30)

    if points == 'CDF':
        pts, = ax1.plot(xest,yest,'ok')
        ax1.legend([cdf,pts], ['CDF', 'Points'])

    if PDF or points == 'PDF':
        ax2 = ax1.twinx()

        pdf, = ax2.plot(x,y, color = blue)
        ax1.legend([pdf,cdf], ['PDF', 'CDF'])
        ax2.set_ylabel('Probability', color = blue, fontsize = 30)
        ax1.tick_params(axis='y', labelcolor=orange, labelsize = 18)
        ax2.tick_params(axis='y', labelcolor=blue, labelsize = 18)
        ax1.tick_params(axis='x', labelsize = 18)

        if points == 'PDF':
            if distribuition == 'lognormal':
                pts, = ax2.plot(xest,sp.lognorm.pdf(xest, sigma, loc = 0, scale = np.exp(mu)), 'ok')
            elif distribuition == 'normal':
                pts, = ax2.plot(xest,sp.norm.pdf(xest, loc = mu, scale = sigma), 'ok')
            ax1.legend([pdf,cdf,pts], ['PDF', 'CDF', 'Points'])
            
    else:
          ax2 = ax1
    if grid:
          ax2.vlines(xest,0,yest,linestyle=':')
          ax2.hlines(yest,a,xest,linestyle = ':')
          ypts, = ax1.plot(xest,np.zeros(nest),'rx', ms = 5)
          ax1.legend([pdf,cdf,pts,ypts], ['PDF', 'CDF', 'CDF Points', 'X Points'], prop = {'size':18}, loc = 1)
          plt.tight_layout()
          #plt.savefig("/media/rafael/DiscoCompartilhado/Faculdade/Bolsa - Atlas/KernelDensityEstimation-Python/Kernel-Discretization-Processes/Figures_log/CDFm.png")


def PDFm(nest, mu = 0, sigma = 1, data = 0, outlier = 0, distribuition = 'normal', grid = False, points = False):
    """
    Returns a generic plot from PDF of a selected distribuition based on PDFm discretization.

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
    data: int, optional
        If data > 0, a randon data will be inserted insted analitcs data.
        Defaut is 0.
    outlier: int, optional
        Is the point of an outlier event, e.g outlier = 50 will put an event in -50 and +50 if mu = 0.
        Defaut is 0
    distribuition: str, optional
        Select the distribuition to analyze.
        ('normal', 'lognormal')
        Defaut is 'normal'
    points: str, optional
        Show the estimation points along the follow plots ('PDF' or 'CDF')
        Defaut is False.
    grid: bool, optional
        If True, a grid of discretization will be show in the plot.
        Defaut is False.
    """

    import numpy as np
    import scipy.stats as sp
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    ngrid = int(1e6)
    if not nest %2:
          nest = nest -1
    if distribuition == 'normal':
        
        outlier_inf = outlier_sup = outlier
        if not data:
              a,b = sp.norm.interval(0.9999, loc = mu, scale = sigma)
              a,b = a-outlier_inf, b+outlier_sup
              x = np.linspace(a,b,ngrid)
              y = sp.norm.pdf(x,loc = mu, scale = sigma)
              
              X1 = np.linspace(a,mu,ngrid)
              Y1 = sp.norm.pdf(X1,loc = mu, scale = sigma)
              interp = interp1d(Y1,X1)
              y1 = np.linspace(Y1[0],Y1[-1],nest//2+1)
              x1 = interp(y1)
      
              X2 = np.linspace(mu,b,ngrid)
              Y2 = sp.norm.pdf(X2, loc = mu, scale = sigma)
              interp = interp1d(Y2,X2)
              y2 = np.flip(y1,0)
              x2 = interp(y2)
        else:
              d = np.random.normal(mu,sigma,data)
              a,b = min(d)-outlier_inf,max(d)+outlier_sup
              yest,xest = np.histogram(d,bins = 'fd',normed = True)
              xest = np.mean(np.array([xest[:-1],xest[1:]]),0)
              M = np.where(yest == max(yest))[0][0]
              m = np.where(yest == min(yest))[0][0]
              interpL = interp1d(yest[:M+1],xest[:M+1], assume_sorted = False, fill_value= 'extrapolate')
              interpH = interp1d(yest[M:],xest[M:], assume_sorted= False, fill_value='extrapolate')
              
              y1 = np.linspace(yest[m],yest[M],nest//2+1)
              x1 = interpL(y1)
              
              y2 = np.flip(y1,0)
              x2 = interpH(y2)
              
        X = np.concatenate([x1[:-1],x2])
        Y = np.concatenate([y1[:-1],y2])

    elif distribuition == 'lognormal':
        outlier_inf = 0
        outlier_sup = outlier
        if not data:
              mode = np.exp(mu-sigma**2)
              a,b = sp.lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
              a,b = a-outlier_inf, b+outlier_sup
              x = np.linspace(a,b,ngrid)
              y = sp.lognorm.pdf(x,sigma, loc = 0, scale = np.exp(mu))
      
              X1 = np.linspace(a,mode,ngrid)
              Y1 = sp.lognorm.pdf(X1,sigma, loc = 0, scale = np.exp(mu))
              interp = interp1d(Y1,X1)
              y1 = np.linspace(Y1[0],Y1[-1],nest//2+1)
              x1 = interp(y1)
      
              X2 = np.linspace(mode,b,ngrid)
              Y2 = sp.lognorm.pdf(X2, sigma, loc = 0, scale = np.exp(mu))
              interp = interp1d(Y2,X2)
              y2 = np.flip(y1,0)
              x2 = interp(y2)
        else:
              d = np.random.lognormal(mu,sigma,data)
              a,b = min(d)-outlier_inf,max(d)+outlier_sup
              y,x = np.histogram(d,bins = 'fd',normed = True)
              x = np.mean(np.array([x[:-1],x[1:]]),0)
              M = np.where(y == max(y))[0][0]
              m = np.where(y == min(y))[0][0]
              interpL = interp1d(y[:M+1],x[:M+1], assume_sorted = False, fill_value= 'extrapolate')
              interpH = interp1d(y[M:],x[M:], assume_sorted= False, fill_value='extrapolate')
              
              y1 = np.linspace(y[m],y[M],nest//2+1)
              x1 = interpL(y1)
              
              y2 = np.flip(y1,0)
              x2 = interpH(y2)
              

        X = np.concatenate([x1[:-1],x2])
        Y = np.concatenate([y1[:-1],y2])
        
    #plt.figure(figsize=(12,8),dpi=200)
    plt.plot(x,y,label = 'PDF')
    plt.ylabel('Probability', fontsize = 30)
    plt.xlabel('x', fontsize = 30)
    #plt.title('$\mu$ = %.1f, $\sigma$ = %.1f - PDFm' %(mu,sigma))
    if points:
      plt.plot(X,Y,'ok', label = 'PDF points')

    if grid:
        plt.vlines(X,0,Y,linestyle=':')
        plt.hlines(Y,a,X,linestyle = ':')
        plt.plot(X,np.zeros(nest),'rx', ms = 5, label = 'X points')

    plt.legend(prop = {'size':18})
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    #plt.savefig("/media/rafael/DiscoCompartilhado/Faculdade/Bolsa - Atlas/KernelDensityEstimation-Python/Kernel-Discretization-Processes/Figures_log/PDFm.png")


def iPDF1(nest, mu = 0, sigma = 1, outlier = 0, data = 0, distribuition = 'normal', points = None, grid = False,CDF = True, PDF = False, dPDF = False):
    """
    Returns a generic plot from CDF of a selected distribuition based on iPDF1 discretization.

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
    data: int, optional
        If data > 0, a randon data will be inserted insted analitcs data.
        Defaut is 0.
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
    CDF: bool, optional
        If True, the CDF plot will be show.
        Defaut is True.
    PDF: bool, optional
        If True, the PDF plot will be show.
        Defaut is False.
    dPDF: bool, optional
        If True, a plot from a derivative PDF will be ploted.
        Defaut is False.
    """
        
    import numpy as np
    import scipy.stats as sp
    import matplotlib.pyplot as plt
    from distAnalyze import dpdf
    from scipy.interpolate import interp1d

    ngrid = int(1e6)
    eps = 5e-5
    blue = '#1f77b4ff'
    orange = '#ff7f0eff'

    if distribuition == 'normal':
        outlier_inf = outlier_sup = outlier
        if not data:
              a,b = sp.norm.interval(0.9999, loc = mu, scale = sigma)
              a,b = a-outlier_inf, b+outlier_sup
              x = np.linspace(a,b,ngrid)
              y = dpdf(x, mu, sigma, distribuition)
              xgrid = x
        else:
              d = np.random.normal(mu,sigma,data)
              a,b = min(d)-outlier_inf,max(d)+outlier_sup
              
              y,x = np.histogram(d,bins = 'fd',normed = True)
              x = np.mean(np.array([x[:-1],x[1:]]),0)
              y2 = y
              y = abs(diff(mediaMovel(y,10)))
              x = xgrid = x[:-1]

    elif distribuition == 'lognormal':
        outlier_inf = 0
        outlier_sup = outlier
        if not data:
              a,b = sp.lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
              a,b = a-outlier_inf, b+outlier_sup
              
              x = np.linspace(a,b,ngrid)
              y = dpdf(x, mu, sigma, distribuition)
              xgrid = x
              
        else:
              d = np.random.lognormal(mu,sigma,data)
              a,b = min(d)-outlier_inf,max(d)+outlier_sup
              
              y,x = np.histogram(d,bins = 'fd',normed = True)
              x = np.mean(np.array([x[:-1],x[1:]]),0)
              y2 = y
              y = abs(diff(mediaMovel(y,10)))
              x = xgrid = x[:-1]
              
    difx = diff(x)[0]
    y = abs(y/(sum(y)*difx))
        
    ygrid = np.cumsum(y)
    ygrid = ygrid/max(ygrid)
    

    interp = interp1d(ygrid,xgrid, fill_value = 'extrapolate')
    yest = np.linspace(eps, max(ygrid)-eps,nest)
    xest = interp(yest)

    fig, ax1 = plt.subplots(figsize=(12,8),dpi=200)
    #ax1.set_title('$\mu$ = %.1f, $\sigma$ = %.1f - iPDF1' %(mu,sigma))
    
    if CDF == True:
          cdf, = ax1.plot(xgrid,ygrid, color = orange)
          ax1.set_ylabel('Cumulative Probability', color = orange, fontsize = 30)
          ax1.legend([cdf], ['CDF'])
          ax2 = ax1.twinx()
    else: ax2 = ax1
    
    ax1.set_xlabel('x', fontsize = 30)
    
    if PDF:
        if distribuition == 'normal':
              pdf, = ax2.plot(x,sp.norm.pdf(x,loc = mu, scale = sigma), color = blue)
        elif distribuition == 'lognormal':
              pdf, = ax2.plot(x,sp.lognorm.pdf(x,sigma,loc = 0, scale = np.exp(mu)), color = blue, label = 'PDF')
        ax2.set_ylabel('Probability', color = blue, fontsize = 30)

    if points == 'CDF':
        pts, = ax1.plot(xest,yest,'ok')
        ax1.legend([cdf,pts], ['CDF', 'Points'])

    if dPDF or points == 'PDF':
        
        if CDF:
              dpdf, = ax2.plot(x,y, color = blue)
              ax1.tick_params(axis='y', labelcolor=orange, labelsize = 18)
              ax1.tick_params(axis='x', labelsize = 18)
              ax2.tick_params(axis='y', labelcolor=blue, labelsize = 18)
              
        else:
              dpdf, = ax2.plot(x,y, color = orange, label = 'dPDF')
              ax2.set_ylabel('Probability',color = 'black')
              ax2.tick_params(labelsize = 18)
        
        #ax1.legend([dpdf,cdf], ['dPDF', 'CDF'])
        
       

        if points == 'PDF':
            if data:
                  pdf, = ax2.plot(x,y2[:-1], color = blue)
                  pts, = ax2.plot(xest,sp.lognorm.pdf(xest, sigma, loc = 0, scale = np.exp(mu)), 'ok')
            else:
                  if distribuition == 'lognormal':
                      pdf, = ax2.plot(x,sp.lognorm.pdf(x, sigma, loc = 0, scale = np.exp(mu)), color = blue)
                      pts, = ax2.plot(xest,sp.lognorm.pdf(xest, sigma, loc = 0, scale = np.exp(mu)), 'ok')
                      
                  elif distribuition == 'normal':
                      pdf, = ax2.plot(x,sp.norm.pdf(x,loc = mu, scale = sigma), color = blue)
                      pts, = ax2.plot(xest,sp.norm.pdf(xest, loc = mu, scale = sigma), 'ok')
               
            ax1.legend([pdf,cdf,pts], ['PDF', 'CDF', 'iPDF1 Points'],prop = {'size':18})
    if grid:
          ax1.vlines(xest,0,yest,linestyle=':')
          ax1.hlines(yest,a,xest,linestyle = ':')
          ypts, = ax1.plot(xest,np.zeros(nest),'rx', ms = 5)
          ax1.legend([pdf,cdf,pts,ypts], ['PDF', 'CDF', 'iPDF1 Points', 'X Points'],prop = {'size':18}, loc = 1)
# =============================================================================
#     ax1.tick_params(axis='y', labelcolor=orange, labelsize = 18)
#     ax1.tick_params(axis='x', labelsize = 18)
#     ax2.tick_params(axis='y', labelcolor=blue, labelsize = 18)      
# =============================================================================
    #ax1.legend(prop = {'size':18}, loc = 1)
    plt.tight_layout()
    #plt.savefig("/media/rafael/DiscoCompartilhado/Faculdade/Bolsa - Atlas/KernelDensityEstimation-Python/Kernel-Discretization-Processes/Figures_log/iPDF1_1.png")
    
        
        
def iPDF2(nest, mu = 0, sigma = 1, outlier = 0, data = 0, distribuition = 'normal', points = None, grid = False, PDF = False, dPDF = False, CDF = True):
    """
    Returns a generic plot from CDF of a selected distribuition based on iPDF1 discretization.

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
        Defaut is 0.
    data: int, optional
        If data > 0, a randon data will be inserted insted analitcs data.
        Defaut is 0.
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
        Defaut is False.
    dPDF: bool, optional
        If True, a plot from a derivative PDF will be ploted.
        Defaut is False.
    """
        
    import numpy as np
    import scipy.stats as sp
    import matplotlib.pyplot as plt
    from distAnalyze import ddpdf
    from scipy.interpolate import interp1d

    ngrid = int(1e6)
    eps = 5e-5
    blue = '#1f77b4ff'
    orange = '#ff7f0eff'

    if distribuition == 'normal':
        outlier_inf = outlier_sup = outlier
        if not data:
              a,b = sp.norm.interval(0.9999, loc = mu, scale = sigma)
              a,b =   a-outlier_inf, b+outlier_sup
              
              x = np.linspace(a,b,ngrid)
              y = ddpdf(x, mu, sigma, distribuition)
              xgrid = x
              
        else:
              d = np.random.normal(mu,sigma,data)
              a,b = min(d)-outlier_inf,max(d)+outlier_sup
              
              y,x = np.histogram(d,bins = 'fd',normed = True)
              x = np.mean(np.array([x[:-1],x[1:]]),0)
              
              y = abs(diff(mediaMovel(y,10),2))
              xgrid = x[:-2]

    elif distribuition == 'lognormal':
        outlier_inf = 0
        outlier_sup = outlier
        if not data:
              a,b = sp.lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
              a,b = a-outlier_inf, b+outlier_sup
              
              x = np.linspace(a,b,ngrid)
              y = ddpdf(x, mu, sigma, distribuition)
              xgrid = x
        else:
              d = np.random.lognormal(mu,sigma,data)
              a,b = min(d)-outlier_inf,max(d)+outlier_sup
              
              y,x = np.histogram(d,bins = 'fd',normed = True)
              x = np.mean(np.array([x[:-1],x[1:]]),0)
              
              y = abs(diff(mediaMovel(y,10),2))
              xgrid = x[:-2]
    
    difx = np.diff(x)[0]
    y = y/(sum(y)*difx)
    #ygrid = np.sum(np.tri(ngrid)*y,1)
    ygrid = np.cumsum(y)
    
    ygrid = ygrid/max(ygrid)
   
    

    interp = interp1d(ygrid,xgrid, fill_value = 'extrapolate')
    yest = np.linspace(eps, max(ygrid)-eps,nest)
    xest = interp(yest)

    fig, ax1 = plt.subplots(figsize = (12,8),dpi=200)
    
    #ax1.set_title('$\mu$ = %.1f, $\sigma$ = %.1f - iPDF2' %(mu,sigma))
    if CDF:
          ax2 = ax1.twinx()
          cdf, = ax1.plot(xgrid,ygrid, color = orange)
          ax1.set_ylabel('Cumulative Probability', color = orange, fontsize = 30)
          ax1.legend([cdf], ['CDF'])
          ax1.set_xlabel('x', fontsize = 30)
          ax1.tick_params(axis='x', labelsize = 18)
          ax1.tick_params(axis='y', labelsize = 18, labelcolor = orange)
          ax2.tick_params(axis='y', labelsize = 18, labelcolor = blue)
          ax2.set_ylabel('Probability',color = blue, fontsize = 30)
    else:
          ax2 = ax1
          ax1.set_ylabel('Probability', fontsize = 30)
          ax1.tick_params(labelsize = 18)
          ax1.set_xlabel('x', fontsize = 30)




    if points == 'CDF':
        pts, = ax1.plot(xest,yest,'ok')
        ax1.legend([cdf,pts], ['CDF', 'Points'])
        
    if PDF:
        if distribuition == 'lognormal':
            pdf, = ax2.plot(x,sp.lognorm.pdf(x, sigma, loc = 0, scale = np.exp(mu)), color = blue, label = 'PDF')
        elif distribuition == 'normal':
            pdf, = ax2.plot(x,sp.norm.pdf(x,loc = mu, scale = sigma), color = blue)
        

    if dPDF or points == 'PDF':
        
        if CDF: 
              color = blue
              ax2.set_ylabel('Probability', color = color, fontsize = 30)
              ax1.tick_params(axis='y', labelcolor=orange, labelsize = 18)
              ax2.tick_params(axis='y', labelcolor=blue, labelsize = 18)
        
        else: 
              color = orange 
        dpdf, = ax2.plot(x,y, color = color, label = 'dPDF2')
        #ax1.legend([dpdf,cdf], ['dPDF', 'CDF'])
        
        if points == 'PDF':
            if distribuition == 'lognormal':
                pdf, = ax2.plot(x,sp.lognorm.pdf(x, sigma, loc = 0, scale = np.exp(mu)), color = blue)
                pts, = ax2.plot(xest,sp.lognorm.pdf(xest, sigma, loc = 0, scale = np.exp(mu)), 'ok')
                
            elif distribuition == 'normal':
                pdf, = ax2.plot(x,sp.norm.pdf(x,loc = mu, scale = sigma), color = blue)
                pts, = ax2.plot(xest,sp.norm.pdf(xest, loc = mu, scale = sigma), 'ok')
               
            ax1.legend([pdf,cdf,pts], ['PDF', 'CDF', 'Points'])
    if grid:
          ax1.vlines(xest,0,yest,linestyle=':')
          ax1.hlines(yest,a,xest,linestyle = ':')
          ypts, = ax1.plot(xest,np.zeros(nest),'rx', ms = 5)
          ax1.legend([pdf,cdf,pts,ypts], ['PDF', 'CDF', 'iPDF2 Points', 'X Points'], prop = {'size':18}, loc = 1)
    plt.legend(prop = {'size':18}, loc = 1)
    plt.tight_layout()
    #plt.savefig("/media/rafael/DiscoCompartilhado/Faculdade/Bolsa - Atlas/KernelDensityEstimation-Python/Kernel-Discretization-Processes/Figures_log/iPDF2_1.png")

 
        
def logs(sigma = 1, mu = 0, area = 0.9999):
      from scipy.stats import lognorm
      import numpy as np
      import matplotlib.pyplot as plt
      from numpy import log, exp
      
      scale = median = exp(mu)
      mode = exp(mu - sigma**2)
      mean = exp(mu + (sigma**2/2))
      shape = sigma
      a,b = lognorm.interval(area, shape, loc = 0, scale = np.exp(mu))
      x = np.linspace(a,b,1000000)
      mode = exp(mu - shape**2)
      mean = exp(mu + (shape**2/2))
      pdf = lognorm.pdf(x,shape, loc = 0, scale = scale)
      plt.figure(figsize = (12,8), dpi=200)
      plt.plot(x,pdf, label = 'PDF ($\sigma = %.2f$)' %shape)
      plt.vlines(mode,0,pdf.max(), linestyle = ':', label = 'Mode = %.2f' %mode)
      plt.vlines(mean,0,lognorm.pdf(mean,shape,loc=0,scale=scale),color = 'green', linestyle = '--',label = 'Mean = %.2f' %mean)
      plt.vlines(median,0,lognorm.pdf(median,shape,loc=0,scale=scale),color = 'blue', label = 'Median = %.2f' %median)
      plt.legend(loc = 1)
      
      plt.legend(prop = {'size':18})
      plt.xlabel('x', fontsize = 45)
      plt.ylabel('Probability', fontsize = 45)
      plt.xticks(size = 18)
      plt.yticks(size = 18)
      plt.tight_layout()
      #plt.savefig("/media/rafael/DiscoCompartilhado/Faculdade/Bolsa - Atlas/KernelDensityEstimation-Python/Kernel-Discretization-Processes/Figures_log/sigma_%.2f.png" %sigma)
      
def mediaMovel(x,n):
      for i in range(len(x)):
            if i < n//2:
                  x[i] = mean(x[:n//2])
            else:
                  x[i] = mean(x[i-n//2:i+n//2])
      return x