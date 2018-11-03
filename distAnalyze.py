
def diffArea(nest, outlier = 0, data = 0, kinds = 'all', axis = 'probability', ROI = 20 , mu = 0, sigma = 1, weight = False, interpolator = 'linear', distribuition = 'normal',seed = None, plot = True):
    
    """
    Return an error area between a analitic function and a estimated discretization from a distribuition.

    Parameters
    ----------
    nest: int
        The number of estimation points.
    outlier: int, optional
        Is the point of an outlier event, e.g outlier = 50 will put an event in -50 and +50 if mu = 0.
        Defaut is 0
    data: int, optional
        If data > 0, a randon data will be inserted insted analitcs data.
        Defaut is 0.
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
        
    Returns
    -------
    a, [b,c]: float and float of ndarray. area,[probROIord,areaROIord]
       returns the sum of total error area and the 'x' and 'y' values.   
    

    """    
    import numpy as np
    from scipy.stats import norm, lognorm
    from scipy.interpolate import interp1d
    from numpy import  exp
    import matplotlib.pyplot as plt
    from statsmodels.distributions import ECDF
    from distAnalyze import pdf, dpdf, ddpdf, PDF, dPDF, ddPDF

    area = []
    n = []
    data = int(data)
    if distribuition == 'normal': 
        outlier_inf = outlier_sup = outlier
    elif distribuition == 'lognormal': 
        outlier_inf = 0
        outlier_sup = outlier

    ngrid = int(1e6)
    truth = pdf
        
    if axis == 'probability':
        truth1 = pdf
    elif axis == 'derivative':
        truth1 = dpdf
    elif axis == '2nd_derivative':
        truth1 = ddpdf
    elif axis == 'X':
        truth1 = lambda x,mu,sigma,distribuition: x
    #else: return 'No valid axis'
            
    probROIord = {}
    areaROIord = {}
    div = {}
    if seed is not None:
          np.random.set_state(seed)
    if data:
          if distribuition == 'normal':
                d = np.random.normal(mu,sigma,data)
          elif distribuition == 'lognormal':
                d = np.random.lognormal(mu, sigma, data)
          
                
          
    if kinds == 'all': 
        kinds = ['Linspace', 'CDFm', 'PDFm', 'iPDF1', 'iPDF2']
    elif type(kinds) == str:
        kinds = [kinds]

    for kind in kinds:
        if distribuition == 'normal':
              inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
            
        elif distribuition == 'lognormal':
              inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = exp(mu))
              inf = lognorm.pdf(sup, sigma, loc = 0, scale = np.exp(mu))
              inf = lognorm.ppf(inf, sigma, loc = 0, scale = np.exp(mu))

        xgrid = np.linspace(inf,sup,ngrid)
        xgridROI = xgrid.reshape([ROI,ngrid//ROI])
        
        dx = np.diff(xgrid)[0]
        
        if kind == 'Linspace':
            if not data:  
                  xest = np.linspace(inf-outlier_inf,sup+outlier_sup,nest)
            else:
                  if distribuition == 'normal':
                        #d = np.random.normal(loc = mu, scale = sigma, size = data)
                        inf,sup = min(d),max(d)
                        xest = np.linspace(inf-outlier_inf,sup+outlier_sup,nest)
                  elif distribuition == 'lognormal':
                        #d = np.random.lognormal(mean = mu, sigma = sigma, size = data)
                        inf,sup = min(d),max(d)
                        xest = np.linspace(inf-outlier_inf,sup+outlier_sup,nest)
                        
            yest = pdf(xest,mu,sigma,distribuition)
            
        elif kind == 'CDFm':
            eps = 5e-5
            yest = np.linspace(0+eps,1-eps,nest)
            if distribuition == 'normal':
                if not data:
                      xest = norm.ppf(yest, loc = mu, scale = sigma)
                      yest = pdf(xest,mu,sigma,distribuition)
                else:
                      #d = np.random.normal(loc = mu, scale = sigma, size = data)
                      ecdf = ECDF(d)
                      inf,sup = min(d),max(d)
                      xest = np.linspace(inf,sup,data)
                      yest = ecdf(xest)
                      interp = interp1d(yest,xest,fill_value = 'extrapolate', kind = 'nearest')
                      yest = np.linspace(eps,1-eps,nest)
                      xest = interp(yest)
                
            elif distribuition == 'lognormal':
                if not data:
                      xest = lognorm.ppf(yest, sigma, loc = 0, scale = exp(mu))
                      yest = pdf(xest,mu,sigma,distribuition)
                else:
                      #d = np.random.lognormal(mean = mu, sigma = sigma, size = data)
                      ecdf = ECDF(d)
                      inf,sup = min(d),max(d)
                      xest = np.linspace(inf,sup,nest)
                      yest = ecdf(xest)
                      interp = interp1d(yest,xest,fill_value = 'extrapolate', kind = 'nearest')
                      yest = np.linspace(eps,1-eps,nest)
                      xest = interp(yest)
            
            
        elif kind == 'PDFm':
            xest, yest = PDF(nest,mu,sigma, distribuition, outlier, data, seed)
        elif kind == 'iPDF1':
            xest, yest = dPDF(nest,mu,sigma, distribuition, outlier, data, 10, seed)
        elif kind == 'iPDF2':
            xest, yest = ddPDF(nest,mu,sigma, distribuition, outlier, data, 10, seed)      
       
        YY = pdf(xest,mu, sigma,distribuition)
        fest = interp1d(xest,YY,kind = interpolator, bounds_error = False, fill_value = (YY[0],YY[-1]))
        
        #fest = lambda x: np.concatenate([fest1(x)[fest1(x) != -1],np.ones(len(fest1(x)[fest1(x) == -1]))*fest1(x)[fest1(x) != -1][-1]])
            
        yestGrid = []
        ytruthGrid = []
        ytruthGrid2 = []
        divi = []
        
        for i in range(ROI):
            yestGrid.append([fest(xgridROI[i])])
            ytruthGrid.append([truth(xgridROI[i],mu,sigma,distribuition)])
            ytruthGrid2.append([truth1(xgridROI[i],mu,sigma,distribuition)])
            divi.append(len(np.intersect1d(np.where(xest >= min(xgridROI[i]))[0], np.where(xest < max(xgridROI[i]))[0])))

        diff2 = np.concatenate(abs((np.array(yestGrid) - np.array(ytruthGrid))*dx))
        #diff2[np.isnan(diff2)] = 0
        areaROI = np.sum(diff2,1)
        
        divi = np.array(divi)   
        divi[divi == 0] = 1
        
        try:
            probROI = np.mean(np.sum(ytruthGrid2,1),1)
        except:
            probROI = np.mean(ytruthGrid2,1)
        
        
        probROIord[kind] = np.sort(probROI)
        index = np.argsort(probROI)
        
        areaROIord[kind] = areaROI[index]
        #deletes = ~np.isnan(areaROIord[kind])
        #areaROIord[kind] = areaROIord[kind][deletes]
        #probROIord[kind] = probROIord[kind][deletes]
        
        area = np.append(area,np.sum(areaROIord[kind]))
        n = np.append(n,len(probROIord[kind]))
        div[kind] = divi[index]
        if plot:
            if weight:
                plt.logy(probROIord[kind],areaROIord[kind]*div[kind],'-o',label = kind, ms = 3)
            else: plt.plot(probROIord[kind],areaROIord[kind],'-o',label = kind, ms = 3)
            

            plt.yscale('log')
            plt.xlabel(axis)
            plt.ylabel('Error')
            plt.legend()
        
        #plt.title('%s - Pontos = %d, div = %s - %s' %(j,nest, divs,interpolator))
        
    return area,[probROIord,areaROIord]

def diffArea3(nest = None, outlier = 0, data = 0, kinds = 'all', axis = 'probability', ROI = 20 , mu = 0, sigma = 1, weight = False, interpolator = 'linear', distribuition = 'normal', plot3d = False, seed=None, hold = False):

    """
    Return an error area between a analitic function and a estimated discretization from a distribuition.

    Parameters
    ----------
    nest: ndarray, int, optional
        The array of the estimation points (e.g. nest = [100,200,300,400,500]).
        if nest = None: 
              nest = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 
                      150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 300, 
                      350, 400, 450, 500, 600, 700, 800, 900, 1000, 1500, 2000, 
                      2500, 3000, 3500, 4000, 4500, 5000]
        Defaut is None.
    data: int, optional
        If data > 0, a randon data will be inserted insted analitcs data.
        Defaut is 0.
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
        If True, a plot will be ploted with the analyzes in 3d with Nest x error x axis
        If False, a 2d plot will be ploted with Nest x Area
        Defaut is False
    hold: bool, optional
        If False, a new a plot will be ploted in a new figure, else, a plot 
        will be ploted in the same figure.
        Defaut is False.
        
    Returns
    -------
    a,b,c
    return the number of estimation points, error area and distribuition if plot3 is True

    """    
    
    #nest1 = np.concatenate([list(range(10,250,10)),list(range(250,550,50)),list(range(600,1100,100)),list(range(1500,5500,500))])
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from distAnalyze import diffArea

    if nest is None:     
       nest = np.concatenate([list(range(10,250,10)),list(range(250,550,50)),list(range(600,1100,100)),list(range(1500,5500,500))])

    if seed is not None:
        np.random.set_state(seed)
    else:
        seed = np.random.get_state()
        
    if kinds == 'all': 
        kinds = ['Linspace', 'CDFm', 'PDFm', 'iPDF1', 'iPDF2']
    elif type(kinds) == str:
        kinds = [kinds]
  
    probROIord = {}
    areaROIord = {}
    area = {}
    for n in nest:
        area[n],[probROIord[n],areaROIord[n]] = diffArea(n, outlier, data, kinds, axis, ROI, mu, sigma, weight, interpolator, distribuition, seed, plot = False)
        
    #x = np.sort(nest*ROI) #Nest
    #y = np.array(list(probROIord[nest[0]][list(probROIord[nest[0]].keys())[0]])*len(nest)) #Prob
    
    
    area2 = {kinds[0]:[]}
    for k in range(len(kinds)):
          area2[kinds[k]] = []
          for n in nest:
                area2[kinds[k]].append(area[n][k])
    
                       
    x,y = np.meshgrid(nest,list(probROIord[nest[0]][list(probROIord[nest[0]].keys())[0]]))
    area = area2
    
# =============================================================================
#     z = {} #error
#     
#     for k in kinds:
#           z[k] = []
#           for i in nest:
#                 z[k].append(areaROIord[i][k])
#           z[k] = np.reshape(np.concatenate(z[k]),x.shape,'F')
# =============================================================================
    if plot3d:
          fig = plt.figure()
          ax = fig.gca(projection='3d')
                
          z = {} #error
          for k in kinds:
               z[k] = []
               for i in nest:
                     z[k].append(areaROIord[i][k])
               z[k] = np.reshape(np.concatenate(z[k]),x.shape,'F')
               ax.plot_surface(x,y,np.log10(z[k]),alpha = 0.4, label = k, antialiased=True)
      
          ax.set_xlabel('Nº of estimation points', fontsize = 20)
          ax.set_xticks(nest)
          ax.set_ylabel(axis, fontsize = 20)
          ax.zaxis.set_rotate_label(False)
          ax.set_zlabel('Sum of errors', fontsize = 20, rotation = 90)
          ax.view_init(20, 225)
          plt.draw()
          #ax.yaxis.set_scale('log')
          plt.legend(prop = {'size':25}, loc = (0.6,0.5))
          ax.show()
          return x,y,np.log10(z[k])
    else:
          if not hold:
                plt.figure(figsize = (12,8),dpi = 100)
          for k in kinds:
                plt.plot(nest,area[k], 'o-', label = k)
          plt.xlabel('Nº of estimation points', fontsize = 30)
          plt.ylabel('Error', fontsize = 30)
          plt.legend(prop = {'size':18})
          plt.yscale('log')
          plt.tick_params(labelsize = 18)
          plt.tight_layout()
          #plt.savefig("/media/rafael/DiscoCompartilhado/Faculdade/Bolsa - Atlas/KernelDensityEstimation-Python/Kernel-Discretization-Processes/Figures_log/error_sigma_%.2f_interpolator_%s.png"%(sigma,interpolator))

          return nest, area 
                
      
# =============================================================================
# x, y = np.meshgrid(nest,sigma)
# 
# z = {}
# kinds = ['Linspace', 'CDFm', 'PDFm', 'iPDF1', 'iPDF2']
# for k in kinds:
# 	z[k] = []
# 	for i in range(len(sigma)):
# 		z[k].append(area2[i][k])
# 	z[k] = np.reshape(np.concatenate(z[k]),x.shape)
# 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# 
# for k in kinds:
#     ax.plot_surface(x,y,np.log10(z[k]),alpha = 0.4, label = k, antialiased=True)
# 
# =============================================================================

def PDF(pts,mu,sigma, distribuition, outlier = 0, data = 0, seed = None):
    from scipy.stats import norm, lognorm
    import numpy as np
    from scipy.interpolate import interp1d
    from someFunctions import ash
    eps = 5e-5
    
    if distribuition == 'normal':
        outlier_inf = outlier_sup = outlier
        if not data:   
              inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
              
      
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
              
        else:
              np.random.set_state(seed)
              d = np.random.normal(mu,sigma,data)
              inf,sup = min(d)-outlier_inf,max(d)+outlier_sup
              #yest,xest = np.histogram(d,bins = 'fd',normed = True)
              xest,yest = ash(d)
              xest = np.mean(np.array([xest[:-1],xest[1:]]),0)
              M = np.where(yest == max(yest))[0][0]
              m = np.where(yest == min(yest))[0][0]
              interpL = interp1d(yest[:M+1],xest[:M+1], assume_sorted = False, fill_value= 'extrapolate')
              interpH = interp1d(yest[M:],xest[M:], assume_sorted= False, fill_value='extrapolate')
              
              y1 = np.linspace(yest[m]+eps,yest[M],pts//2+1)
              x1 = interpL(y1)
              
              y2 = np.flip(y1,0)
              x2 = interpH(y2)

    elif distribuition == 'lognormal':
        outlier_inf = 0
        outlier_sup = outlier
        inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
        inf = lognorm.pdf(sup, sigma, loc = 0, scale = np.exp(mu))
        inf = lognorm.ppf(inf, sigma, loc = 0, scale = np.exp(mu))  
        if not data:  
              
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
        else:
              np.random.set_state(seed)
              d = np.random.lognormal(mu,sigma,data)
              #inf,sup = min(d)-outlier_inf,max(d)+outlier_sup
              #yest,xest = np.histogram(d,bins = 'fd',normed = True)
              #xest = np.mean(np.array([xest[:-1],xest[1:]]),0)
              xest,yest = ash(d)
              yest = yest[xest<sup]
              xest = xest[xest<sup]
              M = np.where(yest == max(yest))[0][0]
              m = np.where(yest == min(yest))[0][0]
              interpL = interp1d(yest[:M+1],xest[:M+1], fill_value = 'extrapolate')
              interpH = interp1d(yest[M:],xest[M:])
              
              y1 = np.linspace(yest[m]+eps,yest[M],pts//2+1)
              x1 = interpL(y1)
              
              y2 = np.flip(y1,0)
              x2 = interpH(y2)
       
        
    X = np.concatenate([x1[:-1],x2])
    Y = np.concatenate([y1[:-1],y2])
    
    return X,Y

def dPDF(pts,mu,sigma, distribuition, outlier = 0, data = 0, n=10, seed = None):
    import numpy as np
    from scipy.interpolate import interp1d
    from distAnalyze import dpdf, mediaMovel
    from scipy.stats import norm, lognorm
    from someFunctions import ash
    
    eps = 5e-5
    ngrid = int(1e6)

    if distribuition == 'normal':
        outlier_inf = outlier_sup = outlier  
        if not data:  
              inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
              x = np.linspace(inf-outlier_inf,sup+outlier_sup,ngrid)
              y = dpdf(x,mu,sigma,distribuition)
              
        else:
              np.random.set_state(seed)
              d = np.random.normal(mu,sigma,data)
              inf,sup = min(d)-outlier_inf,max(d)+outlier_sup
              
              #y,x = np.histogram(d,bins = 'fd',normed = True)
              #x = np.mean(np.array([x[:-1],x[1:]]),0)
              x,y = ash(d)
              
              y = abs(np.diff(y))
              #y = abs(np.diff(mediaMovel(y,n)))
              x = x[:-1]+np.diff(x)[0]/2
              
    elif distribuition == 'lognormal':
        outlier_inf = 0
        outlier_sup = outlier
        inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
        inf = lognorm.pdf(sup, sigma, loc = 0, scale = np.exp(mu))
        inf = lognorm.ppf(inf, sigma, loc = 0, scale = np.exp(mu))
        if not data:
              
              x = np.linspace(inf-outlier_inf,sup+outlier_sup,ngrid)
              y = dpdf(x,mu,sigma,distribuition)
        else:
              np.random.set_state(seed)
              d = np.random.lognormal(mu,sigma,data)
              #inf,sup = min(d)-outlier_inf,max(d)+outlier_sup
              
             # y,x = np.histogram(d,bins = 'fd',normed = True)
             # x = np.mean(np.array([x[:-1],x[1:]]),0)
              x,y = ash(d)
              y = y[x<sup]
              x = x[x<sup]
              
              #y = abs(np.diff(mediaMovel(y,n)))
              y = abs(np.diff(y))
              x = x[:-1]+np.diff(x)[0]/2
              y = y/(np.diff(x)[0]*sum(y))
    #dy = lambda x,u,s : abs(1/(s**3*sqrt(2*pi))*(u-x)*np.exp(-0.5*((u-x)/s)**2))
    
  
    cdf = np.cumsum(y)
       
    #cdf = np.sum(np.tri(len(x))*y,1)    
    #cdf = np.concatenate(cdf)
    cdf = cdf/max(cdf)
    #time.time()-t
    
    interp = interp1d(cdf,x, fill_value = 'extrapolate')
    Y = np.linspace(eps,1-eps,pts)
    X = interp(Y)
    
    return X,Y

    
def ddPDF(pts,mu,sigma, distribuition, outlier = 0, data = 0, n=10, seed = None):
    import numpy as np
    from scipy.interpolate import interp1d
    from distAnalyze import ddpdf, mediaMovel
    from scipy.stats import norm, lognorm
    from someFunctions import ash
    eps = 5e-5 
    ngrid = int(1e6)
    #ddy = lambda x,u,s: abs(-(s**2-u**2+2*u*x-x**2)/(s**5*sqrt(2*pi))*np.exp(-0.5*((u-x)/s)**2))
    if distribuition == 'normal':
        outlier_inf = outlier_sup = outlier
        if not data:  
              inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
              x = np.linspace(inf-outlier_inf,sup+outlier_sup,ngrid)
              y = ddpdf(x,mu,sigma,distribuition)
        else:
              np.random.set_state(seed)
              d = np.random.normal(mu,sigma,data)
              inf,sup = min(d)-outlier_inf,max(d)+outlier_sup
              
              #y,x = np.histogram(d,bins = 'fd',normed = True)
              #x = np.mean(np.array([x[:-1],x[1:]]),0)
              
              x,y = ash(d)
              y = abs(np.diff(y,2))
              x = x[:-2]+np.diff(x)[0]
              
              #y = abs(np.diff(mediaMovel(y,n),2))
              #x = x[:-2]+np.diff(x)[0]
              y = y/(np.diff(x)[0]*sum(y))
        
    elif distribuition == 'lognormal':
        outlier_inf = 0
        outlier_sup = outlier
        inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
        inf = lognorm.pdf(sup, sigma, loc = 0, scale = np.exp(mu))
        inf = lognorm.ppf(inf, sigma, loc = 0, scale = np.exp(mu))
        if not data:  
              
              x = np.linspace(inf-outlier_inf,sup+outlier_sup,ngrid)
              y = ddpdf(x,mu,sigma,distribuition)
        else:
              np.random.set_state(seed)
              d = np.random.lognormal(mu,sigma,data)
              #inf,sup = min(d)-outlier_inf,max(d)+outlier_sup
              
             # y,x = np.histogram(d,bins = 'fd',normed = True)
             #x = np.mean(np.array([x[:-1],x[1:]]),0)
             
              x,y = ash(d)
              
              
              y = y[x<sup]
              x = x[x<sup]
              
              y = abs(np.diff(y,2))
            
              #y = abs(np.diff(mediaMovel(y,n),2))
              x = x[:-2]+np.diff(x)[0]
              y = y/(np.diff(x)[0]*sum(y))
       
    #cdf = np.sum(np.tri(len(x))*y,1)
    cdf = np.cumsum(y)
# =============================================================================
#     for i in range(1,ngrid):
#         cdf.append(y[i]+cdf[i-1])
    cdf = cdf/max(cdf)
#     
# =============================================================================
    interp = interp1d(cdf,x, fill_value = 'extrapolate')
    Y = np.linspace(eps,1-eps,pts)
    X = interp(Y)

    return X,Y



def pdf(x, u, s, distribuition):
    import numpy as np
    from numpy import pi, sqrt, log, exp, isnan
    from scipy.stats import norm, lognorm
    from scipy.interpolate import interp1d


    if distribuition == 'normal':
        #y = 1/(s*sqrt(2*pi))*exp(-0.5*((x-u)/s)**2)
        y = norm.pdf(x, loc = u, scale = s)

    elif distribuition == 'lognormal':
        #y = 1/(x*s*sqrt(2*pi))*exp(-(log(x)-u)**2/(2*s**2))
        #y[isnan(y)] = 0
        y = lognorm.pdf(x, s, loc = 0, scale = np.exp(u))
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



def mediaMovel(x,n):
      from numpy import mean
      for i in range(len(x)):
            if i < n//2:
                  x[i] = mean(x[:n//2])
            else:
                  x[i] = mean(x[i-n//2:i+n//2])
      return x

def crossValidation(ind1,numblock,k,aux):    
    '''
    ind1 = eventos
    numblock = quantidade de blocos
    k = divisão dos blocos
    aux = rotação
    '''
    
    import numpy as np
    
    inde = ind1
    
    event = int(len(inde)/numblock)
    
    blocksort = np.roll(range(1,numblock+1),(aux))
    
    indet = []
    
    for i in blocksort[0:len(blocksort)//k]:
        indet.append(inde[event*(i-1):event*i])
        
    indev = []
    
    for i in blocksort[len(blocksort)//k:]:
        indev.append(inde[event*(i-1):event*i])
        
    return [np.concatenate(indet), np.concatenate(indev)]