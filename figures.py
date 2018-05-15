def generic(nest, mu = 0, sigma = 1, outlier = 0, distribuition = 'normal'):
	"""
	Returns a generic plot of a selected distribuition.

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
	"""
