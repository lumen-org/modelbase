######################################
# Chagos rats d15N
######################################
def create_chagos_rats_d15n(modelname='Chagos Rats d15N', fit=True, testcasedata_path=''):
    if fit:
        modelname = modelname+'_fitted'

    # Return list of unique items and an index of their position in L
    def indexall(L):
        poo = []
        for p in L:
            if not p in poo:
                poo.append(p)
        Ix = np.array([poo.index(p) for p in L])
        return poo,Ix

    # Return list of unique items and an index of their position in long, relative to short
    def subindexall(short,long):
        poo = []
        out = []
        for s,l in zip(short,long):
            if not l in poo:
                poo.append(l)
                out.append(s)
        return indexall(out)

    # Function to standardize covariates
    def stdize(x):
        return (x-np.mean(x))/(2*np.std(x))

    # Import age and growth data
    path = testcasedata_path + 'Chagos_isotope_data.csv'
    if not os.path.isfile(path):
        print('Training data for Chagos rats not found. Store the training data in ' + testcasedata_path)
    xdata = pd.read_csv(path)

    # Make arrays locally available
    organism = xdata.Tissue.values
    Organism,Io = indexall(organism)
    norg = len(Organism)

    It = xdata.Treatment.values=='Rats'

    atoll = xdata.Atoll.values
    island = xdata.Island.values

    Atoll,Ia = subindexall(atoll,island)
    natoll = len(Atoll)
    Island,Is = indexall(island)
    nisland = len(Island)
    TNI = np.log(np.array([xdata.TNI_reef_area[list(island).index(i)] for i in Island]))
    ReefArea = np.log(np.array([xdata.Reef_area[list(island).index(i)] for i in Island]))

    # Distance to shore in metres
    Dshore_ = xdata.To_shore_m.values
    Dshore_[np.isnan(Dshore_)] = 0
    Dshore = stdize(Dshore_)
    Dshore[Dshore<0] = 0

    d15N = xdata.cal_d15N.values

    basic_model = pm.Model()
    It = It.astype(int)

    data = pd.DataFrame({'Is': Is, 'Io': Io, 'It': It, 'Dshore': Dshore, 'Yi': d15N})
    Is = theano.shared(Is)
    It = theano.shared(It)
    Io = theano.shared(Io)
    Dshore = theano.shared(Dshore)

    with basic_model:
        # Global prior
        γ0 = pm.Normal('Mean_d15N', mu=0.0, tau=0.001)
        # Reef-area effect
        # γ1 = pm.Normal('ReefArea', mu=0.0, tau=0.001)

        # Island-level model
        # γ = γ0+γ1*ReefArea
        γ = γ0
        # Inter-island variablity
        σγ = pm.Uniform('SD_reef', lower=0, upper=100)
        τγ = σγ ** -2
        β0 = pm.Normal('Island_d15N', mu=γ, tau=τγ, shape=nisland)

        # Organism mean (no rats)
        β1_ = pm.Normal('Organism_', mu=0.0, tau=0.001, shape=norg - 1)
        β1 = theano.tensor.set_subtensor(theano.tensor.zeros(shape=norg)[1:], β1_)

        # Organism-specific rat effects
        β2 = pm.Normal('Rat_effect_', mu=0.0, tau=0.001, shape=norg)

        # Distance to shore
        β3 = pm.Normal('Dist_to_shore', mu=0.0, tau=0.001)

        # Mean model
        μ = β0[Is] + β1[Io] + β2[Io] * It + β3 * Dshore

        # Organism-specific variance
        σ = pm.Uniform('SD', lower=0, upper=100)
        τ = σ ** -2

        # Likelihood
        Yi = pm.StudentT('Yi', nu=4, mu=μ, lam=τ, observed=d15N)

######################################
# Chagos rats vonB
######################################
def create_chagos_rats_vonB(modelname='Chagos Rats vonB', fit=True, testcasedata_path=''):
    if fit:
        modelname = modelname+'_fitted'
    # Helper functions
    def indexall(L):
        poo = []
        for p in L:
            if not p in poo:
                poo.append(p)
        Ix = np.array([poo.index(p) for p in L])
        return poo,Ix

    def subindexall(short,long):
        poo = []
        out = []
        for s,l in zip(short,long):
            if not l in poo:
                poo.append(l)
                out.append(s)
        return indexall(out)

    match = lambda a, b: np.array([ b.index(x) if x in b else None for x in a ])

    path = testcasedata_path + 'chagos_otolith.csv'
    if not os.path.isfile(path):
        print('Training data for Chagos rats not found. Store the training data in ' + testcasedata_path)
    # Import age and growth data
    xdata = pd.read_csv(path)

    # Site
    site = xdata.Site.values

    # Fish ID
    ID = xdata.OtolithID.values

    # Length
    TL = xdata.TL.values
    lTL = np.log(TL)
    maxTL = max(TL)
    minTL = min(TL)

    # Bird or rat island
    Treatment,It = indexall(xdata.Treatment.values)

    # Age
    age = xdata.Age.values

    # Plotting age
    agex = np.linspace(min(age),max(age),num=100)

    Model = pm.Model()

    with Model:
        Linf = pm.Uniform('Linf',maxTL, maxTL*2)
        L0 = pm.Uniform('L0', 0, minTL)
        k0 = pm.Uniform('k0', 0.001, 1)
        k1 = pm.Normal('k1', 0, 10)
        σ = pm.Uniform('σ', 0, 1000)
        μ = theano.tensor.log(Linf-(Linf-L0)*theano.tensor.exp(-(k0+k1*It)*age))
        yi = pm.Normal('yi',μ, σ, observed=lTL)