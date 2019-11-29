from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from pandas import read_csv
import numpy as np


class JSONModelCreator(object):
    def __init__(self, file, whitelist=None, discrete_variables=[], continuous_variables=[], blacklist=None):
        self.file = file
        self.whitelist = whitelist
        self.blacklist = blacklist
        self.discrete_variables = discrete_variables
        self.continuous_variables = continuous_variables

    def get_vars(self):
        variables = []
        data = read_csv(self.file)
        for name, _ in zip(data.columns, np.array(data).T):
            variables.append(name)
        return variables

    def generate_model_as_json_with_r(self, verbose=False):
        vars = self.get_vars()
        for var in vars:
            if var not in self.discrete_variables and var not in self.continuous_variables:
                self.continuous_variables.append(var)
        discrete_variables_r = "c(" + ",".join(["'" + var + "'" for var in self.discrete_variables]) + ")"
        continuous_variables = "c(" + ",".join([f"'{var}'" for var in self.continuous_variables]) + ")"

        whitelist = "data.frame()"
        from_nodes = []
        to_nodes = []
        for start, end in self.whitelist:
            from_nodes.append(f"'{start}'")
            to_nodes.append(f"'{end}'")

        if len(from_nodes) > 0:
            assert len(from_nodes) == len(to_nodes)
            from_nodes = "c(" + ",".join(from_nodes) + ")"
            to_nodes = "c(" + ",".join(to_nodes) + ")"
            whitelist = f"data.frame(from={from_nodes}, to={to_nodes})"
        else:
            whitelist = "NULL"

        blacklist = "data.frame()"
        from_nodes = []
        to_nodes = []
        for start, end in self.blacklist:
            from_nodes.append(f"'{start}'")
            to_nodes.append(f"'{end}'")

        if len(from_nodes) > 0:
            assert len(from_nodes) == len(to_nodes)
            from_nodes = "c(" + ",".join(from_nodes) + ")"
            to_nodes = "c(" + ",".join(to_nodes) + ")"
            blacklist = f"data.frame(from={from_nodes}, to={to_nodes})"
        else:
            blacklist = "NULL"

        file = "'" + self.file + "'"
        bnlearn = """
            library(bnlearn)
            # Load Data
            data <- read.csv(paste({file},sep=''))
            # Setting discrete variables
            discrete_variables <- {disc_vars}
            whitelist_continuous_variables <- {cont_vars}
            for (var in discrete_variables)  data[[var]] <- factor(data[[var]], c(sort(unique(data[[var]]))))
            for (var in whitelist_continuous_variables)  data[[var]] <- as.numeric(data[[var]])  
            # Set whitelist_edges
            whitelist_edges = {whitelist}
            blacklist_edges = {blacklist}
            # Learn the structure
            m.hc = hc(data, whitelist=whitelist_edges, blacklist=blacklist_edges) #, score="log-cg")
            # graphviz.plot(m.hc)
            # Fit the parameters
            fit <- bn.fit(m.hc, data)
            # Export as JSON
            library(jsonlite)
            model <- list()
            #data$data <- data
            model$structure <- m.hc
            model$parameter <- fit
            level <- list()
            for (var in names(data)) level[[var]] <- levels(data[[var]])        
            model$levels <- level
            json.file <- toJSON(model, force=TRUE)
            write(json.file,paste({file},'.json',sep=''))
                    """.format(disc_vars=discrete_variables_r, file=file, whitelist=whitelist, cont_vars=continuous_variables, blacklist=blacklist)
        if verbose:
            print(bnlearn)
        t = SignatureTranslatedAnonymousPackage(bnlearn, "powerpack")
        return (self.file + ".json", self.discrete_variables, self.continuous_variables)
