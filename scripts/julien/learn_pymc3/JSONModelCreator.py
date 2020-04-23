from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from pandas import read_csv
import numpy as np


class JSONModelCreator(object):
    def __init__(self, file, whitelist=None, discrete_variables=[], continuous_variables=[], blacklist=None, score="",
                 algo=""):
        self.file = file
        self.whitelist = whitelist
        self.blacklist = blacklist
        self.discrete_variables = discrete_variables
        self.continuous_variables = continuous_variables
        # check if score and algorithm is available
        self.score = score
        self.algo = algo

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

        # Check for different scores
        # https://www.bnlearn.com/documentation/man/network.scores.html
        score = None
        if len(self.discrete_variables) == 0:
            if self.score in ["loglik-g", "aic-g", "bic-g", "pred-loglik-g", "bge"]:
                score = self.score
            else:
                if verbose:
                    print("Score not allowed, allowed scores are: ",
                          ["loglik-g", "aic-g", "bic-g", "pred-loglik-g", "bge"],
                          "\nWe use now bic-g.")
                score = "bic-g"
        elif len(self.continuous_variables) == 0:
            if self.score in ["loglik", "aic", "bic", "pred-loglik", "bde", "uniform", "bds", "bdla", "k2"]:
                score = self.score
            else:
                if verbose:
                    print("Score not allowed, allowed scores are: ",
                          ["loglik", "aic", "bic", "pred-loglik", "bde", "uniform", "bds", "bdla", "k2"],
                          "\nWe use now bic.")
                score = "bic"
        else:
            if self.score in ["loglik-cg", "aic-cg", "bic-cg", "pred-loglik-cg"]:
                score = self.score
            else:
                if verbose:
                    print("Score not allowed, allowed scores are: ",
                          ["loglik-cg", "aic-cg", "bic-cg", "pred-loglik-cg"], "\nWe use now bic-cg.")
                score = "bic-cg"
        # Check for allowed algorithms
        algo = None
        if self.algo in ["hc", "tabu", "gs", "iamb", "fast.iamb", "inter.iamb", "mmpc"]:
            algo = self.algo
        else:
            print("Algorithm not allowed, allowed algorithms are: ",
                  ["hc", "tabu", "gs", "iamb", "fast.iamb", "inter.iamb", "mmpc"],
                  "\nWe use now hc.")
        r_parameter = {
            'disc_vars': discrete_variables_r,
            'file': file,
            'whitelist': whitelist,
            'cont_vars': continuous_variables,
            'blacklist': blacklist,
            'score': score,
            'algo': algo
        }
        bnlearn_code = """
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
            stru = {algo}(data, whitelist=whitelist_edges, blacklist=blacklist_edges, score='{score}')
            # Structure has to be directed
            if (!directed(stru)){
                h <- stru
                while (!directed(h)){
                    tryCatch(expr={h <- pdag2dag(stru, sample(nodes(stru)))})
                }
                stru <- h
            } 
            # graphviz.plot(stru)
            # Fit the parameters
            fit <- bn.fit(stru, data)
            # Export as JSON
            library(jsonlite)
            model <- list()
            #data$data <- data
            model$structure <- stru
            model$parameter <- fit
            level <- list()
            for (var in names(data)) level[[var]] <- levels(data[[var]])        
            model$levels <- level
            json.file <- toJSON(model, force=TRUE)
            write(json.file,paste({file},'.json',sep=''))
                    """
        replaced_code = bnlearn_code
        for key, value in r_parameter.items():
            replaced_code = replaced_code.replace('{' + key + '}', str(value))
        try:
            t = SignatureTranslatedAnonymousPackage(replaced_code, "powerpack")
        except:
            if verbose:
                print("Score not working, we remove the keyword.\n")
            replaced_code = replaced_code.replace(", score=", ")#")
            try:
                t = SignatureTranslatedAnonymousPackage(replaced_code, "powerpack")
            except:
                print("Fatal Error, skip configuration")
                return (None, None, None)
        if verbose:
            print(replaced_code)
        return (self.file + ".json", self.discrete_variables, self.continuous_variables)
