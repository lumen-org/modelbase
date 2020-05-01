
import pandas as pd
from scripts.julien.learn_pymc3.BayesianModel import BayesianModel

from scripts.julien.learn_pymc3.PPLModelCreator import PPLModel




if __name__ == "__main__":
    json_file = '../data/bnlearn_example_grade.json'
    path = "../data/"
    file = 'burglary_cleaned.csv'
    #file = 'students_cleaned.csv'
    #file = 'grade_model.csv'
    #file = 'mixture_gaussian_cleaned.csv'
    file = "allbus_train.csv"
    #file = "titanic_orig.csv"
    #file = "bank_cleaned.csv"
    # file = "sprinkler_cleaned.csv"
    file = path + file

    # clean allbus data and save them in a new file
    allbus_forward_map = {'sex': {'Female': 0, 'Male': 1}, 'eastwest': {'East': 0, 'West': 1},
                          'lived_abroad': {'No': 0, 'Yes': 1},
                          'happiness': {'h0': 0, 'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6, 'h7': 7, 'h8': 8,
                                        'h9': 9, 'h10': 10}}
    allbus_data = pd.read_csv(file)
    for feature, feature_map in allbus_forward_map.items():
        allbus_data[feature] = pd.Series(allbus_data[feature]).map(feature_map)
    file = file[:-4] + "_cleaned.csv"
    allbus_data.to_csv(file, index=False)
    """
    from bayesian_network_learning.util.DataTransformer import DataTransformer
    dt = DataTransformer()
    dt.transform(file, ending_comma=False, discrete_variables=['cloudy','rain','sprinkler','grass_wet'])
    """

    continuous_variables = []
    discrete_variables = ['sex','eastwest','happiness','lived_abroad']
    whitelist = []
    blacklist = []
    #blacklist_edges = [('income', 'age')]
    #continuous_variables = ['income', 'age']
    #whitelist = [('sex', 'educ')]


    algorithms = ["tabu", "hc", "gs", "iamb", "fast.iamb", "inter.iamb"]
    scores = ["loglik-cg", "bic-cg"]
    could_not_fit = 0
    number_of_fitted = 0
    for algo in algorithms:
        if algo in ["tabu", "hc"]:
            scores = ["bic-cg", "aic-cg"]
        else:
            scores = [""]
        for score in scores:
            model_name = f"allbus_{algo}{score}".replace("-", "").replace(".", "")
            ppl_model = PPLModel(model_name, file, discrete_variables=discrete_variables, verbose=False, algo=algo,
                                 score=score)
            error = ppl_model.generate_pymc(model_name=model_name, save=True, output_file="/home/julien/PycharmProjects/modelbase/scripts/julien/pymc_models_allbus2.py")
            if error:
                could_not_fit += error
                print(algo, score)
            else:
                number_of_fitted += 1
    print("COULD NOT FIT: ", could_not_fit)
    print("COULD FIT: ", number_of_fitted)


    """
        conditional_node_order = bayesian_model.get_condition_node_order()
        bayesian_json = {}
        bayesian_json["conditional node order"] = [node.get_name() for node in conditional_node_order]
        
        for node in conditional_node_order:
            print_prob_table(node.get_parameter().get_prob_graph(), file_name=node.get_name(), view=False, simple=True)
            bayesian_json[node.get_name()] = node.get_parameter().to_json()
        import json
        print(json.dumps(bayesian_json))
        
        
        found = False
        
        
        for node in bayesian_model.get_graph().get_nodes():
            leafs = node.get_parameter().get_prob_graph().get_leafs()
            for leaf_a in leafs:
                for leaf_b in leafs:
                    if leaf_a is not leaf_b:
                        if is_similar(0.001, leaf_a, leaf_b):
                            merge_nodes(leaf_a, leaf_b)
                            found = True
                            break
                if found:
                    break
        
    """