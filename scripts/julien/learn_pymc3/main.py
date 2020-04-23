
from scripts.julien.learn_pymc3.BayesianModel import BayesianModel

from scripts.julien.learn_pymc3.PPLModelCreator import PPLModel



if __name__ == "__main__":
    json_file = '../data/bnlearn_example_grade.json'
    path = "../data/"
    file = 'burglary_cleaned.csv'
    #file = 'students_cleaned.csv'
    #file = 'grade_model.csv'
    #file = 'mixture_gaussian_cleaned.csv'
    file = "allbus_cleaned.csv"
    #file = "titanic_orig.csv"
    #file = "bank_cleaned.csv"
    # file = "sprinkler_cleaned.csv"
    file = path + file

    # clean data
    """
    from bayesian_network_learning.util.DataTransformer import DataTransformer
    dt = DataTransformer()
    dt.transform(file, ending_comma=False, discrete_variables=['cloudy','rain','sprinkler','grass_wet'])
    """

    continuous_variables = []
    discrete_variables = ['sex', 'lived_abroad']
    whitelist = []
    blacklist = []
    #blacklist_edges = [('income', 'age')]
    #continuous_variables = ['income', 'age']
    #whitelist = [('sex', 'educ')]

    algorithms = ["tabu", "hc", "gs", "iamb", "fast.iamb", "inter.iamb", "mmpc"]
    scores = ["loglik-cg", "aic-cg", "bic-cg", "pred-loglik-cg"]
    could_not_fit = 0
    for algo in algorithms:
        for score in scores:
            model_name = f"allbus_{algo}_{score}".replace("-", "").replace(".", "")
            ppl_model = PPLModel(model_name, file, discrete_variables=discrete_variables, verbose=False, algo=algo,
                                 score=score)
            could_not_fit += ppl_model.generate_pymc(model_name=model_name, save=True, output_file="/home/julien/PycharmProjects/modelbase/scripts/julien/pymc_models_allbus2.py")
    print("COULD NOT FIT: ", could_not_fit)


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