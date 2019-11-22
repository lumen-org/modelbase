
from scripts.julien.learn_pymc3.BayesianModel import BayesianModel



if __name__ == "__main__":
    json_file = '../data/bnlearn_example_grade.json'
    path = "/home/julien/PycharmProjects/lumen/modelbase/scripts/julien/data/"
    file = 'burglary_cleaned.csv'
    #file = 'students_cleaned.csv'
    #file = 'grade_model.csv'
    #file = 'mixture_gaussian_cleaned.csv'
    file = "allbus_cleaned.csv"
    file = "titanic.csv"
    # file = "sprinkler_cleaned.csv"
    file = path + file

    # clean data
    """
    from bayesian_network_learning.util.DataTransformer import DataTransformer
    dt = DataTransformer()
    dt.transform(file, ending_comma=False, discrete_variables=['cloudy','rain','sprinkler','grass_wet'])
    """

    continuous_variables = []
    whitelist = []
    blacklist = []
    #blacklist = [('income', 'age')]
    continuous_variables = ['income', 'age']
    whitelist = [('sex', 'educ')]
    bayesian_model = BayesianModel(continuous_variables=continuous_variables, whitelist=whitelist, blacklist=blacklist)
    bayesian_model.learn_through_r(file, relearn=True, verbose=True)

    descr = bayesian_model.get_graph_description()
    print(descr)

    bayesian_model.get_graph().export_as_graphviz("allbus_4", view=True)

    from scripts.julien.learn_pymc3.ProbParameter import generate_prob_graphs, is_similar, merge_nodes, print_prob_table

    if True:

        generate_prob_graphs(bayesian_model)
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
        from scripts.julien.learn_pymc3.BlogCreator import BlogCreator
        from scripts.julien.learn_pymc3.PyMCCreator import PyMCCreator
        # """
        bc = BlogCreator(bayesian_model)
        print("#### BLOG ####\n", bc.generate(), sep="")
        """
        """
        pc = PyMCCreator(bayesian_model)
        print("#### PYMC3 ####\n", pc.generate(), sep="")

        print()
        print(f"Learned bayesian network learned with {bayesian_model.get_number_of_parameter()} parameter.")
        # """