import numpy as np
from sklearn.cluster import KMeans

from mb_modelbase.models_core.gspn.spn import SPN, SumNode, ProductNode, BernoulliNode, CategoricalNode, GaussianNode


class KMeansLearnSPN(object):
    def __init__(self, min_number_of_samples=10, max_cluster=8):
        self.min_number_of_samples = min_number_of_samples
        self.max_cluster = max_cluster

    def _get_best_k_and_predictions(self, data):
        best_k = None
        best_score = None
        best_prediction = None
        max_cluster = np.min([len(data), self.max_cluster])
        max_cluster = np.max([max_cluster, 3])
        for i in range(2, max_cluster):
            kmeans = KMeans(n_clusters=i, random_state=0)
            predictions = kmeans.fit_predict(data)
            score = kmeans.score(data)
            if not best_score or best_score < score:
                best_score = score
                best_k = i
                best_prediction = predictions
        return best_k, best_score, best_prediction

    def _sum_and_product_score(self, data):
        _, sum_score, _ = self._get_best_k_and_predictions(data)
        _, product_score, _ = self._get_best_k_and_predictions(data.T)
        return sum_score, product_score

    def _learn_variable_and_return_node(self, variable, data, var_types):
        var_type = var_types[variable]
        if var_type == "Gaussian":
            child_node = GaussianNode(variable, [], {})
        elif var_type == "Category":
            child_node = CategoricalNode(variable, [], {})
        else:
            raise Exception(f"Variable type {var_type} not supported.")
        child_node.learn_parameter(data)
        return child_node

    def learn(self, spn, data, var_types, params):
        root = None
        # work_list hold data as (scope, current_data, parent_node)
        work_list = []
        # create the root of the SPN
        # data has the form (entry, feature), each element of the scope indicates one feature
        scope = [i for i in range(len(data.T))]
        sum_score, prod_score = self._sum_and_product_score(data)
        if sum_score > prod_score:
            root = SumNode(scope)
        else:
            root = ProductNode(scope)
        spn.set_root(root)
        work_list.append((scope, data, root))
        # stepwise creation of the SPN
        while work_list:
            print(work_list)
            scope, data, parent = work_list.pop(0)
            # learn parameters if only one element in the scope remains or if number of data is to small
            if len(scope) == 1 or len(data) <= self.min_number_of_samples:
                self._learn_leaf_nodes(scope, data, parent, var_types)
            else:
                if isinstance(parent, SumNode):
                    best_k, best_score, best_prediction = self._get_best_k_and_predictions(data)
                    # the data is strongly dependent and we can learn them individually
                    if len(np.unique(best_prediction)) == 1:
                        self._learn_leaf_nodes(scope, data, parent, var_types)
                    else:
                        for cluster, cluster_scope in self._get_cluster_and_scope(data, best_prediction, scope, product_node=False, k=best_k):
                            node = ProductNode(scope, [])
                            parent.add_child_with_weight(node, len(cluster)/len(data))
                            work_list.append((scope, cluster, node))
                else:
                    best_k, best_score, best_prediction = self._get_best_k_and_predictions(data.T)
                    # the data is strongly dependent and we can learn them individually
                    if len(np.unique(best_prediction)) == 1:
                        self._learn_leaf_nodes(scope, data, parent, var_types)
                    else:
                        for cluster, cluster_scope in self._get_cluster_and_scope(data.T, best_prediction, scope, product_node=True, k=best_k):
                            node = SumNode(cluster_scope, [], [])
                            parent.add_child(node)
                            work_list.append((cluster_scope, cluster.T, node))

    def _learn_leaf_nodes(self, scope, data, parent, var_types):
        for variable, cluster in zip(scope, data.T):
            node = ProductNode([variable], [])
            child_node = self._learn_variable_and_return_node(variable, cluster, var_types)
            node.add_child(child_node)
            if isinstance(parent, SumNode):
                parent.add_child_with_weight(node, 1 / len(scope))
            else:
                parent.add_child(node)

    def _get_cluster_and_scope(self, data, prediction, scope, product_node, k=2):
        cluster = []
        for index in range(k):
            cluster_tmp = np.array([i for i, j in zip(data, prediction) if j == index])
            if product_node:
                scope_tmp = [i for i, j in zip(scope, prediction) if j == index]
            else:
                scope_tmp = scope
            cluster.append([cluster_tmp, scope_tmp])
        return cluster
