from anytree import Node, RenderTree, search

def create_tree(data, head_node_name):
    """creates a representation of the tree. For each head node this function gets called seperately
    
    Arguments:
        data: the json input file as string
        head_node_name: the name of the head node for which the tree gets created
    
    Returns:
        the tree for the head_node
    """
    head_node = Node(head_node_name, case=None, parameter=None,
                     parent_name=None, number=0, normal=False, factor=None)
    wannabe_nodes = []
    for key in data[head_node_name].keys():
        option = data[head_node_name][key]
        is_normal_distribution = False
        factor = None
        if (option["parent"] != None):
            name = option["name"]
            if ("parameter" in option):
                parameter_str = option["parameter"]
                # parameter with normal distribution
                if (parameter_str.startswith("[")):
                    without_brackets = (parameter_str[1:-1])
                    parts = without_brackets.split()
                    parameter = (float(parts[0]), float(parts[1]))
                    is_normal_distribution = True
                else:  # parameter with single probability
                    parameter = float(parameter_str)
            elif ("factor" in option):
                parameter = None
                factor = float(option["factor"])
            else:
                parameter = None

            if ("case" in option):
                case = int(option["case"])
            else:
                case = None
            node_content = Node(name, case=case, parameter=parameter, parent_name=option["parent"],
                                number=int(key), normal=is_normal_distribution, factor=factor)
            wannabe_nodes.append(node_content)

    wannabe_nodes.sort(key=lambda x: x.parent_name)
    for wannabe_node in wannabe_nodes:
        desired_parent_node_number = wannabe_node.parent_name
        parent_node = _find_node_by_number(
            head_node, desired_parent_node_number)
        wannabe_node.parent = parent_node
    return head_node

def _print_tree(tree):
    """Debug Function to show the created tree, currently not used

    Arguments:
        tree: The tree to print
    """
    for pre, _, node in RenderTree(tree):
        print("%s%s(%s)[%s: %s]" % (pre, node.name, node.case, node.number, node.parameter))


def _find_node_by_number(tree, number):
    """search a node in a tree by a node number

    Arguments:
        tree: the tree to search in
        number: the parameter for the search

    Returns:
        the node which has the number as attribute
    """
    res = search.find_by_attr(tree, name="number", value=number)
    if res == None:
        print("error finding node", number)
        exit()
    return res