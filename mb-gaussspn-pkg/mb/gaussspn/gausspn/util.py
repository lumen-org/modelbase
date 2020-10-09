from mb.modelbase import SumNode
from mb.modelbase import NormalLeafNode
from mb.modelbase import MultiNormalLeafNode

def count_nodes(network):
	nextnodes = [network.root.children[0]]
	count = 0
	while len(nextnodes) > 0:
		node = nextnodes.pop()
		count += 1
		nextnodes.extend(node.children)
	return count

def count_params(network):
	nextnodes = [network.root.children[0]]
	count = 0
	while len(nextnodes) > 0:
		node = nextnodes.pop()
		if type(node) == SumNode:
			count += len(node.children)
		elif type(node) == NormalLeafNode:
			count += 2
		elif type(node) == MultiNormalLeafNode:
			k = len(node.scope)
			count += k*(k+3)//2
		nextnodes.extend(node.children)
	return count
	
