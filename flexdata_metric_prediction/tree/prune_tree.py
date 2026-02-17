def pruneCastNodes(root):
    cur_nodes = [root]
    while 0 < len(cur_nodes):
        next_nodes = []
        for cur_node in cur_nodes:
            new_children = []
            for child in cur_node.children:
                if child.name == "cast":
                    if 0 < len(child.children):
                        assert len(child.children) == 1
                        new_children.append(child.children[0])
                else:
                    new_children.append(child)
                    next_nodes.append(child)
            cur_node.children = new_children
        cur_nodes = next_nodes
    return root
