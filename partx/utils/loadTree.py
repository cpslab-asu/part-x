import pickle

def load_tree(tree_name):
    """Load the tree

    Args:
        tree_name ([type]): Load a tree for a particular replication

    Returns:
        [type]: tree
    """
    with open(tree_name, "rb") as f:
        ftree = pickle.load(f)
    # f.close()
    return ftree