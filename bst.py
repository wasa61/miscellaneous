class Node(object):
    """节点类"""
    def __init__(self, data, left_child=None, right_child=None):
        self.data = data
        self.parent = None
        self.left_child = left_child
        self.right_child = right_child


class SearchBinaryTree(object):
    """二叉树类"""
    def __init__(self):
        self.__root = None
        self.prefix_branch = '├'
        self.prefix_trunk = '|'
        self.prefix_leaf = '└'
        self.prefix_empty = ''
        self.prefix_left = '─L─'
        self.prefix_right = '─R─'

    def is_empty(self):
        return not self.__root

    @property
    def root(self):
        return self.__root

    @root.setter
    def root(self, value):
        self.__root = value if isinstance(value, Node) else Node(value)

    def show_tree(self):
        if self.is_empty():
            print('空二叉树')
            return
        print('-' * 20)
        print(self.__root.data)
        self.__print_tree(self.__root)
        print('-' * 20)

    def __print_tree(self, node, prefix=None):
        if prefix is None:
            prefix, prefix_left_child = '', ''
        else:
            prefix = prefix.replace(self.prefix_branch, self.prefix_trunk)
            prefix = prefix.replace(self.prefix_leaf, self.prefix_empty)
            prefix_left_child = prefix.replace(self.prefix_leaf, self.prefix_empty)
        if self.has_child(node):
            if node.right_child is not None:
                print(prefix + self.prefix_branch + self.prefix_right + str(node.right_child.data))
                if self.has_child(node.right_child):
                    self.__print_tree(node.right_child, prefix + self.prefix_branch + ' ')
            else:
                print(prefix + self.prefix_branch + self.prefix_right)
            if node.left_child is not None:
                print(prefix + self.prefix_leaf + self.prefix_left + str(node.left_child.data))
                if self.has_child(node.left_child):
                    prefix_left_child += '  '
                    self.__print_tree(node.left_child, self.prefix_leaf + prefix_left_child)
            else:
                print(prefix + self.prefix_leaf + self.prefix_left)

    def has_child(self, node):
        return node.left_child is not None or node.right_child is not None

    def __str__(self):
        return str(self.__class__)

    def insert(self, root, value):
        """二叉搜索树插入节点-递归"""
        node = value if isinstance(value, Node) else Node(value)
        if self.is_empty():
            self.root = node
            return
        if root is None:
            root = node
        elif node.data < root.data:
            root.left_child = self.insert(root.left_child, value)
            root.left_child.parent = root
        elif node.data > root.data:
            root.right_child = self.insert(root.right_child, value)
            root.right_child.parent = root
        return root

    def insert_normal(self, value):
        """二叉搜索树插入节点-非递归"""
        node = value if isinstance(value, Node) else Node(value)
        if self.is_empty():
            self.root = node
            return
        else:
            current_node = self.root
            while True:
                if node.data < current_node.data:
                    if current_node.left_child:
                        current_node = current_node.left_child
                    else:
                        current_node.left_child = node
                        node.parent = current_node
                        break
                elif node.data > current_node.data:
                    if current_node.right_child:
                        current_node = current_node.right_child
                    else:
                        current_node.right_child = node
                        node.parent = current_node
                        break
                else:
                    break

if __name__ == '__main__':
    tree = SearchBinaryTree()
    data = [50, 77, 55, 29, 10, 50, 30, 66, 18, 80, 51, 18, 90]
    for i in data:
        tree.insert(tree.root, i)
        #tree.insert_normal(i)
    tree.show_tree()
