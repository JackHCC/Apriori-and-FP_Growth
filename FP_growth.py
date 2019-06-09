# python3
# -*- coding: utf-8 -*-
# @Author  : hcc
# @Time    : 2019/6/9
from collections import defaultdict, namedtuple
import time

def find_frequent_itemsets(transactions, minimum_support, include_support=False):

    items = defaultdict(lambda: 0)

    for transaction in transactions:
        for item in transaction:
            items[item] += 1

    # 频繁1项集筛选
    items = dict((item, support) for item, support in items.items()
        if support >= minimum_support)

    def clean_transaction(transaction):
        transaction = filter(lambda v: v in items, transaction)
        transaction_list = list(transaction)   # 为了防止变量在其他部分调用，这里引入临时变量transaction_list
        transaction_list.sort(key=lambda v: items[v], reverse=True)
        return transaction_list

    #FPtree构建
    master = FPTree()
    for transaction in map(clean_transaction, transactions):
        master.add(transaction)

    def find_with_suffix(tree, suffix):
        for item, nodes in tree.items():
            support = sum(n.count for n in nodes)
            if support >= minimum_support and item not in suffix:

                found_set = [item] + suffix
                yield (found_set, support) if include_support else found_set

                cond_tree = conditional_tree_from_paths(tree.prefix_paths(item))
                for s in find_with_suffix(cond_tree, found_set):
                    yield s

    for itemset in find_with_suffix(master, []):
        yield itemset

class FPTree(object):

    Route = namedtuple('Route', 'head tail')

    def __init__(self):
        # 初始化根节点和分支
        self._root = FPNode(self, None, None)

        self._routes = {}

    @property
    def root(self):
        # 创建根节点
        return self._root

    def add(self, transaction):
        # 添加节点
        point = self._root

        for item in transaction:
            next_point = point.search(item)
            if next_point:
                #当前已存在节点
                next_point.increment()
            else:
                # 创建节点
                next_point = FPNode(self, item)
                point.add(next_point)

                # 更新链路
                self._update_route(next_point)

            point = next_point

    def _update_route(self, point):
        assert self is point.tree

        try:
            route = self._routes[point.item]
            route[1].neighbor = point # route[1] is the tail
            self._routes[point.item] = self.Route(route[0], point)
        except KeyError:
            # 开启新节点
            self._routes[point.item] = self.Route(point, point)

    def items(self):

        for item in self._routes.keys():
            yield (item, self.nodes(item))

    def nodes(self, item):

        try:
            node = self._routes[item][0]
        except KeyError:
            return

        while node:
            yield node
            node = node.neighbor

    def prefix_paths(self, item):

        def collect_path(node):
            path = []
            while node and not node.root:
                path.append(node)
                node = node.parent
            path.reverse()
            return path

        return (collect_path(node) for node in self.nodes(item))

    def inspect(self):
        #print('Tree:')
        self.root.inspect(1)

        #print
        #print('Routes:')
        for item, nodes in self.items():
            #print('  %r' % item)
            for node in nodes:
                print('    %r' % node)

def conditional_tree_from_paths(paths):
    tree = FPTree()
    condition_item = None
    items = set()

    for path in paths:
        if condition_item is None:
            condition_item = path[-1].item

        point = tree.root
        for node in path:
            next_point = point.search(node.item)
            if not next_point:
                # Add a new node to the tree.
                items.add(node.item)
                count = node.count if node.item == condition_item else 0
                next_point = FPNode(tree, node.item, count)
                point.add(next_point)
                tree._update_route(next_point)
            point = next_point

    assert condition_item is not None

    # 计算节点
    for path in tree.prefix_paths(condition_item):
        count = path[-1].count
        for node in reversed(path[:-1]):
            node._count += count

    return tree

class FPNode(object):

    def __init__(self, tree, item, count=1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None

    def add(self, child):

        if not isinstance(child, FPNode):
            raise TypeError("Can only add other FPNodes as children")

        if not child.item in self._children:
            self._children[child.item] = child
            child.parent = self

    def search(self, item):
        try:
            return self._children[item]
        except KeyError:
            return None

    def __contains__(self, item):
        return item in self._children

    @property
    def tree(self):
        return self._tree

    @property
    def item(self):
        return self._item

    @property
    def count(self):
        return self._count

    def increment(self):
        if self._count is None:
            raise ValueError("Root nodes have no associated count.")
        self._count += 1

    @property
    def root(self):
        return self._item is None and self._count is None

    @property
    def leaf(self):
        return len(self._children) == 0

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a parent.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a parent from another tree.")
        self._parent = value

    @property
    def neighbor(self):
        return self._neighbor

    @neighbor.setter
    def neighbor(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a neighbor.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a neighbor from another tree.")
        self._neighbor = value

    @property
    def children(self):
        return tuple(self._children.itervalues())

    def inspect(self, depth=0):
        #print(('  ' * depth) + repr(self))
        for child in self.children:
            child.inspect(depth + 1)

    def __repr__(self):
        if self.root:
            return "<%s (root)>" % type(self).__name__
        return "<%s %r (%r)>" % (type(self).__name__, self.item, self.count)



# 数据集
dataset = [
    ['牛奶', '鸡蛋', '面包','薯片'],
    [ '鸡蛋','爆米花', '薯片','啤酒'],
    ['牛奶','啤酒','面包'],
    ['牛奶','鸡蛋','面包','爆米花','啤酒','薯片'],
    ['鸡蛋','面包','薯片'],
    ['鸡蛋','面包','啤酒', ],
    ['牛奶','面包', '薯片'],
    ['牛奶','鸡蛋','面包','黄油','薯片'],
    ['牛奶','鸡蛋','黄油','薯片']
]

#主函数
if __name__ == '__main__':

    '''
    调用find_frequent_itemsets()生成频繁项
    @:param minimum_support表示设置的最小支持度，即若支持度大于等于inimum_support，保存此频繁项，否则删除
    @:param include_support表示返回结果是否包含支持度，若include_support=True，返回结果中包含itemset和support，否则只返回itemset
    '''
    start = time.time()

    frequent_itemsets = find_frequent_itemsets(dataset, minimum_support=1, include_support=True)
    #print(type(frequent_itemsets))   # print type

    result = []
    for itemset, support in frequent_itemsets:    # 将generator结果存入list
        result.append((itemset, support))

    result = sorted(result, key=lambda i: i[0])   # 排序后输出
    for itemset, support in result:
        print(str(itemset) + ' ' + str(support))

    end = time.time()
    print('运行时间：', str(end - start))