from __future__ import print_function

class Decision_Tree(object):
    def __init__(self, val):
        self.val = val
        self.children = []
        self.children_dict = {}

    def assign(self, decision):
        if decision[0] not in self.children_dict:
            self.children_dict[decision[0]] = Decision_Tree(decision[1])
            self.children.append([decision[0], self.children_dict[decision[0]]])
        if len(decision) > 2:
            self.children_dict[decision[0]].assign(decision[2:])

    def dfs(self, path='If ', is_first=True):
        if not self.children:
            print(path + ', Golf=' + self.val)
        for key, value in self.children:
            if is_first:
                value.dfs(path + self.val + '=' + key, False)
            else:
                value.dfs(path + ' and ' + self.val + '=' + key, False)


if __name__ == '__main__':
    root = Decision_Tree('Outlook')
    root.assign(['overcast', 'yes'])
    root.assign(['rain', 'Wind', 'false', 'yes'])
    root.assign(['rain', 'Wind', 'true', 'no'])
    root.assign(['sunny', 'Humidity', 'high', 'no'])
    root.assign(['sunny', 'Humidity', 'low', 'yes'])
    root.dfs()