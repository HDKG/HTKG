import torch
import heapq


class Node(object):
    def __init__(self, beta=None, depth=0):
        self.beta = beta
        self.childs = []
        self.depth = depth


def build_tree(depend2, depend, beta1, beta2, beta3, effective_list):
    root = Node()
    par = torch.argmax(depend2, 1).to('cpu')
    par2 = torch.argmax(depend, 1).to('cpu')
    Beta1 = beta1.to('cpu')
    Beta2 = beta2.to('cpu')
    Beta3 = beta3.to('cpu')

    for i, beta3 in enumerate(Beta3):
        childs = par == i
        level1 = Node(beta=beta3, depth=1)
        cnt = 0
        for j, flag in enumerate(childs):
            if flag == 1:
                level2 = Node(beta=Beta2[j], depth=2)
                childs2 = (par2 == j) * effective_list[0]
                if torch.sum(childs2) > 0:
                    for k, flag2 in enumerate(childs2):
                        if flag2 == 1:
                            level3 = Node(beta=Beta1[k], depth=3)
                            level2.childs.append(level3)
                    level1.childs.append(level2)
                    cnt += 1
        if cnt > 0:
            root.childs.append(level1)
    return root


def print_tree(Node, vocab):
    if Node.depth != 0:
        phi = Node.beta.tolist()
        words = map(phi.index, heapq.nlargest(10, phi))
        words10 = []
        s = '   ' * Node.depth + 'level ' + str(Node.depth)
        for w in words:
            words10.append(vocab[w])
            s += ' ' + vocab[w]
        print(s)
    for child in Node.childs:
        print_tree(child, vocab)


def get_topics(Node, beta_list):
    if Node.depth != 0:
        beta_list[Node.depth - 1].append(Node.beta.tolist())
    for child in Node.childs:
        get_topics(child, beta_list)
