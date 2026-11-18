from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Node:
    op: str
    inputs: List[str]
    attrs: Dict[str, Any]

def topo_order(graph: Dict[str, Node], outputs: List[str]) -> List[str]:
    seen, order = set(), []
    def dfs(v):
        if v in seen: return
        seen.add(v)
        for u in graph[v].inputs:
            if u in graph:
                dfs(u)
        order.append(v)
    for o in outputs:
        dfs(o)
    return order

def fuse_add_relu(graph: Dict[str, Node]) -> Dict[str, Node]:
    """
    Fuse pattern: relu(add(x, c)) -> add_relu(x, c)
    Very simplified to teach the idea.
    """
    newg = dict(graph)
    for name, node in list(graph.items()):
        if node.op == "relu":
            src = node.inputs[0]
            if src in graph and graph[src].op == "add":
                add_node = graph[src]
                fused = Node(op="add_relu", inputs=add_node.inputs, attrs={})
                newg[name] = fused
                # keep add node but it's now dead (real compilers DCE it)
    return newg

def main():
    # Graph for y = relu(add(matmul(a,b), c))
    g = {
        "t1": Node("matmul", ["a", "b"], {}),
        "t2": Node("add", ["t1", "c"], {}),
        "y":  Node("relu", ["t2"], {}),
    }

    print("Original graph:")
    for k in topo_order(g, ["y"]):
        print(k, g[k])

    g2 = fuse_add_relu(g)

    print("\nAfter fuse_add_relu:")
    for k in topo_order(g2, ["y"]):
        print(k, g2[k])

if __name__ == "__main__":
    main()
