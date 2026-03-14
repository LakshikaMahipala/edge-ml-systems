from __future__ import annotations
from dag import DAG, Task, Edge

def make_simple_pipeline():
    devices = ["cpu","gpu","fpga"]
    tasks = {
        "pre": Task("pre", {"cpu":1.0,"gpu":2.0,"fpga":3.0}),
        "infer": Task("infer", {"cpu":15.0,"gpu":2.5,"fpga":4.0}),
        "post": Task("post", {"cpu":1.2,"gpu":2.2,"fpga":3.2}),
    }
    edges = [Edge("pre","infer",262144), Edge("infer","post",1024)]
    return DAG(devices, tasks, edges)
