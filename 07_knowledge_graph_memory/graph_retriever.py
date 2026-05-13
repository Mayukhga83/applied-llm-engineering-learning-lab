import networkx as nx


class MemoryGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_triple(self, subject: str, relation: str, obj: str):
        self.graph.add_node(subject)
        self.graph.add_node(obj)
        self.graph.add_edge(subject, obj, relation=relation)

    def search_entity(self, entity: str):
        if entity not in self.graph:
            return []

        results = []
        for _, target, data in self.graph.out_edges(entity, data=True):
            results.append((entity, data.get("relation"), target))
        return results
