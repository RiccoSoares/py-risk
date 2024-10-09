from igraph import Graph
from .game_types import MapStructure, Bonus

def create_simple_map():
    g = Graph()
    g.add_vertices(9)
    
    connections = [
        (0, 1), (1, 2),    # Connections within Region 1
        (3, 4), (4, 5),    # Connections within Region 2
        (6, 7), (7, 8),    # Connections within Region 3
        (2, 3),            # Connection between Region 1 and Region 2
        (5, 6)             # Connection between Region 2 and Region 3
    ]
    g.add_edges(connections)
    
    ids = {i: i for i in range(9)}
    
    bonuses = [
        Bonus("Region 1", {0, 1, 2}, 2),
        Bonus("Region 2", {3, 4, 5}, 2),
        Bonus("Region 3", {6, 7, 8}, 2)
    ]
    
    return MapStructure(mapid=1, name="Simple Map", graph=g, bonuses=bonuses, ids=ids)