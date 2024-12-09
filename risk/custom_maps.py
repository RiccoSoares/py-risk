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
    
    return MapStructure(mapid=4, name="Simple Map", graph=g, bonuses=bonuses, ids=ids)

def create_banana_map():
    """
        Banana Map with 12 territories and 4 bonuses. Link to the map: https://www.warzone.com/SinglePlayer?PreviewMap=29633
        Regions are:
            0 - First Bite
            1 - Second Bite
            2 - Right up
            3 - Right middle
            4 - Right down
            5 - Left up
            6 - Left middle
            7 - Left down
            8 - Third bite
            9- The brown part
            10- Holding place
            11- The End
    """

    num_vertices = 12
    g = Graph()
    g.add_vertices(num_vertices)

    connections = [
        (0,1),# First Bite connections
        (1,2),(1,8),(1,9), #Second Bite connections
        (2,3),(2,9),# Right up connections
        (3,4),# Right middle connections
        (4,11),# Right down connections
        (5,6),(5,8),# Left up connections
        (6,7),(6,8),# Left middle connections
                        # Left down connections
        (8,9),(8,10),# Third bite connections
        (9,10),(9,11),# The brown part connections
        (10,11)# Holding place connections
                # The End connections
    ]

    g.add_edges(connections)
    ids = {i: i for i in range(num_vertices)}

    bonuses = [ Bonus("Banana", {0, 1, 8}, 2),
                Bonus("Bottom Scale", {9, 10, 11}, 2),
                Bonus("Left Scale", {5, 6, 7}, 2),
                Bonus("Right Scale", {2, 3, 4}, 2)]
    return MapStructure(mapid=3, name="Banana", graph=g, bonuses=bonuses, ids=ids)

def create_owl_island_map():
    """
        Owl Map with 12 territories and 4 bonuses. Link to the map: https://www.warzone.com/SinglePlayer?PreviewMap=56763
        Regions are:
            0 - Eastpoint
            1 - Blue River
            2 - Eastern Jungle
            3 - Owl Mountain
            4 - Saltwater Falls
            5 - Eastpath
            6 - Westpath
            7 - Brine Cliffs
            8 - Forgotten Path
            9 - Northern Dunes
            10- Western Jungle
            11- Westpoint
    """

    num_vertices = 12
    g = Graph()
    g.add_vertices(num_vertices)

    connections = [
        (0,1), (0,2), # Eastpoint connections
        (1,2), (1,5), # Blue River connections
        (2,4), (2,5), # Eastern Jungle connections
        (3,5), (3,6), # Owl Mountain connections
        (4,5), # Saltwater Falls connections
        (5,6),(5,7), # Eastpath connections
        (6,7),(6,8),(6,10), # Westpath connections
        (7,10), # Brine Cliffs connections
        (8,9),(8,10), # Forgotten Path connections
        (9,10),(9,11), # Northern Dunes part connections
        (10,11) # Western Jungle connections
                # Westpoint connections
    ]

    g.add_edges(connections)
    ids = {i: i for i in range(num_vertices)}

    bonuses = [ Bonus("East", {0, 1, 2}, 2),
                Bonus("North", {3}, 1),
                Bonus("South", {4, 5, 6, 7}, 3),
                Bonus("West", {8, 9, 10, 11}, 3)]
    return MapStructure(mapid=2, name="Owl Island", graph=g, bonuses=bonuses, ids=ids)

def create_italy_map():
    """
        Italy Map with 20 territories and 10 bonuses. Link to the map: https://www.warzone.com/SinglePlayer?PreviewMap=3448
        Regions are:
            0 - Trentino-Alto Adige
            1 - Veneto
            2 - Friuli-Venezia Giulia
            3 - Lombardia
            4 - Emilia-Romagna
            5 - Toscana
            6 - Valle d'Aosta
            7 - Piemonte
            8 - Liguria
            9 - Marche
            10 - Umbria
            11 - Lazio
            12 - Abruzzo
            13 - Molise
            14 - Campania
            15 - Puglia
            16 - Basilicata
            17 - Calabria
            18 - Sicilia
            19 - Sardegna
    """
    num_vertices = 20
    g = Graph()
    g.add_vertices(num_vertices)

    connections = [
        (0,1),(0,3),   # Trentino-Alto Adige connections
        (1,2),(1,3),(1,4),    # Veneto connections
                        # Friuli-Venezia Giulia connections
        (3,4),(3,7),(3,8),  # Lombardia connections
        (4,5),(4,8),(4,7),(4,9),         # Emilia-Romagna connections
        (5,8),(5,9),(5,10),(5,11),        # Toscana connections
        (6,7),         # Valle d'Aosta connections
        (7,8),         # Piemonte connections
        (8,19),         # Liguria connections
        (9,10),(9,11),(9,12),        # Marche connections
        (10,11),       # Umbria connections
        (11,12),(11,13),(11,14),(11,19),       # Lazio connections
        (12,13),       # Abruzzo connections
        (13,14),(13,15),       # Molise connections
        (14,15),(14,16),(14,18),       # Campania connections
        (15,16),       # Puglia connections
        (16,17),       # Basilicata connections
        (17,18),       # Calabria connections
        (18,19)        # Sicilia connections
                        # Sardegna connections
    ]

    g.add_edges(connections)
    ids = {i: i for i in range(num_vertices)}

    bonuses = [
        Bonus("East Italy", {0, 1, 2}, 3),
        Bonus("Islands", {18, 19}, 2),
        Bonus("Mafia", {13, 14}, 2),
        Bonus("Middle", {5, 9, 10, 11, 12, 19}, 7),
        Bonus("Middle Italy", {9, 10, 11, 12}, 5),
        Bonus("North", {0, 1, 2, 3, 4, 6, 7, 8}, 9),
        Bonus("Padania", {3, 4, 5}, 4),
        Bonus("South", {13, 14, 15, 16, 17, 18}, 7),
        Bonus("Terronia", {15, 16, 17}, 3),
        Bonus("West Italy", {6, 7, 8}, 3)
    ]

    return MapStructure(mapid=1, name="Italy", graph=g, bonuses=bonuses, ids=ids)