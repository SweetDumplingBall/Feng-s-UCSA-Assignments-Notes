from mesa.visualization.modules import NetworkModule
from mesa.visualization.ModularVisualization import ModularServer
from model import Environment

def portray_network(graph):
    portrayal = {}
    for node in graph.nodes():
        portrayal[node] = {"Shape": "circle", "Color": "blue", "Layer": 0, "Size": 5}
    for edge in graph.edges():
        portrayal[edge] = {"Weight": 1, "Color": "black"}
    return portrayal

def run_server():
    env_model = Environment(num_humans=10, num_media=5, num_robots=3, k=2, p=0.1)

    network_module = NetworkModule(portray_network, 400, 400)

    server = ModularServer(
        Environment,
        [network_module],
        "Network Visualization",
        {"num_humans": 10, "num_media": 5, "num_robots": 3, "k": 2, "p": 0.1}
    )

    server.port = 8521
    server.launch()

if __name__ == "__main__":
    run_server()
