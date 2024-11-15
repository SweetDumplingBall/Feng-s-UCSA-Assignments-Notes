import mesa
import random
import networkx as nx
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa import Agent, Model
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.datacollection import DataCollector



class Environment(mesa.Model):
    def __init__(self, num_humans, num_media, num_robots, k, p):
        super().__init__()
        self.grid = MultiGrid(20, 20, True)
        self.num_humans = num_humans
        self.num_media = num_media
        self.num_robots = num_robots
        self.k = k
        self.p = p
        # 初始化 DataCollector
        self.datacollector = DataCollector(
            agent_reporters={"Opinion": "opinion"},
            model_reporters={"Humans": "num_humans"}  # 记录人类用户数量
        )
        self.schedule = SimultaneousActivation(self)
        self.network = nx.DiGraph()

        # 创建用户
        for i in range(num_humans):
            human = HumanUser(i, self)
            self.schedule.add(human)
            self.network.add_node(i)
            self.grid.place_agent(human, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))


        # 创建媒体用户
        for j in range(num_media):
            media = MediaUser(num_humans + j, self)
            self.schedule.add(media)
            self.network.add_node(num_humans + j)
            self.grid.place_agent(media, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))


        # 创建机器人用户
        for k in range(num_robots):
            robot = RobotUser(num_humans + num_media + k, self)
            self.schedule.add(robot)
            self.network.add_node(num_humans + num_media + k)
            self.grid.place_agent(robot, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))


        self.initial_connect()

        self.G = self.network

    def initial_connect(self):
        # 初始连接
        for node in self.network.nodes():
            connections = random.sample(list(self.network.nodes()), self.k)
            for conn in connections:
                self.network.add_edge(node, conn)

    def step(self):
        self.schedule.step()
        self.update_network()

    def update_network(self):
        # 更新网络连边
        for node in self.network.nodes():
            if isinstance(self.schedule.agents[node], HumanUser):
                self.update_connections(node)

    def update_connections(self, node):
        # 根据观点相似度和动态变化率更新连接
        neighbors = list(self.network.successors(node))
        num_possible_connections = len(self.network.nodes()) - len(neighbors) - 1  # 减去自身

        # 只有当可用节点大于零时才进行抽样
        if num_possible_connections > 0:
            # 移除旧连接
            for neighbor in neighbors:
                if random.random() < self.p:
                    self.network.remove_edge(node, neighbor)

            # 随机抽样新连接
            new_connections = random.sample([n for n in self.network.nodes() if n not in neighbors and n != node],
                                            min(self.k, num_possible_connections))
            for new_conn in new_connections:
                self.network.add_edge(node, new_conn)

    def agent_portrayal(self, agent):
        """定义如何显示不同类型的代理"""
        portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}
        if isinstance(agent, HumanUser):
            portrayal["Color"] = "blue"
            portrayal["Layer"] = 1
        elif isinstance(agent, MediaUser):
            portrayal["Color"] = "green"
            portrayal["Layer"] = 2
        elif isinstance(agent, RobotUser):
            portrayal["Color"] = "red"
            portrayal["Layer"] = 3
        return portrayal



class HumanUser(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.opinion = random.uniform(-1, 1)
        self.activity_level = random.randint(1, 10)  # 活动水平

    def step(self):
        self.express_opinion()

    def express_opinion(self):
        # 表达观点或选择沉默
        if self.activity_level > random.uniform(0, 10):
            neighbors = list(self.model.network.successors(self.unique_id))
            if neighbors:
                for neighbor in neighbors:
                    neighbor_agent = self.model.schedule.agents[neighbor]
                    if neighbor_agent.opinion * self.opinion < 0:
                        # 反对意见吸引关注
                        self.opinion += 0.1
                    else:
                        # 支持意见
                        self.opinion -= 0.1
        self.opinion = max(min(self.opinion, 1), -1)  # 限制在[-1, 1]之间

class MediaUser(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.opinion = random.uniform(-1, 1)
        self.influence_factor = 1.5  # 媒体用户影响因子

    def step(self):
        # 媒体用户仅被关注，不主动影响其他用户
        for neighbor in self.model.network.successors(self.unique_id):
            neighbor_agent = self.model.schedule.agents[neighbor]
            # 根据邻居的观点决定是否吸引关注
            if neighbor_agent.opinion * self.opinion < 0 and random.random() < 0.9:  # 90%概率对反对意见吸引关注
                neighbor_agent.opinion -= 0.1 * (1 + neighbor_agent.opinion) * self.influence_factor
                neighbor_agent.opinion = max(min(neighbor_agent.opinion, 1), -1)
            elif neighbor_agent.opinion * self.opinion > 0 and random.random() < 0.7:  # 70%概率对支持意见吸引关注
                neighbor_agent.opinion += 0.1 * (1 - neighbor_agent.opinion) * self.influence_factor
                neighbor_agent.opinion = max(min(neighbor_agent.opinion, 1), -1)

class RobotUser(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.opinion = random.uniform(-1, 1)
        self.intervention_strength = random.uniform(0, 1)  # 干预强度
        self.activity_level = random.randint(1, 10)  # 机器人活动水平

    def step(self):
        self.intervene_opinion()

    def intervene_opinion(self):
        # 基于干预强度和活动水平的干预
        if self.activity_level > random.uniform(0, 10):
            for neighbor in self.model.network.successors(self.unique_id):
                neighbor_agent = self.model.schedule.agents[neighbor]
                # 根据干预强度对人类用户的观点进行干预
                neighbor_agent.opinion += self.intervention_strength * (self.opinion - neighbor_agent.opinion)
                neighbor_agent.opinion = max(min(neighbor_agent.opinion, 1), -1)  # 限制在[-1, 1]之间

# 创建模型实例
model = Environment(num_humans=10, num_media=5, num_robots=3, k=2, p=0.1)

# 运行模型
for step in range(100):  # 运行100个时间步
    model.step()
    print(f"Step {step}:")
    for agent in model.schedule.agents:
        connections = list(model.network.successors(agent.unique_id))
        print(f"Agent {agent.unique_id} has opinion {agent.opinion} and connections {connections}") # 打印每个代理的观点和连接


# 创建可视化的组件
grid = CanvasGrid(model.agent_portrayal, 20, 20, 500, 500)
chart = ChartModule([{"Label": "Opinion", "Color": "blue"}])
                     # ,{"Label": "Media", "Color": "green"},
                     # {"Label": "Robots", "Color": "red"}])

# 创建服务器
server = ModularServer(Environment,  # 使用类而不是实例
                       [grid, chart],
                       "Social Influence Model",
                       {"num_humans": 10, "num_media": 5, "num_robots": 3, "k": 2, "p": 0.1})

server.port = 8521  # 可以修改为其他端口
server.launch()


