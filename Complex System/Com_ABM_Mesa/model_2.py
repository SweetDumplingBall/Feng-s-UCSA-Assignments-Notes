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
            model_reporters={
                "Humans": lambda m: sum([1 for a in m.schedule.agents if isinstance(a, HumanUser)]),
                "Media": lambda m: sum([1 for a in m.schedule.agents if isinstance(a, MediaUser)]),
                "Robots": lambda m: sum([1 for a in m.schedule.agents if isinstance(a, RobotUser)])
            }
        )

        self.schedule = SimultaneousActivation(self)
        self.network = nx.DiGraph()
        self.agents_to_remove = []  # 新增：存储需要删除的代理

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

        # self.G = self.network

    def initial_connect(self):
        # 初始连接
        for node in self.network.nodes():
            connections = random.sample(list(self.network.nodes()), self.k)
            for conn in connections:
                self.network.add_edge(node, conn)

    def step(self):
        self.schedule.step()
        self.update_humans()  # 更新人类用户的数量
        self.update_network()

        # 删除标记的代理
        for agent in self.agents_to_remove:
            self.schedule.remove(agent)  # 从调度器中移除
            self.network.remove_node(agent.unique_id)  # 从网络中移除
            self.grid.remove_agent(agent)  # 从网格中移除
            self.num_humans -= 1  # 更新人类用户数量

        self.agents_to_remove.clear()  # 清空删除列表

        self.datacollector.collect(self)  # 收集当前时间步的数据

    def update_humans(self):
        """模拟人类用户的进入和退出"""
        for agent in list(self.schedule.agents):
            if isinstance(agent, HumanUser):
                # 如果人类用户的沉默时间太久或者随机的，则调用 HumanUser 类的 exit 方法，将这个用户剔除
                if agent.silent_time > agent.silent_threshold or random.random() < 0.005:
                    agent.exit()  # 调用退出方法，移除用户
        # 人类用户的加入（加入概率为 0.8）
        if random.random() < 0.8:  # 有 80% 的概率加入一个新的人类用户
            new_id = self.num_humans
            new_human = HumanUser(new_id, self)
            self.schedule.add(new_human)
            self.network.add_node(new_id)
            self.grid.place_agent(new_human,
                                  (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))
            self.num_humans += 1
            print(f"Human {new_id} entered.")

    def update_network(self):
        # 更新网络连边
        for node in self.network.nodes():
            if node in self.schedule.agents and isinstance(self.schedule.agents[node], HumanUser):
                self.update_connections(node)

    def update_connections(self, node):
        # 根据观点相似度和动态变化率更新连接
        if self.unique_id in self.model.network:
            neighbors = list(self.model.network.successors(self.unique_id))
        else:
            neighbors = []  # 如果代理已经从网络中移除，则设置为空
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
        self.express_threshold = random.uniform(0, 10)  # 每个用户有不同的表达阈值，范围从0到10
        self.silent_time = 0  # 沉默计时器，记录用户多久没有表达观点
        self.silent_threshold = 3  # 沉默阈值，超过此值将退出

    def step(self):
        self.express_opinion()

    def express_opinion(self):
        """用户表达观点或选择沉默"""
        if self.activity_level >= self.express_threshold:  # 如果活动水平>表达阈值，则表达，否则沉默
            neighbors = list(self.model.network.successors(self.unique_id))
            # 移除已经退出的代理
            neighbors = [neighbor for neighbor in neighbors if neighbor in self.model.schedule.agents]

            if neighbors:
                for neighbor in neighbors:
                    # 检查邻居是否还在调度器中
                    if neighbor in self.model.schedule.agents:
                        neighbor_agent = self.model.schedule.agents[neighbor]
                        if neighbor_agent.opinion * self.opinion < 0:
                            # 反对意见吸引关注
                            self.opinion += 0.1
                        else:
                            # 支持意见
                            self.opinion -= 0.1
                # 发表过观点后，重置沉默计时器
                self.silent_time = 0
                self.activity_level += 1  # 更新这个主体的活动水平
            else:
                # 如果没有表达观点，增加沉默时间，以便区分是否会退出
                self.activity_level -= 1
                self.silent_time += 1
        else:
            # 如果没有表达观点，增加沉默时间，以便区分是否会退出
            self.activity_level -= 1
            self.silent_time += 1

        self.opinion = max(min(self.opinion, 1), -1)  # 限制在[-1, 1]之间

        # 判断是否需要退出
        if self.silent_time >= self.silent_threshold:
            self.exit()

    def exit(self):
        """当用户活动水平过低时退出，改为标记删除"""
        print(f"Human {self.unique_id} exiting due to low activity level.")

        # 确保从网络中移除所有与该代理相关的连接
        neighbors = list(self.model.network.successors(self.unique_id))
        for neighbor in neighbors:
            self.model.network.remove_edge(self.unique_id, neighbor)
        neighbors = list(self.model.network.predecessors(self.unique_id))
        for neighbor in neighbors:
            self.model.network.remove_edge(neighbor, self.unique_id)

        # 将代理添加到删除队列
        self.model.agents_to_remove.append(self)

        # 从调度器中移除代理
        self.model.schedule.remove(self)

        # 从网格中移除代理
        self.model.grid.remove_agent(self)

        # 从网络中移除代理节点
        self.model.network.remove_node(self.unique_id)


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
        print(f"Agent {agent.unique_id}, Type: {type(agent).__name__}, Opinion: {agent.opinion:.2f}, Connections: {connections}")

# 创建可视化的组件
grid = CanvasGrid(model.agent_portrayal, 20, 20, 500, 500)
chart = ChartModule([
    # {"Label": "Opinion", "Color": "blue"},
    {"Label": "Humans", "Color": "yellow"},
    {"Label": "Media", "Color": "green"},
    {"Label": "Robots", "Color": "red"}
])


# 创建服务器
server = ModularServer(Environment,  # 使用类而不是实例
                       [grid, chart],
                       "Social Influence Model",
                       {"num_humans": 10, "num_media": 5, "num_robots": 3, "k": 2, "p": 0.1})

server.port = 8521
server.launch()


