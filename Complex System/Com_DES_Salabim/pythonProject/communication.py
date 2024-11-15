import salabim as sim
import networkx as nx
import random
import matplotlib.pyplot as plt

def draw_network_topology(G):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=random.randint(1,100))  # 使用 spring_layout 布局，确保节点间距较均匀
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.title("Small World Network Topology")
    plt.show()

# 创建小世界网络
n_nodes = 30
k_neighbors = 4
rewiring_prob = 0.3
G = nx.watts_strogatz_graph(n_nodes, k_neighbors, rewiring_prob)

# 绘制网络拓扑结构
draw_network_topology(G)


class NegativeNewsGenerator(sim.Component):
    def process(self):
        while True:
            NegativeNews()
            self.hold(sim.Uniform(10, 20).sample())  # 负面信息的生成间隔时间


class ClarificationNewsGenerator(sim.Component):
    def process(self):
        while True:
            ClarificationNews()
            self.hold(sim.Uniform(12, 24).sample())  # 澄清信息的生成间隔时间


class UserProcessor(sim.Component):
    def __init__(self, name, next_processors=None):
        super().__init__(name=name)
        self.next_processors = next_processors if next_processors is not None else []
        self.q_negative = sim.Queue(f"{name}_q_negative")
        self.q_clarification = sim.Queue(f"{name}_q_clarification")

    def process(self):
        while True:
            # 如果两个队列均为空，则等待消息
            while len(self.q_negative) == 0 and len(self.q_clarification) == 0:
                self.passivate()
            # 优先处理负面信息
            if len(self.q_negative) > 0:
                message = self.q_negative.pop()
            else:
                message = self.q_clarification.pop()
            # 模拟处理时间
            self.hold(sim.Uniform(4, 8).sample())
            # 信息处理后，决定是否转发
            self.forward_message(message)

    def forward_message(self, message):
        # 随机选择一个或多个下一级用户进行消息转发
        for processor in random.sample(self.next_processors, k=random.randint(1, len(self.next_processors))):
            if isinstance(message, NegativeNews):
                if message not in processor.q_negative:
                    message.enter(processor.q_negative)
            elif isinstance(message, ClarificationNews):
                if message not in processor.q_clarification:
                    message.enter(processor.q_clarification)
            if processor.ispassive():
                processor.activate()


class NegativeNews(sim.Component):
    def process(self):
        self.enter(user_processors[0].q_negative)  # 将负面消息放入网络中一个节点的负面队列
        if user_processors[0].ispassive():
            user_processors[0].activate()


class ClarificationNews(sim.Component):
    def process(self):
        self.enter(user_processors[0].q_clarification)  # 将澄清消息放入网络中一个节点的澄清队列
        if user_processors[0].ispassive():
            user_processors[0].activate()


# 创建小世界网络
def create_small_world_network(n, k, p):
    G = nx.watts_strogatz_graph(n, k, p)
    return G

def do_animation():
    # 将整体图表上移并增大图表之间的间距
    for i, user in enumerate(user_processors):
        sim.AnimateMonitor(user.q_negative.length, x=10, y=600 - i * 55, vertical_scale=10, width=200, height=30,
                           title=f'{user.name()} Negative News Queue Length')
        sim.AnimateMonitor(user.q_clarification.length, x=250, y=600 - i * 55, vertical_scale=10, width=200, height=30,
                           title=f'{user.name()} Clarification News Queue Length')

if __name__ == '__main__':
    env = sim.Environment()

    # 定义网络节点数、邻居数和重连概率
    n_nodes = 30
    k_neighbors = 4
    rewiring_prob = 0.3

    # 创建小世界网络
    G = create_small_world_network(n_nodes, k_neighbors, rewiring_prob)

    # 创建节点和其相应的UserProcessor
    user_processors = [UserProcessor(f"User {i}") for i in range(n_nodes)]

    # 根据小世界网络结构定义每个UserProcessor的下一级用户（邻居）
    for i, user in enumerate(user_processors):
        neighbors = list(G.neighbors(i))
        user.next_processors = [user_processors[neighbor] for neighbor in neighbors]

    # 添加负面和澄清消息生成器
    NegativeNewsGenerator()
    ClarificationNewsGenerator()

    # 启动动画
    do_animation()
    env.animate(True)
    env.speed(50)
    env.run(till=300)

    # 输出统计信息
    print("Simulation completed.")
    for user in user_processors:
        user.q_negative.print_info()
        user.q_clarification.print_info()
