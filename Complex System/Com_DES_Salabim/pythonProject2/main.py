from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid


class Household(Agent):
    def __init__(self, my_id, my_model, dct_init):
        super().__init__(my_id, my_model)

        self.xv_tpl_init_position = dct_init["pos"]
        self.xv_int_type = dct_init["type"]
        self.xv_flt_similarity_threshold = dct_init["similarity_threshold"]

        self.nv_is_happy = None
        self.nv_int_totals_nearby = None
        self.nv_int_similar_nearby = None
        self.nv_tpl_position = self.xv_tpl_init_position

        pass
    def step(self) -> None:
        pass

class SegregationModel(Model):
    def __init__(self, dct_init):
        self.xv_int_grid_height = dct_init["grid_height"]
        self.xv_int_grid_width = dct_init["grid_width"]
        self.xv_flt_agent_density = dct_init["agent_density"]
        self.xv_flt_similarity_threshold = dct_init["similarity_threshold"]

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(self.xv_int_grid_height, self.xv_int_grid_width, torus=True)

        #创建household
        for i in range(int(self.xv_int_grid_height * self.xv_int_grid_width * self.xv_flt_agent_density)):
            dct_init = {
                "pos": self.random.choices,
                "type": 1 if self.random.random() < 0.5 else 2,
                "similarity_threshold": self.xv_flt_similarity_threshold,
            }
            household = Household(i, self, dct_init)
            self.schedule.add(household)
            self.grid.place_agent(household, household.xv_tpl_init_position)

    def step(self) -> None:
        pass

if __name__ == "__main__":
    model = SegregationModel()
    for i in range(200):
        model.step()


    pass