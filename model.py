# Importing necessary libraries
import networkx as nx
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import numpy as np
import geopandas as gpd
import rasterio as rs
import matplotlib.pyplot as plt
import random
import os


# Import the agent class(es) from agents.py
from agents import Households, Government

# Import functions from functions.py
from functions import get_flood_map_data, calculate_basic_flood_damage
from functions import map_domain_gdf, floodplain_gdf


# Define the AdaptationModel class
class AdaptationModel(Model):
    """
    The main model running the simulation. It sets up the network of household agents,
    simulates their behavior, and collects data. The network type can be adjusted based on study requirements.
    """

    def __init__(self, 
                people_centered = False,
                top_down = False,
                seed = None,
                number_of_households = 250, # number of household agents
                # Simplified argument for choosing flood map. Can currently be "harvey", "100yr", or "500yr".
                flood_map_choice='harvey',
                # ### network related parameters ###
                # The social network structure that is used.
                # Can currently be "erdos_renyi", "barabasi_albert", "watts_strogatz", or "no_network"
                network = 'watts_strogatz',
                # likeliness of edge being created between two nodes
                probability_of_network_connection = 0.4,
                # number of edges for BA network
                number_of_edges = 3,
                # number of nearest neighbours for WS social network
                number_of_nearest_neighbours = 5,
                # maximal insurance costs per quarter
                state_quarterly_max = 100,
                private_quarterly_max = 300,
                low_effort_insurance_reduction = 0.8,
                high_effort_insurance_reduction = 0.5,
                social_interact_boundary = 0.5,
                awareness_interaction_boundary = 0.25,
                interaction_effect = 0.5, 
                efficacy_boundary = 0.8,
                efficacy_friends_influence = 1.01,
                flood_step = 50,
                low_effort_damage_reduction = 1, 
                high_effort_damage_reduction = 2, 
                house_ground_rate = 0.8,
                state_insurance_boundary = 250000,
                low_effort_depth_boundary = 0.1,
                high_effort_depth_boundary = 0.25, 
                awareness_adaptation_boundary = 0.3,
                high_effort_cost_rate = 0.04, 
                low_effort_cost_rate = 0.005, 
                awareness_insurance_boundary = 0.3,
                insurance_income_rate = 0.2,
                step_awareness_reduction = 0.99,
                step_efficacy_reduction = 0.995, 
                awareness_increase_shock = 0.25,
                awareness_shock_boundary = 0.1,
                spending_chance = 0.5,
                large_spending = 0.85,
                flood_aware_increase_td =0.1,
                self_efficacy_increase_td=0.1,
                high_risk_hh_threshold = 50,
                flood_aware_increase_pc =0.2,
                self_efficacy_increase_pc=0.2
                ):
        
        super().__init__(seed = seed)
        
                
        # defining the variables and setting the values
        self.number_of_households = number_of_households  # Total number of household agents
        self.seed = seed
        #np.random.seed(40)

        #Communication
        self.top_down = top_down
        self.people_centered = people_centered

        #Government 
        self.government = Government(unique_id=0, model=self)

        # network
        self.network = network # Type of network to be created
        self.probability_of_network_connection = probability_of_network_connection
        self.number_of_edges = number_of_edges
        self.number_of_nearest_neighbours = number_of_nearest_neighbours


        # insurance variables
        self.state_quarterly_max = state_quarterly_max
        self.private_quarterly_max = private_quarterly_max
        self.low_effort_insurance_reduction = low_effort_insurance_reduction
        self.high_effort_insurance_reduction = high_effort_insurance_reduction
        self.state_insurance_boundary = state_insurance_boundary
        self.insurance_income_rate = insurance_income_rate

        # social interaction variables
        self.social_interact_boundary = social_interact_boundary
        self.awareness_interaction_boundary = awareness_interaction_boundary
        self.interaction_effect = interaction_effect
        self.efficacy_boundary = efficacy_boundary
        self.efficacy_friends_influence = efficacy_friends_influence

        # Flood variables
        self.flood_step = flood_step
        self.low_effort_damage_reduction = low_effort_damage_reduction
        self.high_effort_damage_reduction = high_effort_damage_reduction
        self.house_ground_rate = house_ground_rate
        self.awareness_increase_shock = awareness_increase_shock
        self.awareness_shock_boundary = awareness_shock_boundary

        # Adaptation variables
        self.low_effort_depth_boundary = low_effort_depth_boundary
        self.high_effort_depth_boundary = high_effort_depth_boundary
        self.low_effort_cost_rate = low_effort_cost_rate
        self.high_effort_cost_rate = high_effort_cost_rate
        self.awareness_adaptation_boundary = awareness_adaptation_boundary
        self.awareness_insurance_boundary = awareness_insurance_boundary

        self.step_awareness_reduction = step_awareness_reduction
        self.step_efficacy_reduction = step_efficacy_reduction

        # Spending variables
        self.spending_chance = spending_chance
        self.large_spending = large_spending

        #Influence of communication tools
        self.flood_aware_increase_td=flood_aware_increase_td
        self.self_efficacy_increase_td=self_efficacy_increase_td
        self.high_risk_hh_threshold =high_risk_hh_threshold
        self.flood_aware_increase_pc = flood_aware_increase_pc
        self.self_efficacy_increase_pc= self_efficacy_increase_pc


        # At the beginning there is no flood
        self.flood_happening = False


        # generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network()
        # create grid out of network graph
        self.grid = NetworkGrid(self.G)

        # Initialize maps
        self.initialize_maps(flood_map_choice)

        # set schedule for agents
        self.schedule = RandomActivation(self)  # Schedule for activating agents

        # create households through initiating a household on each node of the network graph
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self)
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)

        # You might want to create other agents here, e.g. insurance agents.

        # Data collection setup to collect data
        model_metrics = {
                        "total_adapted_households": self.total_adapted_households,
                        # ... other reporters ...
                        }
        
        agent_metrics = {
                        "FloodDepthEstimated": "flood_depth_estimated",
                        "FloodDamageEstimated" : "flood_damage_estimated",
                        "FloodDepthActual": "flood_depth_actual",
                        "FloodDamageActual" : "flood_damage_actual",
                        "IsAdapted": "is_adapted",
                        "FriendsCount": lambda a: a.count_friends(radius=1),
                        "location":"location",
                        "AdaptationStatus": "adaptation_status",
                        "InsuranceType": "insurance",
                        "FloodAwareness": "threat_vulnerability",
                        "self_efficacy":"self_efficacy",
                        'house_damage':'house_damage',
                        'adaptation_low_effort':lambda a: a.adaptation_status == 'low_effort',
                        'adaptation_high_effort': lambda a: a.adaptation_status == 'high_effort',
                        'efficacy_binair_true':lambda a: a.self_efficacy_boolean == True,
                        'private_insurance':lambda a: a.insurance == 'private',
                        'final_insurance':'final_insurance',
                        "costs_for_insurance":"costs_for_insurance"
                        # ... other reporters ...
                        }
        #set up the data collector 
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)
            


    def initialize_network(self):
        """
        Initialize and return the social network graph based on the provided network type using pattern matching.
        """
        if self.network == 'erdos_renyi':
            return nx.erdos_renyi_graph(n=self.number_of_households,
                                        p=self.number_of_nearest_neighbours / self.number_of_households,
                                        seed=self.seed)
        elif self.network == 'barabasi_albert':
            return nx.barabasi_albert_graph(n=self.number_of_households,
                                            m=self.number_of_edges,
                                            seed=self.seed)
        elif self.network == 'watts_strogatz':
            return nx.watts_strogatz_graph(n=self.number_of_households,
                                        k=self.number_of_nearest_neighbours,
                                        p=self.probability_of_network_connection,
                                        seed=self.seed)
        elif self.network == 'no_network':
            G = nx.Graph()
            G.add_nodes_from(range(self.number_of_households))
            return G
        else:
            raise ValueError(f"Unknown network type: '{self.network}'. "
                            f"Currently implemented network types are: "
                            f"'erdos_renyi', 'barabasi_albert', 'watts_strogatz', and 'no_network'")


    def initialize_maps(self, flood_map_choice):
        """
        Initialize and set up the flood map related data based on the provided flood map choice.
        """
        # Define paths to flood maps
        flood_map_paths = {
            'harvey': r"C:\Users\Ineke\FloodModel.3-2-2024\input_data\floodmaps\Harvey_depth_meters.tif",
            '100yr': r"C:\Users\Ineke\FloodModel.3-2-2024\input_data\floodmaps\Harvey_depth_meters.tif",
            '500yr': r"C:\Users\Ineke\FloodModel.3-2-2024\input_data\floodmaps\Harvey_depth_meters.tif"  # Example path for 500yr flood map
        }

        # Throw a ValueError if the flood map choice is not in the dictionary
        if flood_map_choice not in flood_map_paths.keys():
            raise ValueError(f"Unknown flood map choice: '{flood_map_choice}'. "
                            f"Currently implemented choices are: {list(flood_map_paths.keys())}")

        # Choose the appropriate flood map based on the input choice
        flood_map_path = flood_map_paths[flood_map_choice]

        # Loading and setting up the flood map
        self.flood_map = rs.open(flood_map_path)
        self.band_flood_img, self.bound_left, self.bound_right, self.bound_top, self.bound_bottom = get_flood_map_data(
            self.flood_map)

    def total_adapted_households(self):
        """Return the total number of households that have adapted."""
        #BE CAREFUL THAT YOU MAY HAVE DIFFERENT AGENT TYPES SO YOU NEED TO FIRST CHECK IF THE AGENT IS ACTUALLY A HOUSEHOLD AGENT USING "ISINSTANCE"
        adapted_count = sum([1 for agent in self.schedule.agents if isinstance(agent, Households) and agent.is_adapted])
        return adapted_count
    
    def plot_model_domain_with_agents(self):
        fig, ax = plt.subplots()
        # Plot the model domain
        map_domain_gdf.plot(ax=ax, color='lightgrey')
        # Plot the floodplain
        floodplain_gdf.plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)

        # Collect agent locations and statuses
        for agent in self.schedule.agents:
            color = 'blue' if agent.is_adapted else 'red'
            ax.scatter(agent.location.x, agent.location.y, color=color, s=100, label=color.capitalize() if not ax.collections else "")
            ax.annotate(str(agent.unique_id), (agent.location.x, agent.location.y), textcoords="offset points", xytext=(0,1), ha='center', fontsize=15)
        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Red: not adapted, Blue: adapted")

        # Customize plot with titles and labels
        plt.title(f'Model Domain with Agents at Step {self.schedule.steps}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def ConvertEfficacy(self):
        for agent in self.schedule.agents:
            if agent.self_efficacy > 0.5: 
                agent.self_efficacy_boolean = True
            else: 
                agent.self_efficacy_boolean = False
            

    def step(self):
        """
        introducing a shock: 
        at time step 5, there will be a global flooding.
        This will result in actual flood depth. Here, we assume it is a random number
        between 0.5 and 1.2 of the estimated flood depth. In your model, you can replace this
        with a more sound procedure (e.g., you can devide the floop map into zones and 
        assume local flooding instead of global flooding). The actual flood depth can be 
        estimated differently
        """
        
        if self.schedule.steps % 4 == 0:
            self.government.step()
            self.ConvertEfficacy()
            
        if self.schedule.steps == self.flood_step:
            self.flood_happening = True
            for agent in self.schedule.agents:
                # Calculate the actual flood depth as a random number between 0.5 and 1.2 times the estimated flood depth
                agent.flood_depth_actual = random.uniform(0.5, 1.2) * agent.flood_depth_estimated
                # calculate the actual flood damage given the actual flood depth
                agent.flood_damage_actual = calculate_basic_flood_damage(agent.flood_depth_actual)
        
        if self.schedule.steps == self.flood_step + 1:
            self.flood_happening = False

        # Collect data and advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()
