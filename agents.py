# Importing necessary libraries
import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy
from RBBGovermentCommunication import GovernmentCommunication_PMT
import numpy as np
# Import functions from functions.py
from functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, floodplain_multipolygon


# Define the Households agent class
class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_adapted = False  # Initial adaptation status set to False
        self.adaptation_status = 'zero'

        # getting flood map values
        # Get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = False
        if contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y):
            self.in_floodplain = True

        # Self efficacy and threat vulnerability are both a number between 0 and 1 based on a normal distribution
        self.threat_vulnerability = -1
        while self.threat_vulnerability < 0 or self.threat_vulnerability > 1:
            self.threat_vulnerability = np.random.normal(loc=0.5, scale=0.14, size=None) 
        
        self.self_efficacy = -1
        while self.self_efficacy < 0 or self.self_efficacy > 1:
            
            self.self_efficacy = np.random.normal(loc=0.5, scale=0.14, size=None)
        
        if self.self_efficacy > 0.5: 
            self.self_efficacy_boolean = True
        else: 
            self.self_efficacy_boolean = False


        self.house_value = np.random.normal(300000, 50000)
        self.yearly_income = np.random.normal(53000, 20000)
        self.quarterly_income = self.yearly_income / 4
        self.savings = self.yearly_income * 0.5
        self.costs_for_insurance = 0
        
        # Get the estimated flood depth at those coordinates. 
        # the estimated flood depth is calculated based on the flood map (i.e., past data) so this is not the actual flood depth
        # Flood depth can be negative if the location is at a high elevation
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location, band=model.band_flood_img)
        # handle negative values of flood depth
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0
        
        # calculate the estimated flood damage given the estimated flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since there is not flood yet
        # and will update its value when there is a shock (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0
        
        #calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)
        
        # Set the initial insurance type
        if self.in_floodplain == True or self.flood_damage_estimated >= 0.5:
            self.insurance = 'state'
        else:
            self.insurance = 'zero'
        

    def check_limit(self,number, low, high):
        """Makes sure the number is between its limits"""
        if number > high: 
            number = high
        elif number < low:
            number = low
        return number


    # Function to count friends who can be influencial.
    def count_friends(self, radius):
        """Count the number of neighbors within a given radius (number of edges away). This is social relation and not spatial"""
        friends = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius)
        return len(friends)



    def get_friends(self, radius):
        """Get the number of neighbors within a given radius (number of edges away). This is social relation and not spatial"""
        friends = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius)
        return friends
    


    def get_friends_list(self):
        """Get a list of neighbors within a given radius (number of edges away) with the actual agent objects. This is social relation and not spatial"""
        friends_list = self.model.grid.get_neighbors(self.pos, include_center=False)
        return friends_list
    


    def pay_insurance(self):
        # Set the base case (maximum insurance) for insurance costs (so without any adaptation measures)
        if self.insurance == 'state':
            self.max_insurance = self.model.state_quarterly_max
        elif self.insurance == 'private':
            self.max_insurance = self.model.private_quarterly_max
        
        # Reduce the maximum insurance based on the adaptation status
        if self.insurance in ['state', 'private']:
            if self.adaptation_status == 'zero':
                self.final_insurance = self.max_insurance
            elif self.adaptation_status == 'low_effort':
                self.final_insurance = self.max_insurance*self.model.low_effort_insurance_reduction
            elif self.adaptation_status == 'high_effort':
                self.final_insurance = self.max_insurance*self.model.high_effort_insurance_reduction
            
            # Pay insurance
            self.savings -= self.final_insurance


    
    def choosing_insurance_type(self):
        # All households in floodplain have standard state insurance. Only those will look at upgrading towards private insurance
        if self.insurance == 'state':
            # Households will only upgrade insurance if they know that they are capable of doing that
            if self.self_efficacy_boolean == True:
                # Households will only upgrade insurance if flood awareness is high enough
                if self.threat_vulnerability > self.model.awareness_insurance_boundary:
                    # Households will only upgrade insurance if income is high enough
                    if self.quarterly_income * self.model.insurance_income_rate > self.model.private_quarterly_max:
                        self.insurance = 'private'



    def social_interaction_awareness(self):
        # Households have to have a minimal flood awareness to interact about this topic    
        if self.threat_vulnerability > self.model.awareness_interaction_boundary:
            # There is a random chance that households interact about floods
            if random.random() > self.model.social_interact_boundary:
                
                # Households only interact about floods with direct friends (radius 1)
                friends_demo = self.get_friends(1)
                friends = self.get_friends_list()

                #The mean of the awareness of friends is determined
                total_friends = 0
                for friend in friends:
                    total_friends += friend.threat_vulnerability
                mean_friends = total_friends/len(friends)

                # The difference between own awareness and the mean of friends' awareness is determined
                difference = abs(mean_friends - self.threat_vulnerability)

                # Own awareness is changed based on difference with friends' mean awareness
                if mean_friends < self.threat_vulnerability:
                    self.threat_vulnerability -= difference * self.model.interaction_effect
                
                else:
                    self.threat_vulnerability += difference * self.model.interaction_effect

                self.threat_vulnerability = self.check_limit(self.threat_vulnerability, 0, 1)
                


    def social_interaction_efficacy(self):
        # Self efficacy can only change if it is False
        if self.self_efficacy_boolean == False:

            # If friend has self-efficacy, and if random number is high enough, set self-efficacy to True
            friends = self.get_friends_list()
            for friend in friends:
                if friend.self_efficacy_boolean == True:
                    if random.random() > self.model.efficacy_boundary:
                        self.self_efficacy = self.self_efficacy * self.model.efficacy_friends_influence
                        if self.self_efficacy > 0.5:
                            self.self_efficacy_boolean = True
                            break
                        else:
                            self.self_efficacy_boolean = False
                    
    


    def shock(self):
        # Reduce the actual flood depth based on adaptation status
        if self.adaptation_status == 'low_effort':
            self.flood_depth_actual -= self.model.low_effort_damage_reduction
        elif self.adaptation_status == 'high_effort':
            self.flood_depth_actual -= self.model.high_effort_damage_reduction
        
        # Calculate actual damage and the house damage
        self.flood_damage_actual = calculate_basic_flood_damage(self.flood_depth_actual)
        self.house_damage = self.model.house_ground_rate * self.house_value * self.flood_damage_actual

        # Reduce savings by the monetary value of damage that was done to the house
        # (Note that if the insurance type is 'private', savings stay the same)
        if self.insurance == 'zero':
            self.costs_for_insurance = self.house_damage
            self.savings -= self.costs_for_insurance
        elif self.insurance == 'state':
            if self.house_damage > self.model.state_insurance_boundary:
                self.costs_for_insurance = self.house_damage - self.model.state_insurance_boundary
                self.savings -= self.costs_for_insurance
        elif self.insurance == 'private':
            self.costs_for_insurance = 0

        #When the damage is high enough, the treathvulnerability will be increased
        if self.house_damage/self.house_value > self.model.awareness_shock_boundary:
            self.threat_vulnerability += self.model.awareness_increase_shock



    def adaptation_with_measures(self):
        # Households will only adapt if they know that they can adapt (thus self-efficacy is True)
        if self.self_efficacy_boolean == True:

            # Households will only adapt if their flood awareness is high enough
            if self.threat_vulnerability > self.model.awareness_adaptation_boundary:

                # Households will only adapt if their estimated flood depth is high enough
                if self.flood_depth_estimated > self.model.low_effort_depth_boundary:

                    # Households will choose the high_effort measure if the estimated flood depth is high enough
                    if self.flood_depth_estimated > self.model.high_effort_depth_boundary:

                        # Households will only buy measure if they have not already got it
                        if self.adaptation_status != 'high_effort':

                            # Households will only buy measure if they have enough savings
                            if self.savings >= self.model.high_effort_cost_rate * self.house_value:
                                self.adaptation_status = 'high_effort'
                                self.savings -= self.model.high_effort_cost_rate * self.house_value
                            
                            # If household does not have enough savings for high effort, it will look if it can afford low effort measures
                            elif self.savings >= self.model.low_effort_cost_rate * self.house_value:
                                self.adaptation_status = 'low_effort'
                                self.savings -= self.model.low_effort_cost_rate * self.house_value

                    # Households will choose the low_effort measure if the estimated flood depth is not high enough for high effort
                    else:

                        # Households will only buy measure if they have not already got this measure or better
                        if self.adaptation_status == 'zero':

                            # Households will only buy measure if they have enough savings
                            if self.savings >= self.model.low_effort_cost_rate * self.house_value:
                                self.adaptation_status = 'low_effort'
                                self.savings -= self.model.low_effort_cost_rate * self.house_value



    def step(self):
        self.threat_vulnerability = self.check_limit(self.threat_vulnerability * self.model.step_awareness_reduction, 0, 1)
        self.self_efficacy = self.check_limit(self.self_efficacy * self.model.step_efficacy_reduction, 0, 1)
        self.pay_insurance()
        self.savings += self.quarterly_income * 0.2 # Each step, part of the income goes to savings
        if random.random() > self.model.spending_chance: #There is a chance that households spend x percent of their savings on holiday, furniture, etc.
            self.savings = self.savings * self.model.large_spending

        if self.model.flood_happening == True:
            self.shock()

        #check self_efficacy
        if self.self_efficacy > 0.5: 
            self.self_efficacy_boolean = True
        else: 
            self.self_efficacy_boolean = False
        
        self.social_interaction_awareness()
        self.social_interaction_efficacy()
        self.adaptation_with_measures()
        self.choosing_insurance_type()

        self.friends = self.get_friends_list()
        self.n_friends = self.get_friends(1)

        if self.adaptation_status != 'zero':
            self.is_adapted = True
               
        


# Define the Government agent class
class Government(Agent):
    """
    A government agent that currently doesn't perform any actions.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.top_down = self.model.top_down
        self.people_centered = self.model.people_centered
    def step(self):
        # The government agent doesn't perform any actions.
        percentile = float(np.percentile([agent.flood_depth_estimated for agent in self.model.schedule.agents],self.model.high_risk_hh_threshold))
        self.top_down_parameters ={"receive_p":0.8,"succes_p":0.2,"threat_vulnerability":self.model.flood_aware_increase_td,"threat_severity":0.2,"response_efficacy":0.2,"self_efficacy":self.model.self_efficacy_increase_td}
        self.people_centered_parameters ={"filter_variable_list":[["flood_depth_estimated",percentile,"Bigger"]],"receive_p":0.5,"succes_p":0.8,"threat_vulnerability":self.model.flood_aware_increase_pc,"threat_severity":0.4,"response_efficacy":0.2,"self_efficacy":self.model.self_efficacy_increase_pc}
        
        if self.top_down == True:
            GovernmentCommunication_PMT(agent_schedule=self.model.schedule,
                                        receive_p=self.top_down_parameters["receive_p"],
                                        succes_p=self.top_down_parameters["succes_p"],
                                        threat_vulnerability=self.top_down_parameters["threat_vulnerability"],
                                        threat_severity=self.top_down_parameters["threat_severity"],
                                        response_efficacy=self.top_down_parameters["response_efficacy"],
                                        self_efficacy=self.top_down_parameters["self_efficacy"]
                                        )
        if self.people_centered == True:
            GovernmentCommunication_PMT(agent_schedule=self.model.schedule,
                                        receive_p=self.people_centered_parameters["receive_p"],
                                        succes_p=self.people_centered_parameters["succes_p"],
                                        threat_vulnerability=self.people_centered_parameters["threat_vulnerability"],
                                        threat_severity=self.people_centered_parameters["threat_severity"],
                                        response_efficacy=self.people_centered_parameters["response_efficacy"],
                                        self_efficacy=self.people_centered_parameters["self_efficacy"],
                                        filter_variable_list =self.people_centered_parameters["filter_variable_list"]
                                        )

