import random
import warnings
import numpy as np

def GovernmentCommunication_PMT(agent_schedule,
                                filter_variable_list=[],
                                receive_p=1.0,
                                succes_p=1.0,
                                threat_vulnerability=0.0,
                                threat_severity=0.0,
                                response_efficacy=0.0,
                                self_efficacy=0.0):
    """
    Parameter for Agent Input:
        agent_schedule: Schedule containing the agents of the model/

    Parameters for Protection Motivation Behaviour:
        threat_vulnerability:    This factor refers to an individual's perception of the probability of a particular threat.
        threat_severity:    This factor involves the perceived seriousness of the potential threat.
        response_efficacy:    Response efficacy refers to the perception of the effectiveness of a response to protect against a threat.
        self_efficacy:    Self-efficacy is the individual's belief in their ability to successfully perform a response.

    Parameters for Government Communication:
        receive_p:    The probability that the communication method reaches the individual
        succes_p:    The probability that the communication succeeds in changing the perception of the environment.

    Parameter for Person Centered communication:
        filter_variable_list:    The variables that will be used to determine which agents are extra vulnerable to the threat and which are not. Based on the threshold value.
    
    
    """
    


    # Function to increase attribute values for the agent
    def increase_values():
        # Loop through attributes and their corresponding values
        for index, attribute_value in enumerate([threat_vulnerability, threat_severity, response_efficacy, self_efficacy]):
            attribute_names = ["threat_vulnerability", "threat_severity", "response_efficacy", "self_efficacy"]
            attribute_name = attribute_names[index]
            
            # Check if the agent has the specified attribute
            if hasattr(agent, attribute_name):
                # Increase the attribute value, but ensure it doesn't exceed 1
                increase_value = attribute_value + getattr(agent, attribute_name)
                if increase_value > 1:
                    setattr(agent, attribute_name, 1)
                else:
                    setattr(agent, attribute_name, increase_value)
            else:
                # Warn if the agent doesn't have the specified attribute
                warnings.warn(f"Agent has no attribute '{attribute_name}'. So it will be ignored")

    # Function to simulate contact with the government
    def contact():
        # Check if the agent receives the communication
        if random.uniform(0, 1) <= receive_p:
            # Check if the communication is successful
            if random.uniform(0, 1) <= succes_p:
                # If successful, increase attribute values
                increase_values()

    # Function to check the validity of input parameters
    def check_input():
        # Loop through each filter_list in filter_variable_list
        for filter_list in filter_variable_list:
            # Check if filter_list is a valid list
            if type(filter_list) != list:
                raise ValueError(f"Expected a list not {type(filter_list)}")
            
            # Check if filter_list has three elements
            elif len(filter_list) != 3:
                raise ValueError("filter_variable_list must contain a list formatted like [filter_variable, Threshold, filter_operator]")
            
            # Check the types of elements in filter_list
            for index, value_type in enumerate([str, float, str]):
                if type(filter_list[index]) != value_type:
                    raise ValueError(f"On {index} of list in filter_variable_list is {type(filter_list[index])}, but {value_type} was expected.")
            
            # Check the types and range of numerical parameters
            for parameter in [receive_p, succes_p, threat_vulnerability, threat_severity, response_efficacy, self_efficacy]:
                if type(parameter) != float:
                    raise ValueError(f"{parameter} must be float instead of {type(parameter)}.")
                elif parameter < 0 or parameter > 1.0:
                    raise ValueError(f"{parameter} must be between 0 and 1")        

    #np.random.seed(40)
    # Check input parameters validity
    check_input()
    # Initialisation verification

    # Loop through agents in the agent_schedule
    for agent in agent_schedule.agents:
        # print(f'Household {agent.unique_id}\
        #     \n\tFlood depth: {agent.flood_depth_estimated}\
        #     \n\tOld flood awareness: {agent.threat_vulnerability}\
        #     \n\tOld self_efficacy: {agent.self_efficacy}')

        check = True
        # Check against filter_variable_list criteria
        for filter_list in filter_variable_list:
            filter_variable = filter_list[0]
            filter_threshold = filter_list[1]
            filter_operator = filter_list[2]
            # Compare agent's attribute with specified threshold based on the operator
            
            if filter_operator == "Smaller":
                if getattr(agent, filter_variable) > filter_threshold:
                    check = False
                    break
            elif filter_operator == "Bigger":
                if getattr(agent, filter_variable) < filter_threshold:
                    check = False
                    break
            else:
                raise ValueError("filter_operator must be 'Smaller' or 'Bigger'")
        
        # If all criteria are met, simulate contact with the government
        if check:
            contact()
        # print(f'\tNew flood awareness: {agent.threat_vulnerability}\
        #     \n\tNew self_efficacy: {agent.self_efficacy}')


