from agentMET4FOF.agents import AgentNetwork

from agent_definitions import (
    ConnectorAgent,
    DeconvolutionAgent,
    InputUncertaintyAgent,
    InterpolationAgent,
    MetrologicalMonitorAgent_multichannel,
    DumpAgent,
)


def chain_of_agents():
    # NOTE:
    # ConnectorAgent requires a running instance of "simulate_board.py" in 
    # a separate window
    
    agent_network = AgentNetwork(backend="mesa", dashboard_modules=True, log_filename=False)

    # init agents
    kwargs = {"log_mode": False, "input_data_maxlen": 5000, "output_data_maxlen": 5000}
    agent_1 = agent_network.add_agent(agentType=ConnectorAgent, **kwargs)
    agent_2 = agent_network.add_agent(agentType=InputUncertaintyAgent, **kwargs)
    agent_3 = agent_network.add_agent(agentType=InterpolationAgent, **kwargs)
    agent_4 = agent_network.add_agent(agentType=DeconvolutionAgent, **kwargs)
    agent_5 = agent_network.add_agent(agentType=MetrologicalMonitorAgent_multichannel, **kwargs)
    #agent_6 = agent_network.add_agent(agentType=DumpAgent, **kwargs)

    # establish processing chain
    agent_1.bind_output(agent_2)
    agent_2.bind_output(agent_3)
    agent_3.bind_output(agent_4)

    # plot raw and processed signal
    agent_4.bind_output(agent_5)
    agent_1.bind_output(agent_5)

    # dump raw and processed signal
    #agent_1.bind_output(agent_6)
    #agent_3.bind_output(agent_6)
    #agent_4.bind_output(agent_6)

    # start up
    agent_network.set_running_state()

    return agent_network

if __name__ == "__main__":
    chain_of_agents()
