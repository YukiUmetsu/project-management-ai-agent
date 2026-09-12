
# TODO: 1 - Import the KnowledgeAugmentedPromptAgent and RoutingAgent
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"

# TODO: 2 - Define the Texas Knowledge Augmented Prompt Agent
texas_knowledge = "You know everything about Texas"

# TODO: 3 - Define the Europe Knowledge Augmented Prompt Agent
europe_knowledge = "You know everything about Europe"

# TODO: 4 - Define the Math Knowledge Augmented Prompt Agent
math_knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"


texas_knowledge_augmented_prompt_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, texas_knowledge)
europe_knowledge_augmented_prompt_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, europe_knowledge)
math_knowledge_augmented_prompt_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, math_knowledge)

agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_knowledge_augmented_prompt_agent.respond(x)
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_knowledge_augmented_prompt_agent.respond(x)
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_knowledge_augmented_prompt_agent.respond(x)
    }
]

routing_agent = RoutingAgent(openai_api_key, agents)

# TODO: 8 - Print the RoutingAgent responses to the following prompts:
#           - "Tell me about the history of Rome, Texas"
#           - "Tell me about the history of Rome, Italy"
#           - "One story takes 2 days, and there are 20 stories"
print(routing_agent.route("Tell me about the history of Rome, Texas"))
print(routing_agent.route("Tell me about the history of Rome, Italy"))
print(routing_agent.route("One story takes 2 days, and there are 20 stories"))