# agentic_workflow.py

# TODO: 1 - Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module

import os
from pathlib import Path
from dotenv import load_dotenv
from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent
from pydantic import BaseModel
from typing import Optional
import re

class DevelopmentTask(BaseModel):
    task_id: Optional[str] = ""
    task_title: Optional[str] = ""
    related_user_story: Optional[str] = ""
    description: Optional[str] = ""
    acceptance_criteria: Optional[str] = ""
    estimated_effort: Optional[str] = ""
    dependencies: Optional[list[str]] = []

    def __str__(self):
        result = f"Task ID: {self.task_id}\n"

        if self.task_title:
            result += f"Task Title: {self.task_title}\n"

        if self.related_user_story:
            result += f"Related User Story: {self.related_user_story}\n"

        if self.description:
            result += f"Description: {self.description}\n"

        if self.acceptance_criteria:
            result += f"Acceptance Criteria: {self.acceptance_criteria}\n"

        if self.estimated_effort:
            result += f"Estimated Effort: {self.estimated_effort}\n"

        if self.dependencies:
            result += f"Dependencies: {', '.join(self.dependencies)}\n"

        return result

    def to_dict(self):
        return self.model_dump()

    def from_dict(self, dict):
        return DevelopmentTask(**dict)

    def to_json(self):
        return self.model_dump_json()

    def from_json(self, json):
        return DevelopmentTask.model_validate_json(json)

class ProductFeature(BaseModel):
    feature_name: Optional[str] = ""
    user_stories: Optional[list[str]] = []
    description: Optional[str] = ""
    key_functionality: Optional[list[str]] = []
    user_benefit: Optional[str] = ""

    def __str__(self):
        result = f"Feature Name: {self.feature_name}\n"

        if self.description:
            result += f"Description: {self.description}\n"

        if self.key_functionality:
            result += (
                "Key Functionality:\n"
                + "\n".join(f"- {item}" for item in self.key_functionality)
                + "\n"
            )

        if self.user_benefit:
            result += f"User Benefit: {self.user_benefit}\n"

        if self.user_stories:
            result += (
                "User Stories:\n"
                + "\n".join(f"- {story}" for story in self.user_stories)
                + "\n"
            )

        return result + "\n"

    def to_dict(self):
        return self.model_dump()

class State(BaseModel):
    user_stories: Optional[list[str]] = []
    product_features: Optional[list[ProductFeature]] = []
    development_tasks: Optional[list[DevelopmentTask]] = []
    development_plan: Optional[str] = ""

    def add_user_stories(self, text):
        # break down the text into user stories that includes 'As a ~'
        user_stories = []
        for line in text.split("\n"):
            if "As a" in line:
                user_stories.append(line)
        self.user_stories.extend(user_stories)

    def add_product_features(self, text):
        # break down the text into product features that includes 'Feature Name: ~'
        # Map section headers to Pydantic field names.
        header_map = {
            "Feature Name": "feature_name",
            "Description": "description",
            "Key Functionality": "key_functionality",
            "User Benefit": "user_benefit",
        }

        # Split the text whenever a new "Feature Name:" starts.
        feature_blocks = re.split(r"(?=^Feature Name:\s*)", text, flags=re.MULTILINE)

        features = []

        for block in feature_blocks:
            # Ignore text before the first feature.
            if not block.strip():
                continue

            # Find all recognized headers inside this feature block.
            pattern = r"(?m)^(Feature Name|Description|Key Functionality|User Benefit):\s*"
            matches = list(re.finditer(pattern, block))

            if not matches:
                continue

            data = {}

            # Extract the content between each header and the next header.
            for i, match in enumerate(matches):
                key = header_map[match.group(1)]

                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(block)

                value = block[start:end].strip()

                # Convert bullet points into a list.
                if key == "key_functionality":
                    value = [
                        line.removeprefix("-").strip()
                        for line in value.splitlines()
                        if line.strip()
                    ]

                data[key] = value

            # Validate and add the feature.
            features.append(ProductFeature(**data))
        self.product_features.extend(features)

    def add_development_tasks(self, text):
        # break down the text into development tasks that includes 'Task ID: ~'
        development_tasks = []
        # Map text headers to Pydantic field names.
        header_map = {
            "Task ID": "task_id",
            "Task Title": "task_title",
            "Related User Story": "related_user_story",
            "Description": "description",
            "Acceptance Criteria": "acceptance_criteria",
            "Estimated Effort": "estimated_effort",
            "Dependencies": "dependencies",
        }

        # Split whenever a new Task ID appears.
        # \s* allows spaces/tabs before "Task ID".
        task_blocks = re.split(
            r"(?=^\s*Task ID\s*:)",
            text,
            flags=re.MULTILINE,
        )

        # Match any recognized field, allowing whitespace around the header/colon.
        header_pattern = re.compile(
            r"^\s*(Task ID|Task Title|Related User Story|Description|"
            r"Acceptance Criteria|Estimated Effort|Dependencies)\s*:\s*",
            flags=re.MULTILINE,
        )

        for block in task_blocks:
            block = block.strip()

            # Skip anything before the first Task ID.
            if not block or not re.match(r"^Task ID\s*:", block):
                continue

            matches = list(header_pattern.finditer(block))
            data = {}

            for i, match in enumerate(matches):
                field_name = header_map[match.group(1)]

                start = match.end()
                end = (
                    matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(block)
                )

                value = block[start:end].strip()

                if field_name == "dependencies":
                    # Supports either:
                    #
                    # Dependencies: Some dependency
                    #
                    # or:
                    #
                    # Dependencies:
                    # - Dependency A
                    # - Dependency B
                    value = [
                        line.strip().removeprefix("-").strip()
                        for line in value.splitlines()
                        if line.strip()
                    ]

                data[field_name] = value

            development_tasks.append(DevelopmentTask(**data))

        self.development_tasks.extend(development_tasks)

    def to_dict(self):
        return self.model_dump()

    def from_dict(self, dict):
        return State(**dict)

    def to_json(self):
        return self.model_dump_json()

    def from_json(self, json):
        return State.model_validate_json(json)

    def create_development_plan(self):
        """
        Create a development plan for a product.
        The development plan is a list of user stories, product features, and development tasks.
        The development plan is returned as a string.
        """
        result = ""
        if len(self.user_stories) > 0:
            result += f"#User Stories:\n"
            for story in self.user_stories:
                result += f"{story}\n"

        if len(self.product_features) > 0:
            result += f"\n# Product Features:\n"
            for feature in self.product_features:
                result += f"{feature}\n"

        if len(self.development_tasks) > 0:
            result += f"\n# Development Tasks:\n"
            for task in self.development_tasks:
                result += f"{task}\n"
        return result

    def save_development_plan(self, file_path):
        with open(file_path, "w") as file:
            file.write(self.create_development_plan())

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
load_dotenv(Path(__file__).parent / ".env")
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
with open(Path(__file__).parent / "Product-Spec-Email-Router.txt", "r") as file:
    product_spec = file.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components\n"
    "Example output format: \n"
    "1. Define the user stories with a persona, an action, and a desired outcome\n"
    "2. Define the product features by grouping related user stories\n"
    "3. Define the development tasks for each user story by identifying what needs to be built to implement each user story\n"
)
# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
    "Example output format: \n"
    "US01: As a user, I want to connect to designated email services and retrieve incoming messages promptly in real-time so that I can respond to messages quickly\n"
    "US02: As a user, I want to send emails to designated email services so that I can communicate with other users\n"
    "US03: As a user, I want to delete emails from designated email services so that I can manage my email inbox\n"
    f"The product spec is: {product_spec}"
)
# TODO: 6 - Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_product_manager, knowledge_product_manager)

# Product Manager - Evaluation Agent
# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
evaluation_criteria_product_manager = "The answer should be user stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
persona_product_manager_eval = "You are an product manager evaluation agent that checks the answers of other worker agents."
product_manager_evaluation_agent = EvaluationAgent(openai_api_key, persona_product_manager_eval, evaluation_criteria_product_manager, product_manager_knowledge_agent, 10)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into cohesive groups."
    "\n"
    "IMPORTANT REQUIREMENTS:\n"
    "- You MUST include EVERY provided user story in a product feature.\n"
    "- Do not omit any user story.\n"
    "- Every user story ID must appear in the User Stories field of at least one feature.\n"
    "- Group related user stories together when appropriate.\n"
    "- Before answering, verify that all provided user story IDs are included.\n"
    "\n"
    "Output each feature using this exact format:\n"
    "Feature Name: A clear feature name\n"
    "Description: A description of the feature\n"
    "Key Functionality:\n"
    "- Functionality 1\n"
    "- Functionality 2\n"
    "User Benefit: How the feature benefits the user\n"
    "User Stories: [US01, US02]\n"
)
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
# (This is a necessary step before TODO 8. Students should add the instantiation code here.)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_program_manager, knowledge_program_manager)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

# TODO: 8 - Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria_program_manager = "The answer should be product features that follow the following structure: Feature Name: A clear, concise title that identifies the capability\nDescription: A brief explanation of what the feature does and its purpose\nKey Functionality: The specific capabilities or actions the feature provides\nUser Benefit: How this feature creates value for the user"
program_manager_evaluation_agent = EvaluationAgent(openai_api_key, persona_program_manager_eval, evaluation_criteria_program_manager, program_manager_knowledge_agent, 10)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to implement each user story.\n"
    "You're allowed to use your own software engineering knowledge to break down the user story into smaller development tasks, consider frontend, backend, database, infrastructure, etc. if necessary.\n"
    "Task ID should follow the format: US[related user story number]-DEV[development task number]\n"
    "Example output format:\n"
    "Task ID: US1-DEV01\n"
    "Task Title: Connect to designated email services and retrieve incoming messages promptly in real-time\n"
    "Related User Story: User Story ID 1, Reduces response time\n"
    "Description: Implement a system to connect to designated email services and retrieve incoming messages promptly in real-time\n"
    "Acceptance Criteria: Retrieved emails are stored in a database and displayed in a user interface\n"
    "Estimated Effort: 3 days\n"
    "Dependencies: None\n"
)
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
# (This is a necessary step before TODO 9. Students should add the instantiation code here.)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_dev_engineer, knowledge_dev_engineer)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
# TODO: 9 - Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria_dev_engineer = "The answer should be tasks that follow the following structure: Task ID: A unique identifier for tracking purposes\nTask Title: Brief description of the specific development work\nRelated User Story: Reference to the parent user story\nDescription: Detailed explanation of the technical work required\nAcceptance Criteria: Specific requirements that must be met for completion\nEstimated Effort: Time or complexity estimation\nDependencies: Any tasks that must be completed first"
development_engineer_evaluation_agent = EvaluationAgent(openai_api_key, persona_dev_engineer_eval, evaluation_criteria_dev_engineer, development_engineer_knowledge_agent, 10)

# Routing Agent
# TODO: 10 - Instantiate a routing_agent. You will need to define a list of agent dictionaries (routes) for Product Manager, Program Manager, and Development Engineer. Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). Assign this list to the routing_agent's 'agents' attribute.

# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.

def product_manager_support_function(x, state):
    """
    Support function for the product manager agent.
    EvalueteAgent.evaluete will call the agent respond function, so we don't need to call the agent respond function here.
    """
    return product_manager_evaluation_agent.evaluate(x)["final_response"]

def program_manager_support_function(x, state):
    """
    Support function for the program manager agent.
    EvalueteAgent.evaluete will call the agent respond function, so we don't need to call the agent respond function here.
    """
    user_stories = "\n".join(state.user_stories)
    x += f"\nUser Stories: {user_stories}"
    return program_manager_evaluation_agent.evaluate(x)["final_response"]

def development_engineer_support_function(x, state):
    """
    Support function for the development engineer agent.
    EvalueteAgent.evaluete will call the agent respond function, so we don't need to call the agent respond function here.
    """
    user_stories = "\n".join(state.user_stories)
    x += f"\nUser Stories: {user_stories}"
    return development_engineer_evaluation_agent.evaluate(x)["final_response"]

agents = [
    {
        "name": "product_manager",
        "description": (
            "Responsible for defining product personas and user stories only. "
            "Creates stories in the form: As a [user], I want [action] so that [benefit]. "
            "Does not define features or engineering tasks. Does not group stories."
        ),
        "func": product_manager_support_function
    },
    {
        "name": "program_manager",
        "description": (
            "Responsible for defining product features by grouping related user stories. "
            "Outputs feature name, description, key functionality, and user benefit. "
            "Does not write user stories or engineering tasks."
        ),
        "func": program_manager_support_function
    },
    {
        "name": "development_engineer",
        "description": (
            "Responsible for defining detailed engineering development tasks for implementation "
            "based on the user stories and product features. "
            "Creates tasks with task ID, title, related user story, description, "
            "acceptance criteria, estimated effort, and dependencies. "
            "Does not write user stories or product features."
        ),
        "func": development_engineer_support_function
    }
]
routing_agent = RoutingAgent(openai_api_key, agents)

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# TODO: 12 - Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).
completed_steps = []
state = State(user_stories=[], product_features=[], development_tasks=[])
steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"Total steps to complete: {len(steps)}\n")
for step in steps:
    print(f"\n----------------------------------------------------------------\n")
    print(step)
    print(f"\n----------------------------------------------------------------\n")
    best_agent = routing_agent.route(step)
    current_step_result = best_agent["func"](step, state)
    if best_agent["name"] == "product_manager":
        state.add_user_stories(current_step_result)
    elif best_agent["name"] == "program_manager":
        state.add_product_features(current_step_result)
    elif best_agent["name"] == "development_engineer":
        state.add_development_tasks(current_step_result)
    completed_steps.append(current_step_result)
    print(f"Step {step} completed: {completed_steps[-1]}")

development_plan = state.create_development_plan()
print(f"Final output of the workflow: {development_plan}")

# save the completed steps to a file
with open("completed_steps.log", "w") as file:
    file.write(development_plan)