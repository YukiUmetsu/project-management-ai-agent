# workflow_dependencies.py

import numpy as np
import logging
from workflow_state import State

logger = logging.getLogger(__name__)

# Dependencies between generated artifacts.
ARTIFACT_DEPENDENCIES = {
    "user_stories": [],
    "product_features": ["user_stories"],
    "development_tasks": ["user_stories"],
}


ARTIFACT_PRODUCERS = {
    "user_stories": "product_manager",
    "product_features": "program_manager",
    "development_tasks": "development_engineer",
}


ARTIFACT_GENERATION_PROMPTS = {
    "user_stories": (
        "Define the user stories for this product."
    ),
    "product_features": (
        "Define the product features based on the existing user stories."
    ),
    "development_tasks": (
        "Define the development tasks based on the existing user stories."
    ),
}


# Describes what information must already exist before a step can run.
REQUIRED_CONTEXT_DESCRIPTIONS = {
    "product_spec": (
        "The request requires information from the original product "
        "specification, such as product requirements, users, behavior, "
        "constraints, or functionality."
    ),

    "user_stories": (
        "The request requires existing user stories as input. "
        "Examples include creating product features, creating development "
        "tasks, or analyzing user needs and user stories."
    ),

    "product_features": (
        "The request requires existing product features for analysis, "
        "review, explanation, comparison, or reasoning about features."
    ),

    "development_tasks": (
        "The request requires existing development tasks for analysis, "
        "review, risk analysis, dependencies, effort analysis, acceptance "
        "criteria, or reasoning about engineering work."
    ),

    "none": (
        "The request can be answered directly without requiring the product "
        "specification, user stories, product features, or development tasks."
    ),
}


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


class DependencyResolver:
    def __init__(
        self,
        routing_agent,
        agents,
        product_spec,
    ):
        self.routing_agent = routing_agent
        self.product_spec = product_spec

        self.agents = {
            agent["name"]: agent
            for agent in agents
        }

        # Cache these embeddings instead of generating them for every step.
        self.context_embeddings = {
            name: routing_agent.get_embedding(description)
            for name, description
            in REQUIRED_CONTEXT_DESCRIPTIONS.items()
        }

    def ensure_requirements(
        self,
        step: str,
        selected_agent_name: str,
        state: State,
    ):
        """
        Ensure that any context required by the workflow step exists.

        This method only prepares dependencies. The selected agent still
        executes the original workflow step.
        """

        required_context = self._determine_required_context(
            step,
            selected_agent_name,
        )

        logger.info(
            f"[Dependency Resolver] "
            f"required context = {required_context}"
        )

        if required_context == "none":
            return

        if required_context == "product_spec":
            if not self.product_spec:
                raise RuntimeError(
                    "Product specification is required but unavailable."
                )
            return

        self._ensure_artifact(
            required_context,
            state,
        )

    def _determine_required_context(
        self,
        step: str,
        selected_agent_name: str,
    ) -> str:
        """
        Determine what information must already exist before this step runs.
        """

        text = (
            f"Selected specialist: {selected_agent_name}. "
            f"Requested step: {step}"
        )

        step_embedding = self.routing_agent.get_embedding(
            text
        )

        scores = {
            name: cosine_similarity(
                step_embedding,
                context_embedding,
            )
            for name, context_embedding
            in self.context_embeddings.items()
        }

        return max(
            scores,
            key=scores.get,
        )

    def _ensure_artifact(
        self,
        artifact_name: str,
        state: State,
    ):
        """
        Generate an artifact only when it does not already exist.
        """

        if self._artifact_exists(
            artifact_name,
            state,
        ):
            logger.info(
                f"[Dependency Resolver] "
                f"Reusing existing {artifact_name}"
            )
            return

        # Generate prerequisites first.
        for dependency in ARTIFACT_DEPENDENCIES.get(
            artifact_name,
            [],
        ):
            self._ensure_artifact(
                dependency,
                state,
            )

        producer_name = ARTIFACT_PRODUCERS[
            artifact_name
        ]

        producer = self.agents[
            producer_name
        ]

        prompt = ARTIFACT_GENERATION_PROMPTS[
            artifact_name
        ]

        logger.info(
            f"[Dependency Resolver] "
            f"Generating missing {artifact_name} "
            f"with {producer_name}"
        )

        result = producer["func"](
            prompt,
            state,
        )

        if not result:
            raise RuntimeError(
                f"Failed to generate {artifact_name}"
            )

        self._update_state(
            producer_name,
            result,
            state,
        )

    @staticmethod
    def _artifact_exists(
        artifact_name: str,
        state: State,
    ) -> bool:

        if artifact_name == "user_stories":
            return bool(state.user_stories)

        if artifact_name == "product_features":
            return bool(state.product_features)

        if artifact_name == "development_tasks":
            return bool(state.development_tasks)

        return False

    @staticmethod
    def _update_state(
        agent_name: str,
        result: str,
        state: State,
    ):
        if agent_name == "product_manager":
            state.add_user_stories(result)

        elif agent_name == "program_manager":
            state.add_product_features(result)

        elif agent_name == "development_engineer":
            state.add_development_tasks(result)