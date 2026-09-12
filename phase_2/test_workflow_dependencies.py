from workflow_dependencies import DependencyResolver
from workflow_state import State


class FakeRoutingAgent:
    """Minimal routing agent needed to construct DependencyResolver."""

    def get_embedding(self, text):
        # Dependency generation tests do not care about semantic classification,
        # but DependencyResolver caches embeddings during initialization.
        return [1.0, 0.0]


calls = []


def fake_product_manager(prompt, state):
    calls.append("product_manager")

    return (
        "US01: As a customer, I want automatic email routing "
        "so that I can receive faster support."
    )


def fake_program_manager(prompt, state):
    calls.append("program_manager")

    return """
Feature Name: Automatic Email Routing
Description: Routes incoming emails automatically.
Key Functionality:
- Analyze incoming email
- Route email to the appropriate destination
User Benefit: Customers receive faster support.
""".strip()


def fake_development_engineer(prompt, state):
    calls.append("development_engineer")

    return """
Task ID: US01-DEV01
Task Title: Implement automatic email routing
Related User Story: US01
Description: Implement routing logic for incoming email.
Acceptance Criteria: Incoming emails are routed correctly.
Estimated Effort: 3 days
Dependencies: None
""".strip()


agents = [
    {
        "name": "product_manager",
        "func": fake_product_manager,
    },
    {
        "name": "program_manager",
        "func": fake_program_manager,
    },
    {
        "name": "development_engineer",
        "func": fake_development_engineer,
    },
]


def create_resolver():
    return DependencyResolver(
        routing_agent=FakeRoutingAgent(),
        agents=agents,
        product_spec="Test product specification",
    )


def test_development_tasks_generate_user_stories_first():
    calls.clear()

    state = State()
    resolver = create_resolver()

    resolver._ensure_artifact(
        "development_tasks",
        state,
    )

    assert calls == [
        "product_manager",
        "development_engineer",
    ], calls

    assert len(state.user_stories) > 0
    assert len(state.development_tasks) > 0

    # Development tasks should NOT require product features.
    assert len(state.product_features) == 0

    print("development_tasks generates user_stories first")


def test_existing_user_stories_are_reused():
    calls.clear()

    state = State(
        user_stories=[
            "US01: As a customer, I want email routing "
            "so that I receive faster support."
        ]
    )

    resolver = create_resolver()

    resolver._ensure_artifact(
        "development_tasks",
        state,
    )

    assert calls == [
        "development_engineer"
    ], calls

    assert len(state.development_tasks) > 0

    print("existing user_stories are reused")


def test_existing_development_tasks_are_reused():
    calls.clear()

    state = State()

    state.add_development_tasks(
        """
Task ID: US01-DEV01
Task Title: Existing task
Related User Story: US01
Description: Existing development task.
Acceptance Criteria: Task works.
Estimated Effort: 1 day
Dependencies: None
""".strip()
    )

    resolver = create_resolver()

    resolver._ensure_artifact(
        "development_tasks",
        state,
    )

    # No agent should run because the artifact already exists.
    assert calls == [], calls

    assert len(state.development_tasks) == 1

    print("existing development_tasks are reused")


def test_product_features_generate_user_stories_first():
    calls.clear()

    state = State()
    resolver = create_resolver()

    resolver._ensure_artifact(
        "product_features",
        state,
    )

    assert calls == [
        "product_manager",
        "program_manager",
    ], calls

    assert len(state.user_stories) > 0
    assert len(state.product_features) > 0

    print("product_features generates user_stories first")


def test_development_tasks_do_not_require_product_features():
    calls.clear()

    state = State()
    resolver = create_resolver()

    resolver._ensure_artifact(
        "development_tasks",
        state,
    )

    assert "program_manager" not in calls
    assert len(state.product_features) == 0

    print("development_tasks do not require product_features")


if __name__ == "__main__":
    test_development_tasks_generate_user_stories_first()
    test_existing_user_stories_are_reused()
    test_existing_development_tasks_are_reused()
    test_product_features_generate_user_stories_first()
    test_development_tasks_do_not_require_product_features()

    print("\nAll dependency tests passed.")