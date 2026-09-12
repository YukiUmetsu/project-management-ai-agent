# workflow_state.py
from pydantic import BaseModel
from typing import Optional
import re

class DevelopmentTask(BaseModel):
    """
    A DevelopmentTask is a task that needs to be completed to develop a product.
    It includes the task ID, title, related user story, description, acceptance criteria, estimated effort, and dependencies.
    """
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
    """
    A ProductFeature is a feature that needs to be developed for a product.
    It includes the feature name, user stories, description, key functionality, and user benefit.
    """
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
    """
    A State is a state that contains the user stories, product features, and development tasks.
    """
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
