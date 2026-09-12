# TODO: 1 - import the OpenAI class from the openai library
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
import random
import time
import logging

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

logger = logging.getLogger(__name__)

MODEL = "gpt-3.5-turbo"
BASE_URL = "https://openai.vocareum.com/v1"
EMBEDDING_MODEL = "text-embedding-3-large"

class Agent:
    def __init__(self, openai_api_key, persona=None):
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(
        self,
        messages,
        model=MODEL,
        temperature=0,
        max_attempts=3,
        base_delay=2.0,
    ):
        """
        Generates a response from the OpenAI API.

        Parameters:
        messages (list): The messages to send to the OpenAI API.
        model (str): The model to use for the response.
        temperature (float): The temperature to use for the response.
        max_attempts (int): Maximum number of attempts to get the response.
        base_delay (float): Base delay in seconds.

        Returns:
        str: The response from the OpenAI API.
        """

        client = OpenAI(
            base_url=BASE_URL,
            api_key=self.openai_api_key,
            max_retries=0,  # avoid double retrying
        )

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )

                return response.choices[0].message.content

            except (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                InternalServerError,
            ) as exc:
                if attempt == max_attempts:
                    raise

                # exponential backoff + jitter
                delay = base_delay * (2 ** (attempt - 1))
                delay += random.uniform(0, delay)

                logger.warning(
                    "LLM request retry agent=%s attempt=%d/%d error=%s delay=%.2fs",
                    self.persona,
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    delay,
                )

                time.sleep(delay)

    def get_embedding(self, text, max_attempts=3, base_delay=2.0):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.
        max_attempts (int): Maximum number of attempts to get the embedding.
        base_delay (float): Base delay in seconds.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url=BASE_URL, api_key=self.openai_api_key)

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding

            except (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                InternalServerError,
            ) as exc:
                if attempt == max_attempts:
                    raise

                # exponential backoff + jitter
                delay = base_delay * (2 ** (attempt - 1))
                delay += random.uniform(0, delay)

                logger.warning(
                    "LLM request retry agent=%s attempt=%d/%d error=%s delay=%.2fs",
                    self.persona,
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    delay,
                )

                time.sleep(delay)


# DirectPromptAgent class definition
class DirectPromptAgent(Agent):

    def respond(self, prompt):
        try:
            return super().respond([{"role": "user", "content": prompt}])
        except Exception as e:
            print(f"Error in DirectPromptAgent: {e}")
            return None


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text):
        messages=[
            {"role": "system", "content": f"You are {self.persona}. Forget previous context."},
            {"role": "user", "content": input_text},
        ]
        try:
            return super().respond(messages)
        except Exception as e:
            print(f"Error in AugmentedPromptAgent: {e}")
            return None

# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent(Agent):
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        messages=[
                # - The persona with the following instruction:
                #   "You are _persona_ knowledge-based assistant. Forget all previous context."
                # - The provided knowledge with this instruction:
                #   "Use only the following knowledge to answer, do not use your own knowledge: _knowledge_"
                # - Final instruction:
                #   "Answer the prompt based on this knowledge, not your own."
                {"role": "system", "content": f"You are {self.persona}, knowledge-based assistant. Forget allprevious context. Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}. Answer the prompt based on this knowledge, not your own."},
                {"role": "user", "content": input_text}
            ]
        try:
            return super().respond(messages)
        except Exception as e:
            print(f"Error in KnowledgeAugmentedPromptAgent: {e}")
            return None


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent(Agent):
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, end, chunk_id = [], 0, 0, 0

        while end < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()

            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        messages=[
            {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
            {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
        ]

        try:
            return super().respond(messages)
        except Exception as e:
            logger.error(f"Error in RAGKnowledgePromptAgent: {e}")
            return None


class EvaluationAgent(Agent):

    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        # Initialize the EvaluationAgent with given attributes.
        # TODO: 1 - Declare class attributes here
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        prompt_to_evaluate = initial_prompt
        evaluation = ""

        for i in range(self.max_interactions):
            print(f"\n\n------ Evaluation Agent Interaction {i+1} ------")

            print("\n Step 1: Worker agent generates a response to the prompt")
            print(f"\n Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"\n Worker Agent Response:\n{response_from_worker}")

            print("\n\n Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            messages=[
                {"role": "system", "content": f"You are a {self.persona}, an evaluator agent. You are given a response and a set of criteria. You need to determine if the response meets the criteria. Respond Yes or No, and the reason why it does or doesn't meet the criteria."},
                {"role": "user", "content": eval_prompt}
            ]

            try:
                evaluation = super().respond(messages).strip()
            except Exception as e:
                print(f"Error in EvaluationAgent evaluation step: {e}")
                return None

            print(f"\n Evaluator Agent Evaluation:\n{evaluation}")

            print("\n\n Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print("\n\n Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                    "Only identify the necessary corrections. Preserve all content that is already correct."
                )
                messages=[
                    {"role": "system", "content": f"You are a {self.persona}, an evaluator agent. You are given a response and a set of criteria. You need to generate instructions to fix the response to meet the criteria."},
                    {"role": "user", "content": instruction_prompt}
                ]
                try:
                    instructions = super().respond(messages).strip()
                except Exception as e:
                    print(f"Error in EvaluationAgent correction step: {e}")
                    return None

                print(f"Instructions to fix:\n{instructions}")

                print("\n\n Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                    "Return the complete revised response. "
                    "Preserve all valid existing content and do not remove, omit, or renumber "
                    "items unless the correction explicitly requires it."
                )
        return {
            # TODO: 7 - Return a dictionary containing the final response, evaluation, and number of iterations
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "number_of_iterations": i + 1
        }

class RoutingAgent(Agent):

    def __init__(self, openai_api_key, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        self.agents = agents

    def route(self, user_input):
        # TODO: 4 - Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # TODO: 5 - Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)

            # TODO: 6 - Add logic to select the best agent based on the similarity score between the user prompt and the agent descriptions
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent


class ActionPlanningAgent(Agent):

    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):

        # TODO: 2 - Instantiate the OpenAI client using the provided API key
        client = OpenAI(base_url=BASE_URL, api_key=self.openai_api_key)
        # TODO: 3 - Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        # Provide the following system prompt along with the user's prompt:
        # "You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {pass the knowledge here}"
        messages=[
            {"role": "system", "content": f"You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {self.knowledge}"},
            {"role": "user", "content": prompt}
        ]

        try:
            response_text = super().respond(messages).strip()
        except Exception as e:
            print(f"Error in ActionPlanningAgent: {e}")
            return None

        # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
        steps = re.findall(r"^\d+\.\s*.+$", response_text, flags=re.MULTILINE)
        return steps
