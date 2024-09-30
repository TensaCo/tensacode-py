# TensaCode: An Open Source Neurosymbolic Programming Framework for Learning Structured Programs

![TensaCode Logo](assets/img/logo-color.png)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

> **TensaCode**: Machine Learning + Software Engineering + Runtime Introspection and Code Generation = Programming 2.0

TensaCode is a framework that introduces simple abstractions and functions for encoding, decoding, querying, and manipulating arbitrary Python objects, differentiable programming (including differentiable control flow), and intelligent runtime code generation and execution. Rather than focusing solely on underlying mathematical objects, TensaCode's abstractions enable you to leverage decades of software engineering patterns and paradigms to concisely approach your underlying problem. If you’re building models that require highly structured inputs or outputs (e.g., complex multimodal architectures), researching the next generation of end-to-end differentiable cognitive architectures, or adding learning capabilities on top of an existing program, TensaCode may fit right into your stack!

## 🚀 Quick Start

### Installation

```bash
pip install tensacode
```

### Basic Usage

```python
from tensacode.core.base.base_engine import BaseEngine

# Create an instance of the Engine
engine = BaseEngine()

# Encode some data
text_latent = engine.encode("Hello, world!")

# Decode the latent representation
decoded_text = engine.decode(latent=text_latent, type=str)
print(decoded_text)  # Output: "Hello, world!"
```

## 📖 Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Concepts](#concepts)
- [Code Organization](#code-organization)
- [Operations (Ops)](#operations-ops)
- [Latent Types](#latent-types)
- [Examples](#examples)
  - [Multimodal Reinforcement Learning: GUI Assistant (Vision + Language → Keyboard + Mouse)](#multimodal-reinforcement-learning-gui-assistant-vision--language--keyboard--mouse)
  - [Advanced Agent Architecture with Differentiable Programming](#advanced-agent-architecture-with-differentiable-programming)
  - [Natural Language Data Processing](#natural-language-data-processing)
  - [Graph Data Manipulation](#graph-data-manipulation)
  - [Going Further](#going-further)
- [Contributing](#contributing)
- [License](#license)

## Introduction

TensaCode provides a streamlined approach to building AI engines by centralizing all functionalities into a single `Engine` class. This design enhances flexibility, extensibility, and maintainability, allowing developers to add new operations and latent types without modifying the core engine logic.

## Key Features

- **Single Engine Class**: Centralized management of all operations through the `Engine` class.
- **Dynamic Operation Dispatching**: Automatically selects the appropriate operation based on argument types and specified latent types.
- **Operations Specify Latent Types**: Operations define the latent types they operate with, enabling flexibility.
- **Context Management**: Hierarchical context system for managing state and logging.
- **Extensible Latent Types**: Support for various latent representation types like text, images, audio, etc.
- **Comprehensive Logging**: Methods for commands, notes, feedback, and general information.
- **Differentiable Programming**: Support for differentiable control flow and integrating machine learning models.
- **Runtime Code Generation**: Intelligently generate and execute code at runtime for a given objective.

## Concepts

- **Engine**: The central class that manages operations and execution.
- **Operation**: Represents an action that the Engine can perform, specified with the `@Engine.register_op` decorator.
- **Latent Types**: Classes representing different types of latent representations (e.g., `TextLatent`, `ImageLatent`).
- **Dynamic Dispatching**: The process by which the Engine selects the most suitable operation based on input types.
- **Intermediate Representation (IR)**: Represents objects in a format suitable for processing, such as vector embeddings or abstract syntax trees.

## Code Organization

```plaintext
tensacode
├── core
│   ├── base
│   │   ├── base_engine.py       # Contains the Engine class
│   │   ├── ops                  # Definitions of operations
│   │   │   ├── encode.py
│   │   │   ├── decode.py
│   │   │   ├── modify.py
│   │   │   └── ...              # Other operations
│   │   └── latents
│   │       ├── latents.py       # Definitions of latent types
│   ├── llm
│   │   └── ops                  # LLM-specific operations
│   └── utils
│       ├── misc.py              # Miscellaneous utility functions
│       ├── locator.py           # Locator system for nested data access
│       └── language.py          # Language-related utilities
├── internal
│   ├── consts.py                # Internal constants
│   └── utils
│       └── ...                  # Internal utilities
└── examples
    ├── document_processing.py   # Example showcasing document processing
    ├── tensacode_agent          # Intelligent agents using TensaCode
    ├── tensagame                # Games built with TensaCode
    ├── tensaware                # Aware systems utilizing TensaCode
    └── ...                      # Other examples
```

## Operations (Ops)

TensaCode includes a wide range of built-in operations:

- **encode**: Encode an object into a latent representation.
- **decode**: Decode a latent representation back into an object.
- **modify**: Modify an object iteratively.
- **transform**: Transform inputs into a new form.
- **predict**: Predict the next item or value based on input data.
- **query**: Query an object for specific information.
- **correct**: Correct errors in the input data.
- **convert**: Convert between different object types.
- **select**: Select a specific value from a composite object.
- **similarity**: Compute similarity between two objects.
- **split**: Split an object into multiple components.
- **locate**: Locate a specific part of an input object.
- **plan**: Generate a plan based on prompts or context.
- **program**: Generate code or functions for execution.
- **decide**: Make a decision based on input data.
- **call**: Call a function by obtaining necessary arguments.

## Latent Types

Latent types represent the internal representations used by the Engine:

- **LatentType**: Base class for latent representations.
- **TextLatent**: Represents textual data in latent form.
- **ImageLatent**: Represents image data in latent form.
- **AudioLatent**: Represents audio data in latent form.
- **VideoLatent**: Represents video data in latent form.
- **VectorLatent**: Represents data as vector embeddings.
- **GraphLatent**: Represents graph structures in latent form.

## Examples

### Multimodal Reinforcement Learning: GUI Assistant (Vision + Language → Keyboard + Mouse)

TensaCode can be utilized to create models that require highly structured inputs or outputs, such as multimodal reinforcement learning environments. Here's a comprehensive example that consolidates previous RL examples and aligns with the actual code in `tensacode`.

Imagine you're programming an agent that interacts with an environment using various input and output modalities:

```python
from tensacode.core.base.base_engine import BaseEngine
import pyautogui
import speech_recognition as sr
from typing import Tuple

# Initialize the TensaCode engine
engine = BaseEngine()

# Define the observation data structure
class Observation:
    def __init__(self, vision, text: str):
        self.vision = vision  # Visual input, e.g., a screenshot
        self.text = text      # Textual input, e.g., a voice command

# Define the action data structure
class Action:
    def __init__(self, mouse_action=None, keyboard_input=None, speech_output=None):
        self.mouse_action = mouse_action      # e.g., {'move': (x, y), 'click': True}
        self.keyboard_input = keyboard_input  # e.g., 'Hello World'
        self.speech_output = speech_output    # e.g., 'How can I assist you?'

# Define the agent's step function
def step(obs: Observation) -> Action:
    # Encode the observation into a latent representation
    obs_latent = engine.encode(obs)
    
    # Decide on the best action based on the observation
    action_latent = engine.predict(inputs=[obs_latent])
    
    # Decode the latent action into the Action data structure
    action = engine.decode(type_=Action, latent=action_latent)
    return action

# Example usage
if __name__ == "__main__":
    # Gather visual input (screenshot)
    vision_input = pyautogui.screenshot()

    # Gather language input (voice command)
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice command...")
        audio = recognizer.listen(source)
    try:
        language_input = recognizer.recognize_google(audio)
        print(f"Recognized: {language_input}")
    except Exception as e:
        print(f"Error recognizing speech: {e}")
        language_input = ""

    # Create an observation
    obs = Observation(vision=vision_input, text=language_input)
    
    # Get the action from the agent
    action = step(obs)

    # Execute the action
    def execute_action(action: Action):
        if action.mouse_action:
            x, y = action.mouse_action.get('move', (None, None))
            if x and y:
                pyautogui.moveTo(x, y)
            if action.mouse_action.get('click'):
                pyautogui.click()
        if action.keyboard_input:
            pyautogui.typewrite(action.keyboard_input)
        if action.speech_output:
            import pyttsx3
            tts_engine = pyttsx3.init()
            tts_engine.say(action.speech_output)
            tts_engine.runAndWait()
    
    execute_action(action)
```

**Key Concepts Illustrated**:

- **Declarative Approach**: By defining the `Observation` and `Action` classes, you focus on what data needs to be processed rather than how it should be serialized or handled internally.
- **Flexible Data Handling**: The `engine.encode` and `engine.decode` methods allow for seamless integration of complex data structures and modalities.
- **Simplified Integration**: Modifying existing modalities or adding new ones requires minimal changes in your code. For instance, you can expand the `Observation` class without worrying about serialization.

**Benefits**:

- **Reduce Boilerplate**: Eliminates the need for manual serialization and deserialization of complex data types.
- **Enhance Extensibility**: Easily adapt to changes in data structures or additional modalities.
- **Focus on Core Logic**: Allows you to concentrate on building the agent's logic rather than infrastructure code.

### Advanced Agent Architecture with Differentiable Programming

TensaCode supports runtime code generation and differentiable programming, which are instrumental in developing general cognitive architectures. Here's an example of a self-improving agent that utilizes these features:

```python
def step(self, obs: Observation) -> Action:
    # Encode the observation and retrieve memories
    obs_enc = engine.encode(obs)
    stm = engine.select(query=obs_enc @ self.W_stm, values=self.short_term_memory)
    ltm = engine.select(query=obs_enc @ self.W_ltm, values=self.long_term_memory)

    # Global workspace state
    perception_enc = self.perception_mlp(engine.combine(obs_enc, stm, ltm))
    perception = engine.decode(
        latent=perception_enc,
        type_=Perception,
        context={'vision': obs.vision, 'text': obs.text}
    )

    # Prediction
    prediction_enc = self.prediction_mlp(perception_enc)
    prediction = engine.decode(latent=prediction_enc, type_=Prediction, context=perception)

    # Update memories and prepare action
    self.short_term_memory.remember(
        engine.select('store short-term memories', prediction.thoughts)
    )
    self.long_term_memory.remember(
        engine.select('store long-term memories', prediction.thoughts)
    )
    action = prediction.action

    # Feedback signals
    self.add_loss('reward', prediction.pain - prediction.pleasure)
    self.add_loss('prediction_error', -engine.similarity(perception, self.prev_prediction))
    self.prev_prediction = prediction

    return action
```

**Key Concepts**:

- **Hierarchical Memory**: Utilizes both short-term and long-term memory modules for context-aware decision-making.
- **Global Workspace Theory**: Implements a workspace where different cognitive processes can share information.
- **Differentiable Losses**: Incorporates various loss functions for training, including cognitive load and prediction errors.

### Natural Language Data Processing

Yes, TensaCode is a superset of AI/LLM-frameworks. For example, you can use it to process and analyze large amounts of textual data:

```python
from tensacode.core.base.base_engine import BaseEngine

engine = BaseEngine()

documents = ["Document 1 text...", "Document 2 text...", "Document 3 text..."]

# Encode documents
encoded_docs = [engine.encode(doc) for doc in documents]

# Encode the query
query = "Find documents related to artificial intelligence"
query_latent = engine.encode(query)

# Compute similarities
similarities = [engine.similarity(doc_latent, query_latent) for doc_latent in encoded_docs]

# Rank documents based on similarity
ranked_docs = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
for doc, score in ranked_docs:
    print(f"Document: {doc}, Similarity Score: {score}")
```

**Explanation**:

- **Vector Representations**: Encodes both documents and queries into latent vectors for comparison.
- **Similarity Search**: Computes similarity scores to find the most relevant documents.

### Graph Data Manipulation

Manipulate and analyze complex, real-world data structures using Python objects with Pydantic schemas. In this example, we'll model a simple business scenario involving `Customers`, `Products`, `Orders`, `Vendors`, and `Suppliers`.

```python
from tensacode.core.base.base_engine import BaseEngine
from pydantic import BaseModel
from typing import List

engine = BaseEngine()

# Define the data models using Pydantic
class Product(BaseModel):
    id: int
    name: str
    price: float
    vendor: 'Vendor' = None  # Circular reference

class Vendor(BaseModel):
    id: int
    name: str
    products: List[Product] = []

class Customer(BaseModel):
    id: int
    name: str
    orders: List['Order'] = []

class Order(BaseModel):
    id: int
    products: List[Product]
    customer: Customer
    total_amount: float = 0.0

class Supplier(BaseModel):
    id: int
    name: str
    supplied_products: List[Product] = []

# Update forward references for self-referencing models
Product.update_forward_refs()
Vendor.update_forward_refs()
Customer.update_forward_refs()
Order.update_forward_refs()
Supplier.update_forward_refs()

# Create instances
# Vendors and Products
vendor1 = Vendor(id=1, name='Vendor A')
product1 = Product(id=1, name='Product 1', price=10.0, vendor=vendor1)
product2 = Product(id=2, name='Product 2', price=15.0, vendor=vendor1)
vendor1.products.extend([product1, product2])

vendor2 = Vendor(id=2, name='Vendor B')
product3 = Product(id=3, name='Product 3', price=20.0, vendor=vendor2)
vendor2.products.append(product3)

# Suppliers
supplier1 = Supplier(id=1, name='Supplier X', supplied_products=[product1, product3])

# Customers and Orders
customer1 = Customer(id=1, name='Customer John')
order1 = Order(id=1, products=[product1, product3], customer=customer1)
order1.total_amount = sum(p.price for p in order1.products)
customer1.orders.append(order1)

customer2 = Customer(id=2, name='Customer Jane')
order2 = Order(id=2, products=[product2], customer=customer2)
order2.total_amount = sum(p.price for p in order2.products)
customer2.orders.append(order2)

# Encode the data
data_to_encode = [customer1, customer2, vendor1, vendor2, supplier1]
business_data_latent = engine.encode(data_to_encode)
```

#### Answering Complex Questions

Using TensaCode's neural capabilities, we can answer nuanced questions that require inference and reasoning beyond straightforward database queries.

**Example 1: Identify Customers Likely Interested in New Products from Vendor A**

```python
# Define the query
query = "Which customers are likely to be interested in new products from Vendor A?"

# Perform the query
results_latent = engine.query(business_data_latent, query=query)

# Decode the results
interested_customers = engine.decode(type_=List[Customer], latent=results_latent)

print("Customers likely interested in new products from Vendor A:")
for customer in interested_customers:
    print(customer.name)
# Possible Output:
# Customers likely interested in new products from Vendor A:
# Customer John
```

**Explanation**:

- **Neural Reasoning**: The engine analyzes purchase history, customer preferences, and relationships to infer interest levels.
- **Implicit Patterns**: Goes beyond explicit data to identify patterns, such as a customer's affinity for certain vendors.

**Example 2: Predict Potential Supply Chain Issues**

```python
# Define the query
query = "Predict potential supply chain issues based on supplier and vendor relationships."

# Perform the query
issues_latent = engine.query(business_data_latent, query=query)

# Decode the results
supply_chain_issues = engine.decode(type_=List[str], latent=issues_latent)

print("Potential Supply Chain Issues:")
for issue in supply_chain_issues:
    print(f"- {issue}")
# Possible Output:
# Potential Supply Chain Issues:
# - Supplier X has limited capacity, which may affect Vendor A's product availability.
# - Dependency on single suppliers increases risk for Vendor B.
```

**Explanation**:

- **Complex Inference**: The engine assesses the interconnected data to predict issues that are not explicitly stated.
- **Risk Analysis**: Identifies vulnerabilities in the supply chain by analyzing dependencies and capacities.

**Example 3: Analyze Customer Sentiment Based on Purchase History**

```python
# Define the query
query = "Evaluate customer sentiment based on their purchase history and interactions."

# Perform the query
sentiment_latent = engine.query(business_data_latent, query=query)

# Decode the results
customer_sentiments = engine.decode(type_=List[dict], latent=sentiment_latent)

print("Customer Sentiments:")
for cs in customer_sentiments:
    print(f"Customer: {cs['customer_name']}, Sentiment: {cs['sentiment']}")
# Possible Output:
# Customer Sentiments:
# Customer: Customer John, Sentiment: Positive
# Customer: Customer Jane, Sentiment: Neutral
```

**Explanation**:

- **Sentiment Analysis**: Infers customer satisfaction levels from their orders and interactions.
- **Unstructured Data Handling**: May involve processing notes, feedback, or other unstructured data associated with orders.

**Key Concepts Illustrated**:

- **Neural Capabilities**: Utilizes neural reasoning to answer nuanced questions that require inference and pattern recognition.
- **Beyond Explicit Data**: Moves past direct queries to provide insights based on relationships and implicit information.
- **Flexible Querying**: Accepts natural language queries, making it accessible to non-technical users.

**Benefits**:

- **Insight Generation**: Provides deeper understanding of data, helping in decision-making processes.
- **Versatility**: Applicable to various domains where complex data relationships exist.
- **Efficiency**: Reduces the need for manual data analysis and complex query writing.

### Going Further

These are just a few examples of what you can do with TensaCode. The framework is designed to be highly extensible and customizable, allowing you to define your own operations and latent types.

```python
from tensacode.core.base.base_engine import BaseEngine

# Define a custom operation
@BaseEngine.register_op(name='greet')
def greet(engine: BaseEngine):
    name = engine.query("User name")  # Obtains user name using available context sources
    return f"Hello, {name}!"

# Use the operation
greeting = engine.greet()
print(greeting)  # Output: "Hello, <your name>!"
```

Some ideas for further exploration include:

- **Hybrid Latent Types**: Combine multiple latent types to represent complex data.
- **Differentiable Control Flow**: Integrate differentiable programming to allow for learning and optimization of control flow structures.
- **Runtime Architecture Generation**: Use the `engine.program` method to dynamically generate custom architectures or even extend its own architecture on-the-fly with self-healing capabilities.
## Contributing

We welcome contributions to TensaCode! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/<issue-number>-<short-description>`)
3. Make your changes
4. Commit your changes (`git commit -m '<changes>'`)
5. Push to the branch (`git push origin feature/<issue-number>-<short-description>`)
6. Create a new Pull Request

Please read our [contribution guidelines](CONTRIBUTING.md) for more details on our code of conduct, and the process for submitting pull requests.

## License

TensaCode is licensed under the [MIT License](LICENSE).

## Code of Conduct

We are committed to fostering an open and welcoming environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details on our code of conduct and the process for reporting unacceptable behavior.

## Support

If you're having issues or have questions, please:

1. Check the [documentation](https://tensacode.readthedocs.io/)
2. Look through [existing issues](https://github.com/TensaCo/tensacode-py/issues) or open a new one
3. Join our [community chat](https://discord.gg/tensacode) for real-time help

## Acknowledgements

I'd like to thank:

- Numerous amazing guys and gals who've contributed to the mountain of dependencies underneath this codebase
- Srikanth Srinivas for giving me the incentive to finish this!!!
- Rosanna from MLCollab for creating the community to get me started
- Andrew, whose ML courses got me hooked on AI in the first place
- Geoffrey, for giving his GLOM-presentation that stirred me to think beyond the feedforward paradigm

Your work has been our guiding light. Thank you all!!! :) :) :) 
