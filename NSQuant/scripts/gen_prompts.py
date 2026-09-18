# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
import random

templates = [
    "What is {n}+{m}?",
    "Explain {topic} in one sentence.",
    "Hello, how are you today?",
    "Write a short story about {adj} {noun}.",
    "Translate 'hello' to {lang}.",
    "Fix this bug: int x = {n};",
    "Once upon a time in a {adj} kingdom,",
    "Calculate {n} * {m}.",
    "Why is the sky blue?",
    "Give me a recipe for {food}.",
    "Tell me a joke about {noun}.",
    "What are the benefits of {topic}?",
    "How do I learn {topic}?",
    "Describe a {adj} day.",
    "Write a haiku about {noun}.",
    "Who invented {topic}?",
    "Is {n} a prime number?",
    "Sort the list [{n}, {m}, {p}].",
    "What does {lang} mean?",
    "The quick brown fox jumps over the lazy dog.",
    "Implement a function to add {n} and {m}.",
    "Compare {topic} and {topic2}.",
    "Convert {n} degrees Celsius to Fahrenheit.",
    "What is the capital of {place}?",
    "Generate a name for a {adj} {noun}.",
    "Why does {topic} matter?",
    "List three uses of {topic}.",
    "Once, a {noun} met another {noun}.",
    "Write a todo list for a {topic} project.",
    "Solve: {n}x + {m} = 0.",
    "Summarize the concept of {topic}.",
    "Design a database schema for {noun}.",
    "What is the binary form of {n}?",
    "Create a regex for {topic}.",
    "How many {noun}s fit in a {place}?",
    "Define {topic}.",
    "Write an email to request {noun}.",
    "What is the meaning of life?",
    "Make a pun about {topic}.",
    "Plot a {adj} character.",
    "Predict the next number: {n}, {m}, {p}, ...",
    "Refactor this: for i in range({n}):",
    "What is 2^{n}?",
    "Describe the color {adj}.",
    "How do you say goodbye in {lang}?",
    "List the first {m} prime numbers.",
    "Make a marketing slogan for {topic}.",
    "Why did the {noun} cross the road?",
    "What is recursion?",
    "Explain Q{q} quantization.",
]

adjectives = ["small", "large", "bright", "dark", "happy", "sad", "quick", "lazy", "loud", "quiet", "ancient", "modern", "cold", "warm"]
nouns = ["cat", "dog", "robot", "tree", "car", "house", "book", "computer", "star", "ocean", "city", "bird", "idea", "problem"]
topics = ["machine learning", "quantization", "physics", "history", "poetry", "programming", "mathematics", "space", "cooking", "travel"]
langs = ["Spanish", "French", "German", "Japanese", "Latin", "Russian", "Chinese", "Korean"]
places = ["France", "Japan", "Mars", "Egypt", "Brazil", "Australia", "Canada", "India"]
foods = ["pizza", "sushi", "pasta", "salad", "soup", "curry", "bread", "cake"]

def make(i):
    t = templates[i % len(templates)]
    return t.format(
        n=i % 37 + 1,
        m=(i*3) % 41 + 1,
        p=(i*5) % 29 + 1,
        adj=random.choice(adjectives),
        noun=random.choice(nouns),
        topic=random.choice(topics),
        topic2=random.choice(topics),
        lang=random.choice(langs),
        place=random.choice(places),
        food=random.choice(foods),
        q=(i % 6) + 2,
    )

random.seed(42)
with open("prompts.txt", "w") as f:
    for i in range(500):
        f.write(make(i) + "\n")
print("Generated prompts.txt")
