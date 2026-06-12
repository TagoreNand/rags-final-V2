"""
backend/eval/golden.py

A small, self-contained evaluation set: a fixed in-repo corpus plus golden
questions whose ground-truth relevant documents are ids INTO that corpus.

Keeping the corpus in-repo makes retrieval evaluation fully reproducible (no
network, no live Wikipedia drift) so it can run as a deterministic CI gate.
Each GoldenItem also carries a reference answer and the key facts a faithful
answer must contain, used by the groundedness judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str  # canonical url, used as the relevance key
    title: str
    source: str  # wikipedia | arxiv
    text: str

    def to_meta(self) -> Dict[str, str]:
        return {"title": self.title, "url": self.doc_id, "source": self.source}


@dataclass(frozen=True)
class GoldenItem:
    qid: str
    question: str
    relevant: List[str]            # corpus doc_ids that SHOULD be retrieved
    reference_answer: str
    must_include: List[str] = field(default_factory=list)  # key grounded facts


# ── Fixed corpus (16 passages) ─────────────────────────────────────────────────
CORPUS: List[CorpusDoc] = [
    CorpusDoc("https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback",
              "Reinforcement learning from human feedback", "wikipedia",
              "Reinforcement learning from human feedback (RLHF) trains a reward model from "
              "human preference comparisons, then optimizes a language model policy with "
              "reinforcement learning, typically Proximal Policy Optimization, to align "
              "behaviour with human values."),
    CorpusDoc("https://arxiv.org/abs/2203.02155",
              "Training language models to follow instructions with human feedback", "arxiv",
              "InstructGPT fine-tunes GPT-3 on human demonstrations and then with reinforcement "
              "learning from human feedback. Human labelers rank outputs, a reward model learns "
              "the preferences, and PPO optimizes the policy. The 1.3B InstructGPT model is "
              "preferred by humans over the 175B GPT-3 model."),
    CorpusDoc("https://en.wikipedia.org/wiki/Reward_model",
              "Reward model", "wikipedia",
              "A reward model predicts human preference between candidate responses and outputs "
              "a scalar reward. In RLHF the reward signal updates the policy. Reward models are "
              "the main quality bottleneck and are prone to reward hacking."),
    CorpusDoc("https://arxiv.org/abs/1707.06347",
              "Proximal Policy Optimization Algorithms", "arxiv",
              "Proximal Policy Optimization (PPO) is a policy-gradient reinforcement learning "
              "algorithm that uses a clipped surrogate objective. In RLHF a KL-divergence "
              "penalty keeps the policy close to the supervised model to prevent reward "
              "over-optimization."),
    CorpusDoc("https://en.wikipedia.org/wiki/Large_language_model",
              "Large language model", "wikipedia",
              "A large language model is a neural network with billions of parameters trained on "
              "large text corpora to predict the next token. Instruction tuning and RLHF are "
              "post-training steps that turn a base model into a helpful assistant."),
    CorpusDoc("https://en.wikipedia.org/wiki/AI_alignment",
              "AI alignment", "wikipedia",
              "AI alignment research aims to steer AI systems toward their designers' intended "
              "goals and human values. RLHF is one of the most widely deployed alignment "
              "techniques for making language models helpful, honest, and harmless."),
    CorpusDoc("https://arxiv.org/abs/1706.03762",
              "Attention Is All You Need", "arxiv",
              "The Transformer is a sequence model based entirely on self-attention, dispensing "
              "with recurrence and convolutions. Multi-head scaled dot-product attention lets the "
              "model weigh all positions in parallel, enabling efficient training."),
    CorpusDoc("https://en.wikipedia.org/wiki/Attention_(machine_learning)",
              "Attention (machine learning)", "wikipedia",
              "Attention is a mechanism that computes a weighted sum of value vectors, where the "
              "weights come from the similarity between a query and key vectors. Self-attention "
              "relates different positions of a single sequence."),
    CorpusDoc("https://en.wikipedia.org/wiki/BERT_(language_model)",
              "BERT (language model)", "wikipedia",
              "BERT is a bidirectional Transformer encoder pre-trained with masked language "
              "modelling and next-sentence prediction. It produces contextual embeddings used for "
              "classification, question answering, and retrieval."),
    CorpusDoc("https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
              "Retrieval-augmented generation", "wikipedia",
              "Retrieval-augmented generation (RAG) combines a retriever that fetches relevant "
              "documents with a generator that conditions its output on them, grounding answers "
              "in external knowledge and reducing hallucination."),
    CorpusDoc("https://en.wikipedia.org/wiki/Okapi_BM25",
              "Okapi BM25", "wikipedia",
              "BM25 is a bag-of-words ranking function that scores documents by term frequency "
              "and inverse document frequency with length normalization. It is a strong lexical "
              "baseline for keyword search."),
    CorpusDoc("https://en.wikipedia.org/wiki/Word_embedding",
              "Word embedding", "wikipedia",
              "An embedding maps text to a dense vector so that semantically similar text lies "
              "close together. Cosine similarity between embeddings powers semantic search and "
              "dense retrieval."),
    CorpusDoc("https://en.wikipedia.org/wiki/Vector_database",
              "Vector database", "wikipedia",
              "A vector database stores embeddings and supports approximate nearest neighbour "
              "search using indexes such as HNSW, enabling fast similarity search over millions "
              "of vectors."),
    CorpusDoc("https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)",
              "Fine-tuning (deep learning)", "wikipedia",
              "Fine-tuning adapts a pre-trained model to a downstream task by continuing training "
              "on task-specific data. It is cheaper than training from scratch and transfers "
              "learned representations."),
    CorpusDoc("https://arxiv.org/abs/2106.09685",
              "LoRA: Low-Rank Adaptation of Large Language Models", "arxiv",
              "LoRA freezes the pre-trained weights and injects trainable low-rank matrices into "
              "each layer, drastically reducing the number of trainable parameters for "
              "fine-tuning large language models with no inference latency."),
    CorpusDoc("https://en.wikipedia.org/wiki/Knowledge_distillation",
              "Knowledge distillation", "wikipedia",
              "Knowledge distillation trains a small student model to mimic a larger teacher "
              "model's outputs, transferring capability into a cheaper model for deployment."),
]

CORPUS_BY_ID: Dict[str, CorpusDoc] = {d.doc_id: d for d in CORPUS}


# ── Golden questions ────────────────────────────────────────────────────────────
GOLDEN: List[GoldenItem] = [
    GoldenItem(
        "q1", "How does RLHF align large language models?",
        ["https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback",
         "https://en.wikipedia.org/wiki/AI_alignment",
         "https://en.wikipedia.org/wiki/Reward_model"],
        "RLHF trains a reward model from human preferences and uses RL (PPO) to align the "
        "policy with human values, a widely used alignment technique.",
        ["reward model", "human preference", "align"]),
    GoldenItem(
        "q2", "What reinforcement learning algorithm is used in RLHF and why?",
        ["https://arxiv.org/abs/1707.06347",
         "https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback"],
        "RLHF uses Proximal Policy Optimization with a KL penalty that keeps the policy close "
        "to the supervised model to prevent reward over-optimization.",
        ["Proximal Policy Optimization", "KL"]),
    GoldenItem(
        "q3", "What did InstructGPT demonstrate about model size and human preference?",
        ["https://arxiv.org/abs/2203.02155"],
        "A 1.3B InstructGPT model was preferred by humans over the 175B GPT-3 model, showing "
        "preference alignment can beat raw scale.",
        ["1.3B", "175B", "preferred"]),
    GoldenItem(
        "q4", "What is a reward model and what is its main failure mode?",
        ["https://en.wikipedia.org/wiki/Reward_model"],
        "A reward model predicts human preference as a scalar; its main failure mode is reward "
        "hacking, where the policy exploits flaws in the signal.",
        ["scalar", "reward hacking"]),
    GoldenItem(
        "q5", "What is the core idea of the Transformer architecture?",
        ["https://arxiv.org/abs/1706.03762",
         "https://en.wikipedia.org/wiki/Attention_(machine_learning)"],
        "The Transformer relies entirely on multi-head self-attention instead of recurrence, "
        "weighing all positions in parallel.",
        ["self-attention", "parallel"]),
    GoldenItem(
        "q6", "How does retrieval-augmented generation reduce hallucination?",
        ["https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
         "https://en.wikipedia.org/wiki/Word_embedding"],
        "RAG retrieves relevant documents and conditions generation on them, grounding answers "
        "in external knowledge and reducing hallucination.",
        ["retriev", "ground"]),
    GoldenItem(
        "q7", "How do vector databases make semantic search fast?",
        ["https://en.wikipedia.org/wiki/Vector_database",
         "https://en.wikipedia.org/wiki/Word_embedding"],
        "Vector databases store embeddings and use approximate nearest neighbour indexes such "
        "as HNSW to search millions of vectors quickly.",
        ["approximate nearest neighbour", "HNSW"]),
    GoldenItem(
        "q8", "What is BM25 and how does it rank documents?",
        ["https://en.wikipedia.org/wiki/Okapi_BM25"],
        "BM25 is a lexical ranking function scoring by term frequency and inverse document "
        "frequency with length normalization.",
        ["term frequency", "inverse document frequency"]),
    GoldenItem(
        "q9", "How does LoRA make fine-tuning large models efficient?",
        ["https://arxiv.org/abs/2106.09685",
         "https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)"],
        "LoRA freezes pre-trained weights and trains small low-rank matrices, cutting trainable "
        "parameters with no added inference latency.",
        ["low-rank", "freezes"]),
    GoldenItem(
        "q10", "What does knowledge distillation do?",
        ["https://en.wikipedia.org/wiki/Knowledge_distillation"],
        "Knowledge distillation trains a small student model to mimic a larger teacher, "
        "producing a cheaper deployable model.",
        ["student", "teacher"]),
]
