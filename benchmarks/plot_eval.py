from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_aggregate(metrics: dict, path: Path) -> None:
    labels = [
        "retrieval",
        "answer",
        "answer_strict",
        "grounded",
        "combined",
    ]
    values = [
        metrics["retrieval_pass_rate"],
        metrics["answer_pass_rate"],
        metrics["answer_strict_pass_rate"],
        metrics["answer_grounded_rate"],
        metrics["combined_pass_rate"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pass rate")
    ax.set_title("aggregate")
    ax.grid(alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_outcomes(metrics: dict, path: Path) -> None:
    n = int(metrics["n"])
    labels = ["retrieval", "answer", "combined"]
    rates = [
        metrics["retrieval_pass_rate"],
        metrics["answer_pass_rate"],
        metrics["combined_pass_rate"],
    ]
    passed = np.array([int(round(rate * n)) for rate in rates], dtype=np.int64)
    failed = n - passed
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(y, passed, label="pass", color="#59a14f")
    ax.barh(y, failed, left=passed, label="fail", color="#e15759")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("cases")
    ax.set_title("outcomes")
    ax.legend()
    ax.grid(alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
