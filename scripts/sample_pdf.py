"""Generate a small synthetic PDF so the pipeline can be tested without real
course materials. Drop the output into data/raw/ and run ingest.
"""
import os
import sys

import fitz  # PyMuPDF

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "raw", "sample-lecture.pdf")

CONTENT = [
    ("Introduction to Regularization",
     "Regularization is a technique used to reduce overfitting in machine "
     "learning models. Overfitting occurs when a model learns the noise in the "
     "training data rather than the underlying pattern. Regularization adds a "
     "penalty to the loss function to discourage overly complex models."),
    ("L1 and L2 Regularization",
     "L1 regularization (Lasso) adds the sum of the absolute values of the "
     "coefficients to the loss, which encourages sparsity by driving some "
     "coefficients to zero. L2 regularization (Ridge) adds the sum of the "
     "squares of the coefficients, which shrinks coefficients toward zero "
     "without making them exactly zero. A hyperparameter lambda controls the "
     "strength of the penalty."),
    ("Gradient Descent Intuition",
     "Gradient descent iteratively updates model parameters by moving in the "
     "opposite direction of the gradient of the loss with respect to the "
     "parameters. The learning rate controls the size of each step. If the "
     "learning rate is too large, the model may oscillate or diverge."),
]


def build():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = fitz.open()
    for title, body in CONTENT:
        page = doc.new_page()
        page.insert_textbox((72, 72, 540, 300), f"{title}\n\n{body}", fontsize=14, fontname="helv")
    doc.save(OUT)
    doc.close()
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
