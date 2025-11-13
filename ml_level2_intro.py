"""
ml_level2_intro.py

Quick overview script for the
"ML Classics in Python - Level 2 (Google Colab)" repository.

This script:
- Prints a short description of the project
- Lists the main parts (advanced regression, classification, model comparison)
- Shows which boosting libraries are used
"""

from textwrap import indent

# ---------------------------------------------------------------------------
# 1. High-level description
# ---------------------------------------------------------------------------


def print_header() -> None:
    """Print a short introduction to the repository."""
    print("🚀 ML Classics in Python - Level 2 (Google Colab)")
    print(
        "Advanced machine learning models implemented in Python using "
        "XGBoost, LightGBM and CatBoost."
    )
    print(
        "The project focuses on advanced regression, classification and "
        "model comparison workflows.\n"
    )


# ---------------------------------------------------------------------------
# 2. Structure of the project
# ---------------------------------------------------------------------------

LEVEL2_PARTS = [
    {
        "part": "Part 1 - Advanced Regression",
        "examples": "CatBoost Regressor, LightGBM Regressor, XGBoost Regressor",
    },
    {
        "part": "Part 2 - Advanced Classification",
        "examples": "CatBoost Classifier, LightGBM Classifier, XGBoost Classifier",
    },
    {
        "part": "Part 3 - Model Comparison",
        "examples": "Side-by-side comparison of boosting models "
        "for regression and classification tasks",
    },
]


def print_structure() -> None:
    """Print a compact overview of the main parts."""
    print("📂 Project structure (high level):\n")
    for block in LEVEL2_PARTS:
        print(f"• {block['part']}")
        print(indent(f"- Examples: {block['examples']}\n", prefix="  "))


# ---------------------------------------------------------------------------
# 3. Boosting libraries and installation hints
# ---------------------------------------------------------------------------

BOOSTING_PACKAGES = [
    "xgboost",
    "lightgbm",
    "catboost",
]


def print_dependencies() -> None:
    """Show required boosting packages used in this repository."""
    print("🔧 Core boosting libraries:\n")
    for pkg in BOOSTING_PACKAGES:
        print(f"- {pkg}")
    print(
        "\nHint: in Google Colab you can install all dependencies via:\n"
        "!pip install numpy pandas matplotlib seaborn scikit-learn "
        "xgboost lightgbm catboost\n"
    )


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all info sections for a quick project overview."""
    print_header()
    print_structure()
    print_dependencies()


if __name__ == "__main__":
    main()
