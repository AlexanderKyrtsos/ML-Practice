# Project 5: Code Complexity Predictor

## Goal
Build a CLI tool that analyzes source code files and predicts:
- Bug likelihood (probability this code will have bugs)
- Maintainability score (how hard to modify)
- Estimated review time (how long a code review might take)

## Data Collection Strategy

### Option A: Mine GitHub (Recommended)
```
1. Clone repos with good issue tracking
2. Use git blame to link code regions to commits
3. Link commits to bug-fix issues (look for "fix", "bug", "issue #" in messages)
4. Label: code that was later fixed = buggy, code untouched for 2+ years = stable
```

### Option B: Synthetic Labels
```
1. Use existing static analysis tools (pylint, SonarQube scores)
2. Treat their output as ground truth labels
3. Train your model to approximate these tools (then extend beyond them)
```

**Suggested Scale:** 50,000 - 200,000 functions/methods for training

## Feature Engineering (The Core Challenge)

### AST-Based Features
| Feature | Description |
|---------|-------------|
| `depth` | Max nesting depth of control structures |
| `num_branches` | Count of if/elif/else/switch |
| `num_loops` | Count of for/while loops |
| `num_params` | Function parameter count |
| `loc` | Lines of code |
| `sloc` | Source lines (excluding blanks/comments) |
| `num_calls` | Function/method calls made |
| `num_variables` | Distinct variable names |
| `num_literals` | Hardcoded values |
| `cyclomatic` | Cyclomatic complexity (edges - nodes + 2) |
| `cognitive` | Cognitive complexity (SonarSource's metric) |
| `halstead_*` | Halstead metrics (volume, difficulty, effort) |

### Identifier-Based Features
| Feature | Description |
|---------|-------------|
| `avg_name_length` | Average variable/function name length |
| `name_entropy` | Entropy of identifier characters |
| `abbrev_ratio` | Ratio of short names (<3 chars) |
| `naming_consistency` | snake_case vs camelCase mixing |

### Structural Features
| Feature | Description |
|---------|-------------|
| `has_nested_try` | Try blocks inside try blocks |
| `return_count` | Number of return statements |
| `early_return_ratio` | Returns in first half vs second half |
| `comment_ratio` | Comment lines / total lines |
| `blank_ratio` | Blank lines / total lines |

## Implementation Architecture

```
src/
├── parsers/
│   ├── python_parser.py    # AST parsing for Python
│   ├── javascript_parser.py
│   └── base.py             # Abstract parser interface
├── features/
│   ├── extractor.py        # Coordinates feature extraction
│   ├── ast_features.py     # AST-based metrics
│   ├── halstead.py         # Halstead complexity
│   └── cognitive.py        # Cognitive complexity
├── models/
│   ├── trainer.py          # Training pipeline
│   ├── predictor.py        # Inference
│   └── calibration.py      # Probability calibration
├── cli/
│   ├── analyze.py          # Analyze files/directories
│   ├── train.py            # Train on labeled data
│   └── hook.py             # Git pre-commit hook
└── data/
    ├── collector.py        # GitHub mining
    └── labeler.py          # Heuristic labeling
```

## Model Selection

### Recommended: Gradient Boosted Trees
- XGBoost or LightGBM
- Handles tabular features excellently
- Fast training on CPU
- Built-in feature importance (interpretability)
- Handles missing values gracefully

### Training Pipeline
```
1. Extract features from all functions → DataFrame
2. Split: 80% train, 10% val, 10% test
3. Train XGBoost with early stopping on validation
4. Calibrate probabilities with isotonic regression
5. Evaluate: AUC-ROC for bug prediction, MAE for review time
```

### Hyperparameters to Tune
- `max_depth`: 4-8
- `n_estimators`: 100-500
- `learning_rate`: 0.01-0.1
- `min_child_weight`: 1-10
- `subsample`: 0.7-1.0

## CLI Interface Design

```bash
# Analyze a single file
codepredictor analyze src/parser.py

# Output:
# src/parser.py
#   parse_expression (line 45-89)
#     Bug Probability:  0.73 [HIGH]
#     Maintainability:  34/100 [POOR]
#     Est. Review Time: 12 min
#     Top Risk Factors:
#       - Cyclomatic complexity: 23 (threshold: 10)
#       - Nesting depth: 6 (threshold: 4)
#       - No early returns in long function

# Analyze entire project
codepredictor analyze ./src --format json > report.json

# Git hook mode (for pre-commit)
codepredictor hook --staged --fail-threshold 0.8

# Train on labeled data
codepredictor train --data labeled_functions.parquet --output model.pkl
```

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Class imbalance (few bugs) | SMOTE oversampling, or adjust class weights |
| Language differences | Train separate models per language, or use language-agnostic features only |
| Ground truth noise | Use soft labels, aggregate multiple signals |
| Feature correlation | Let tree models handle it (they're robust) |

## Tech Stack

- **Parsing:** `ast` (Python), `tree-sitter` (multi-language)
- **Features:** Custom extraction code
- **Model:** XGBoost or LightGBM
- **CLI:** Click or Typer
- **Data Storage:** Parquet files via pandas/polars
