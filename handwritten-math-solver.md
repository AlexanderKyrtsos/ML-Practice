# Project 6: Handwritten Math Expression Solver

## Goal
Build a pipeline that:
1. Takes an image of a handwritten math expression
2. Recognizes individual symbols
3. Parses the 2D structure into a syntax tree
4. Evaluates or simplifies the expression
5. Outputs LaTeX and the solution

## The Hard Problem: 2D Structure

Math isn't linear like text. Consider:
```
  x²
  ──  + √(y³)
  2
```

You need to understand:
- Superscripts (exponents)
- Fraction bars (numerator above, denominator below)
- Square root symbols (operand inside)
- Spatial relationships

## Approach Options

### Option A: Two-Stage Pipeline (Recommended for Learning)
```
Stage 1: Symbol Detection + Recognition
  - Object detection to find bounding boxes
  - CNN classifier for each symbol

Stage 2: Structure Parsing
  - Rule-based or ML-based spatial relationship parser
  - Convert 2D layout to expression tree
```

### Option B: End-to-End Sequence Model
```
- Encoder: CNN on image → feature map
- Decoder: Attention-based LSTM → LaTeX sequence
- Trained end-to-end
- Harder to debug, needs more data
```

## Dataset: CROHME

**Competition on Recognition of Handwritten Mathematical Expressions**
- ~10,000 handwritten expressions with ground truth
- InkML format (stroke data) + rendered images
- Labels in LaTeX and symbol-level annotations

**Download:** Search "CROHME dataset TC-11"

### Data Augmentation (Critical)
- Rotation: ±15°
- Scaling: 0.8x - 1.2x
- Stroke thickness variation
- Elastic distortion
- Background noise

## Symbol Classes (~100 categories)

```
Digits:     0-9
Letters:    a-z, A-Z, Greek (α, β, θ, π, Σ, etc.)
Operators:  + - × ÷ = < > ≤ ≥ ±
Grouping:   ( ) [ ] { }
Special:    √ ∫ ∑ ∏ ∂ ∞ ° ′
Relations:  fraction bar, superscript, subscript
```

## Implementation Architecture

```
src/
├── detection/
│   ├── detector.py         # Symbol bounding box detection
│   ├── segmentation.py     # Connected component analysis
│   └── preprocessing.py    # Binarization, deskewing
├── recognition/
│   ├── cnn_classifier.py   # Symbol classification CNN
│   ├── augmentation.py     # Data augmentation
│   └── training.py         # Training loop
├── parsing/
│   ├── spatial.py          # Spatial relationship detection
│   ├── grammar.py          # Expression grammar rules
│   ├── tree_builder.py     # Build expression AST
│   └── latex_export.py     # AST → LaTeX string
├── solver/
│   ├── evaluator.py        # Numeric evaluation
│   ├── simplifier.py       # Symbolic simplification
│   └── sympy_bridge.py     # SymPy integration
└── cli/
    └── main.py             # CLI interface
```

## Symbol Recognition Model

### Architecture (CNN)
```
Input: 48x48 grayscale image

Conv2D(32, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
Conv2D(64, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
Conv2D(128, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
Conv2D(256, 3x3) → BatchNorm → ReLU → GlobalAvgPool
Dense(256) → Dropout(0.5) → Dense(num_classes)
```

### Training Details
- Optimizer: Adam, lr=0.001 with cosine decay
- Batch size: 64
- Epochs: 50-100
- Loss: Cross-entropy with label smoothing

## Spatial Relationship Parser

### Key Relationships
| Relationship | Detection Rule |
|--------------|----------------|
| Horizontal adjacent | Same baseline, x₂.left > x₁.right |
| Superscript | y₂.bottom < y₁.center, x overlap |
| Subscript | y₂.top > y₁.center, x overlap |
| Above (numerator) | y₂.bottom < fraction_bar.top |
| Below (denominator) | y₂.top > fraction_bar.bottom |
| Inside sqrt | x within sqrt symbol bounds |

### Parsing Algorithm
```
1. Sort symbols by x-coordinate (left to right)
2. Identify special structures (fraction bars, sqrt symbols)
3. Group symbols into regions (numerator, denominator, etc.)
4. Recursively parse each region
5. Build expression tree with operator precedence
```

## Expression Tree Example

For: `(x² + 3) / 2`

```
        [DIVIDE]
        /      \
    [ADD]      [2]
    /    \
 [POW]   [3]
 /    \
[x]   [2]
```

## CLI Interface

```bash
# Recognize and solve
mathsolve image equation.png

# Output:
# Detected expression: \frac{x^{2} + 3}{2}
# Rendered: (x² + 3) / 2
#
# If x = 5:  (25 + 3) / 2 = 14
# Simplified: 0.5x² + 1.5

# Batch processing
mathsolve image ./homework_photos/ --output solutions.json

# With variable substitution
mathsolve image eq.png --vars "x=3,y=4"

# Output format options
mathsolve image eq.png --format latex
mathsolve image eq.png --format mathml
mathsolve image eq.png --format sympy
```

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Similar symbols (1, l, I) | Use context + character n-grams |
| Touching characters | Train on augmented connected examples |
| Variable-size expressions | Multi-scale detection or sliding window |
| Ambiguous structure | Beam search with grammar constraints |
| Evaluation edge cases | Integrate SymPy for robust math |

## Tech Stack

- **Image Processing:** OpenCV, Pillow
- **Deep Learning:** PyTorch
- **Symbol Detection:** Faster R-CNN or YOLO (small variant)
- **Math Evaluation:** SymPy
- **CLI:** Click or Typer
- **Data Format:** CROHME InkML + rendered PNGs
