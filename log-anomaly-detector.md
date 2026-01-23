# Project 7: Log File Anomaly Detector

## Goal
Build a system that:
1. Learns "normal" patterns from historical logs
2. Flags anomalous log entries in real-time or batch mode
3. Groups related anomalies into incidents
4. Provides explanations for why something is anomalous

## Why Unsupervised?
- Labeled log anomalies are rare
- "Normal" changes over time (new features, deployments)
- You want to catch unknown-unknowns, not just known patterns

---

## Implementation Timeline (3 Days)

### Day 1: Parsing + Feature Extraction + Basic Model
- Set up project structure
- Implement Drain3 log parsing
- Build per-line feature extraction
- Train basic Isolation Forest
- Simple CLI for train/detect

### Day 2: Advanced Features + LSTM Autoencoder
- Implement sliding window features
- Build LSTM autoencoder model
- Training pipeline for autoencoder
- Ensemble scoring (IF + LSTM)
- Threshold tuning on validation data

### Day 3: Explanations + Grouping + Polish
- Anomaly explanation system
- Incident grouping by time proximity
- Watch mode (tail -f style monitoring)
- Testing on public datasets
- CLI polish and error handling

---

## Log Processing Pipeline

```
Raw Logs → Parsing → Normalization → Embedding → Model → Anomaly Score
```

---

## Step 1: Log Parsing with Drain3

Logs have structure, but it's implicit:
```
# Raw logs
2024-01-15 10:23:45 INFO Connected to database at 192.168.1.1:5432
2024-01-15 10:23:46 INFO Connected to database at 192.168.1.2:5432
2024-01-15 10:23:47 ERROR Connection timeout after 30s to 192.168.1.3:5432

# Parsed templates
Template 1: "Connected to database at <IP>:<PORT>"
Template 2: "Connection timeout after <NUM>s to <IP>:<PORT>"
```

**Drain3** automatically discovers these templates:
```python
from drain3 import TemplateMiner

miner = TemplateMiner()
for line in log_lines:
    result = miner.add_log_message(line)
    # result.cluster_id = template ID
    # result.template_mined = "Connected to database at <*>:<*>"
```

---

## Step 2: Feature Engineering

### Per-Line Features
| Feature | Description |
|---------|-------------|
| `template_id` | Which log template this matches |
| `level` | INFO/WARN/ERROR/DEBUG encoded |
| `hour_of_day` | 0-23 |
| `day_of_week` | 0-6 |
| `msg_length` | Character count |
| `num_params` | Count of <*> placeholders filled |
| `param_values` | Encoded parameter values |

### Time-Window Features (Critical)
| Feature | Description |
|---------|-------------|
| `template_freq` | How often this template appeared in last N minutes |
| `template_burst` | Sudden increase in frequency |
| `level_ratio` | ERROR/WARN ratio in window |
| `unique_templates` | Number of distinct templates in window |
| `new_template` | First time seeing this template? |
| `sequence_probability` | P(this template \| previous templates) |

---

## Step 3: Models

### Model A: Isolation Forest (Primary)
- Works on feature vectors
- Fast training, no hyperparameter tuning needed
- Good baseline

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.01, random_state=42)
model.fit(normal_log_features)
scores = model.decision_function(new_log_features)
# More negative = more anomalous
```

### Model B: LSTM Autoencoder (Secondary)
- Learns normal sequence patterns
- Anomaly = high reconstruction error
- Catches order-dependent anomalies

```
Architecture:
Input sequence (20 log events) →
LSTM(64) → LSTM(32) → [latent space] → LSTM(32) → LSTM(64) →
Output sequence (reconstruct input)

Loss: MSE between input and output sequences
Anomaly score: Reconstruction error
```

**PyTorch Implementation:**
```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super().__init__()
        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
        # Decoder
        self.decoder_lstm1 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        # Encode
        x, _ = self.encoder_lstm1(x)
        x, _ = self.encoder_lstm2(x)
        # Decode
        x, _ = self.decoder_lstm1(x)
        x, _ = self.decoder_lstm2(x)
        return x
```

### Ensemble Scoring
```python
def ensemble_score(log_features, log_sequence):
    # Isolation Forest score (normalize to 0-1)
    if_score = isolation_forest.decision_function([log_features])[0]
    if_score_normalized = (if_score - if_min) / (if_max - if_min)

    # LSTM reconstruction error (normalize to 0-1)
    reconstruction = lstm_autoencoder(log_sequence)
    lstm_error = F.mse_loss(reconstruction, log_sequence).item()
    lstm_score_normalized = (lstm_error - lstm_min) / (lstm_max - lstm_min)

    # Weighted combination
    final_score = 0.6 * if_score_normalized + 0.4 * lstm_score_normalized
    return final_score
```

---

## Implementation Architecture

```
src/
├── parsing/
│   ├── drain_parser.py     # Drain3 wrapper
│   ├── regex_parser.py     # Fallback regex patterns
│   └── timestamp.py        # Timestamp normalization
├── features/
│   ├── line_features.py    # Per-line feature extraction
│   ├── window_features.py  # Sliding window aggregations
│   └── sequences.py        # Sequence preparation for LSTM
├── models/
│   ├── isolation_forest.py # IF wrapper
│   ├── autoencoder.py      # LSTM autoencoder
│   ├── ensemble.py         # Combine multiple models
│   └── threshold.py        # Dynamic threshold tuning
├── detection/
│   ├── scorer.py           # Anomaly scoring
│   ├── grouper.py          # Group anomalies into incidents
│   └── explainer.py        # Why is this anomalous?
├── cli/
│   ├── train.py            # Train on normal logs
│   ├── detect.py           # Run detection
│   └── watch.py            # Tail mode (continuous)
└── data/
    └── loghub/             # Public datasets for testing
```

---

## Training Strategy

```
1. Collect 1-7 days of "normal" operation logs
2. Parse all logs → extract templates
3. Build feature vectors for each log line
4. Create sequences for LSTM (sliding window of 20 events)
5. Train Isolation Forest on feature vectors
6. Train LSTM autoencoder on sequences
7. Determine thresholds on validation set (target 1% false positive rate)
8. Save models + template miner state
```

---

## Anomaly Explanation System

Don't just say "anomalous" - explain why:

```
ANOMALY DETECTED [Score: 0.89]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Timestamp: 2024-01-15 03:42:18
Log Line:  "Database connection pool exhausted, 0 available"

Anomaly Factors:
  • NEW TEMPLATE: Never seen before in training data
  • UNUSUAL TIME: This log level rare at 3 AM (normal: 9 AM - 6 PM)
  • BURST DETECTED: 47 similar errors in last 2 minutes (normal: <5)
  • SEQUENCE ANOMALY: High LSTM reconstruction error
      Normal: [auth_success] → [db_query]
      Actual: [auth_success] → [db_pool_exhausted] → [db_pool_exhausted]

Related Anomalies (same incident):
  • 03:42:15 - "Connection timeout to replica-2"
  • 03:42:16 - "Failover initiated to replica-3"
  • 03:42:17 - "Replica-3 not responding"
```

### Explanation Implementation
```python
def explain_anomaly(log_line, features, thresholds, lstm_contrib):
    explanations = []

    if features['is_new_template']:
        explanations.append("NEW TEMPLATE: Never seen in training data")

    if features['hour_of_day'] not in normal_hours:
        explanations.append(f"UNUSUAL TIME: Rare at {features['hour_of_day']}:00")

    if features['template_freq'] > thresholds['burst']:
        explanations.append(f"BURST DETECTED: {features['template_freq']} in window")

    if lstm_contrib > thresholds['sequence']:
        explanations.append("SEQUENCE ANOMALY: Unusual pattern of events")

    return explanations
```

---

## Incident Grouping

Group anomalies that occur close together in time:

```python
def group_into_incidents(anomalies, time_threshold_seconds=300):
    """Group anomalies within 5 minutes of each other."""
    if not anomalies:
        return []

    incidents = []
    current_incident = [anomalies[0]]

    for anomaly in anomalies[1:]:
        time_diff = anomaly.timestamp - current_incident[-1].timestamp
        if time_diff.total_seconds() <= time_threshold_seconds:
            current_incident.append(anomaly)
        else:
            incidents.append(Incident(current_incident))
            current_incident = [anomaly]

    incidents.append(Incident(current_incident))
    return incidents
```

---

## CLI Interface

```bash
# Train on normal logs
logdetect train --input /var/log/app/*.log --days 7 --output model/

# Options:
#   --contamination 0.01    Expected anomaly rate for IF
#   --sequence-length 20    LSTM sequence window size
#   --epochs 50             LSTM training epochs

# Detect anomalies in new logs
logdetect scan --model model/ --input today.log

# Output:
# Scanned 145,892 log lines
# Found 23 anomalies in 3 incident groups
#
# Incident 1 (03:42:15 - 03:47:22): Database connectivity [CRITICAL]
#   └─ 15 anomalous lines, started with "Connection timeout..."
#
# Incident 2 (14:22:01): Unknown template [LOW]
#   └─ 1 anomalous line: "New feature X enabled for user Y"

# Watch mode (like tail -f but with anomaly detection)
logdetect watch --model model/ --input /var/log/app/current.log

# Tune sensitivity
logdetect scan --model model/ --input today.log --sensitivity high

# Export results
logdetect scan --model model/ --input today.log --format json > anomalies.json
```

---

## Public Datasets for Development

| Dataset | Description | Size |
|---------|-------------|------|
| HDFS | Hadoop logs with labeled anomalies | 11M lines |
| BGL | BlueGene/L supercomputer | 4.7M lines |
| Thunderbird | Supercomputer logs | 211M lines |
| Windows | Windows event logs | 114K lines |
| Apache | Web server access/error logs | 56K lines |

**Download from:** Loghub (GitHub: logpai/loghub)

### Recommended Starting Dataset: HDFS
- Has labeled anomalies for evaluation
- Manageable size
- Well-documented

---

## Threshold Tuning

```python
def tune_threshold(scores, target_fpr=0.01):
    """Find threshold that gives target false positive rate on normal data."""
    sorted_scores = np.sort(scores)
    threshold_idx = int(len(sorted_scores) * target_fpr)
    return sorted_scores[threshold_idx]

# Dynamic thresholds by time of day
def get_dynamic_threshold(hour, base_threshold, time_multipliers):
    """Adjust threshold based on time - be more lenient during deployments."""
    return base_threshold * time_multipliers.get(hour, 1.0)
```

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Concept drift (normal changes) | Periodic retraining, sliding window baseline |
| Too many alerts | Anomaly grouping, severity ranking, adaptive thresholds |
| Template explosion | Hierarchical clustering of similar templates |
| Multi-line logs (stack traces) | Regex-based stitching before parsing |
| Class imbalance in evaluation | Use precision@k, not accuracy |
| LSTM training time on CPU | Use smaller hidden dimensions, shorter sequences |

---

## Tech Stack

- **Log Parsing:** drain3
- **Feature Engineering:** pandas, numpy
- **Isolation Forest:** scikit-learn
- **LSTM Autoencoder:** PyTorch
- **CLI:** Click or Typer
- **Configuration:** PyYAML or TOML
- **Serialization:** joblib (sklearn), torch.save (PyTorch)

---

## Dependencies

```
# requirements.txt
drain3>=0.9.11
scikit-learn>=1.3.0
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
click>=8.1.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

## Evaluation Metrics

Since you're testing on HDFS (which has labels):

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# At various thresholds
for threshold in [0.5, 0.7, 0.9, 0.95]:
    predictions = (scores > threshold).astype(int)
    print(f"Threshold {threshold}:")
    print(f"  Precision: {precision_score(labels, predictions):.3f}")
    print(f"  Recall: {recall_score(labels, predictions):.3f}")
    print(f"  F1: {f1_score(labels, predictions):.3f}")
```

For unlabeled real-world logs, focus on:
- False positive rate (manual review of flagged items)
- Alert fatigue (are incidents grouped well?)
- Detection latency (how fast in watch mode?)
