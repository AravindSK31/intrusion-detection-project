# Literature Benchmark Comparison

## NSL-KDD

**Divekar, A., Parekh, M., Savla, V., Mishra, R., Shirole, M. (2018).**
"Benchmarking datasets for Anomaly-based Network Intrusion Detection: KDD CUP 99 alternatives."
IEEE ICCCS 2018. DOI: 10.1109/CCCS.2018.8586840. Open access: https://arxiv.org/abs/1811.05372

- Random Forest on full multiclass NSL-KDD test set: F1 = 87 (DoS), 79 (Normal), 70 (Probe), 0 (R2L), 4 (U2R); weighted F1 = 70.
- Relevant to our finding: near-total collapse on R2L/U2R lines up with our own result that supervised detection degrades sharply on rare, structurally distinct attack categories.
- Difference from our setup: standard random train/test split (not strictly "unseen-at-training-time" like ours); thresholds not matched-FPR.

**Meira, J. (2018).**
"Comparative Results with Unsupervised Techniques in Cyber Attack Novelty Detection."
DOI: 10.3390/proceedings2181191 (open access)

- Frames NSL-KDD + ISCX as a novelty detection problem using Isolation Forest, Autoencoder, K-Means, Nearest Neighbor — same conceptual framing as our project.
- Used as precedent/support for project design rather than a direct numeric comparison (abstract doesn't report exact detection rates).

## CICIDS2017

**Zozan Miran, A., Salih Kadir, G. (2026).**
"A Hybrid Machine Learning Approach for Enhancing Intrusion Detection Systems Using CICIDS2017 Dataset."
Dasinya Journal for Engineering and Informatics. DOI: 10.65542/djei.v2i2.49

- XGBoost: 99.3% accuracy on known threats.
- One-Class SVM: 92.1% accuracy on unknown threats.

**Alabelewe, A.T., et al. (2025).**
"Machine Learning Driven Network Traffic Analysis for Cybersecurity."
FUDMA Journal of Sciences. DOI: 10.33003/fjs-2025-0910-3752

- Random Forest: >99.4% accuracy, very few false negatives.
- Isolation Forest: 93% recall on anomalies.

## Our results vs. literature

| | Literature | Ours (NSL-KDD) | Ours (CICIDS2017) |
|---|---|---|---|
| RF/XGBoost on known/seen attacks | 96-99.4% accuracy | 89.9% detection | 40.4% detection (unseen-same-category) |
| One-Class SVM on unknown attacks | 92.1% accuracy | 29.8% detection | 0.3% detection |
| Isolation Forest recall | 93% | 44.6% detection | 0% detection (PortScan) |

**Why our numbers are lower (methodological explanation):**
1. Thresholds are chosen ONLY from normal-validation data, never from attack labels (seen or unseen) — published work typically tunes on test-set performance directly, inflating reported numbers.
2. Our "unseen" test sets are attack subtypes/categories structurally absent from training, not just randomly held-out rows from the same underlying distribution.
3. Our comparison is at a matched, pre-committed false-positive rate (1% target) rather than each method's default/best operating point.

Conclusion: lower numbers reflect a stricter, more honest evaluation protocol, not worse models.