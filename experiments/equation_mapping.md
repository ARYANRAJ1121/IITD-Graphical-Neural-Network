# Mathematical equation mapping (MINGLE / HypEHR)

**FROZEN** 2026-08-25 after the loss-function provenance correction. This is the Vanilla BCE mathematical specification. Do not change architecture, dimensions, or objectives without a new audit.

This document provides the exact mathematical reproduction of the HypEHR attention mechanism (as utilized by MINGLE). It ensures mathematical fidelity to the Pooling by Multihead Attention (PMA) formulation while explicitly categorizing every architectural component by its provenance (Exact Paper Equation, Inferred, or Our Choice).

## 1. Provenance & Implementation Audit Table

| Component | Paper equation/text | Verified implementation | Our implementation | Provenance |
| :--- | :--- | :--- | :--- | :--- |
| **Node Update Rule** | $X_v^{(l)} = f_{E \rightarrow V}(\mathcal{E}_{v, E^{(l-1)}})$ | MultiHead(S) Pooling | `MultiHead(S)` with learned query | **EXACT** |
| **Hyperedge Update** | $E_e^{(l)} = f_{V \rightarrow E}(\mathcal{V}_{e, X^{(l-1)}})$ | MultiHead(S) Pooling | `MultiHead(S)` with learned query | **EXACT** |
| **Semantic Infusion** | $E_e^{(l)} = \text{MLP}_2([ f_{V \rightarrow E}(\cdot) ; H_e ])$ | Concat pooled structural node set with semantic $H_e$ | PyTorch `torch.cat` + `nn.Sequential` | **EXACT** |
| **Attention Query** | "leverages multi-head self-attention" (HypEHR/Set Transformer) | Learned parametric query vector $W_i^Q \in \mathbb{R}^{1 \times (d/h)}$ | Learned $W_i^Q$ `nn.Parameter` | **INFERRED** (from Set Transformer PMA) |
| **PMA Output** | Converts set $S$ into fixed vector | Output dimension is $1 \times d$ | Matrix multiplication outputting single vector per target | **EXACT** |
| **PairNorm** | Normalizes pairwise distance to prevent oversmoothing (HypEHR) | Center and scale by Total Pairwise Squared Distance | Implement PairNorm instead of LayerNorm | **INFERRED** (from HypEHR framework) |
| **Residual & FFN** | Used in standard Transformer/SetTransformer blocks | $Z = M + \text{FFN}(M)$ applied after pooling | Applied directly after `MultiHead` pooling | **INFERRED** (from HypEHR framework) |
| **Jumping Knowledge**| "stacks embeddings from different transformer layers" (HypEHR) | Concatenation across all layers | $E_e^{final} = [E_e^{(1)} \parallel \dots \parallel E_e^{(L)}]$ | **VERIFIED IN TEXT** |
| **Prediction Head** | "MLP_CLS classification is used to convert hyperedge embeddings" | Projects stacked embeddings to classes | Linear layer projecting to 25 target conditions | **EXACT** |
| **Loss Function** | MINGLE Eq. (4) binary cross-entropy | Binary Cross Entropy | `BCEWithLogitsLoss` (numerically equivalent to Eq. (4)) | **EXACT — MINGLE Eq. (4)** |

---

## 2. Mathematical Definition of `MultiHead(S)`

The core message-passing relies on a Set-Transformer-based Pooling by Multihead Attention (PMA). 
Given a set of feature vectors $S \in \mathbb{R}^{|S| \times d}$, where $d=48$ and we use $h=4$ heads. The dimension per head is $d_k = d/h = 12$.

For each head $i \in \{1 \dots h\}$, we define a query vector and two projection matrices:
* **Query**: $W_i^Q \in \mathbb{R}^{1 \times d_k}$ (A learned parameter vector, independent of $S$)
* **Key Projection**: $W_i^K \in \mathbb{R}^{d \times d_k}$
* **Value Projection**: $W_i^V \in \mathbb{R}^{d \times d_k}$

**Mechanism:**
1. **Key Generation**: $K_i = S W_i^K \in \mathbb{R}^{|S| \times d_k}$
2. **Value Generation**: $V_i = S W_i^V \in \mathbb{R}^{|S| \times d_k}$
3. **Attention Scores**: $\alpha_i = \text{Softmax}\left( \frac{W_i^Q K_i^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{1 \times |S|}$
4. **Head Output**: $H_i = \alpha_i V_i \in \mathbb{R}^{1 \times d_k}$
5. **Concatenation**: $H_{multi} = [H_1 \parallel \dots \parallel H_h] \in \mathbb{R}^{1 \times d}$
6. **Output Projection**: $M_S = H_{multi} W_O \in \mathbb{R}^{1 \times d}$ (where $W_O \in \mathbb{R}^{d \times d}$)

---

## 3. $f_{V \rightarrow E}$ (Node $\rightarrow$ Hyperedge)

For a given hyperedge $e$, its input set is $S = \mathcal{V}_{e, X^{(l-1)}}$ (the $X$ embeddings of all nodes contained in $e$).
1. **MultiHead Pooling**: $M_e = \text{MultiHead}(\mathcal{V}_{e, X^{(l-1)}}) \in \mathbb{R}^{1 \times d}$
2. **FFN & Residual**: 
   $Z_e = M_e + \text{FFN}(M_e)$
3. **PairNorm**:
   $A_e = \text{PairNorm}(Z_e)$
   *(Shape: $A \in \mathbb{R}^{144510 \times 48}$)*

MINGLE then infuses semantics: $E_e^{(l)} = \text{MLP}_2([A_e ; H_e])$

---

## 4. $f_{E \rightarrow V}$ (Hyperedge $\rightarrow$ Node)

For a given node $v$, its input set is $S = \mathcal{E}_{v, E^{(l)}}$ (the $E$ embeddings of all hyperedges containing $v$).
1. **MultiHead Pooling**: $M_v = \text{MultiHead}(\mathcal{E}_{v, E^{(l)}}) \in \mathbb{R}^{1 \times d}$
2. **FFN & Residual**: $Z_v = M_v + \text{FFN}(M_v)$
3. **PairNorm**: $X_v^{(l)} = \text{PairNorm}(Z_v)$
   *(Shape: $X^{(l)} \in \mathbb{R}^{564 \times 48}$)*

---

## 5. PairNorm Implementation

Given a matrix of embeddings $X \in \mathbb{R}^{N \times d}$:
1. **Center**: $X_c = X - \frac{1}{N} \sum_{i=1}^N X_i$
2. **Compute Scale**: $s = \frac{1}{N} \sum_{i=1}^N ||X_{c,i}||_2^2$
3. **Normalize**: $X_{norm} = \frac{X_c}{\sqrt{s + \epsilon}}$

---

## 6. Jumping Knowledge (JK-Concat) Aggregation

To capture both local and deeper structural information, HypEHR utilizes Jumping Knowledge by stacking (concatenating) the embeddings from all transformer layers before classification.
$$E_e^{final} = [E_e^{(1)} \parallel \dots \parallel E_e^{(L)}]$$
* For $L=2$, $E_e^{final}$ has shape `(143946, 96)`.
* $\text{MLP}_{CLS}$ projects from $96 \rightarrow 25$ target classes.
