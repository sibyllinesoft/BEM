# BEM System Vision and Architecture

> This document presents the conceptual foundation and strategic vision for the Bolt-on Expert Module (BEM) system, a novel approach to dynamic neural network adaptation.

## Concept: Bolt-on Expert Module (BEM)

A BEM is a modular, token-conditioned adapter that can be attached to proven base models to add style, skills, and domain knowledge without forking the foundation. The system prioritizes:

1. **Drop-in modularity** - Easy integration with existing models
2. **High information density** beyond static LoRA
3. **Stable deployment** on cached attention
4. **Multi-model compatibility** - One expert, many bases

### Core Components

A BEM package consists of three key pieces:

1. **Controller**: Reads local hidden states plus optional retrieval/tool/multimodal signals; outputs a compact *code* per layer or per chunk of tokens.
2. **Generator**: Turns that code into a low-rank weight delta for target layers.
3. **Shim**: A model-family adapter that maps a canonical latent to specific layer shapes across different base models.

This makes the adapter *context-dependent*. Instead of a single rank-r delta per layer, you get a *field* of deltas that changes with the problem, style, or evidence shown to the model.

## Why BEM Encodes More Information than Standard LoRA

A static LoRA at layer ℓ carries ~r(d_in+d_out) parameters. A BEM adds:

- A **codebook or basis** (size K, rank r) + a **controller** generating codes c_{ℓ,t} ∈ ℝ^r
- Effective capacity grows as K *and* with the entropy of c_{ℓ,t} across tokens/spans
- The number of distinct realizable deltas is combinatorial in sequence length
- You trade a small runtime cost for a huge representational jump

## Mathematical Formulation

Let W_ℓ be a frozen base weight. The BEM injects:

### Banked Mixture (MoL)
```
ΔW_{ℓ,t} = Σ_{k∈Top-k} π_{ℓ,t,k} A_{ℓ,k} B_{ℓ,k}^T
```

### Factorized Hyper (Gen)
```
ΔW_{ℓ,t} = U_ℓ Diag(c_{ℓ,t}) V_ℓ^T
```

Here c_{ℓ,t} is produced by the controller from hidden state h_{ℓ,t} plus optional side signals (retrieved doc embeddings, tool outputs, style tokens, vision features).

## Implementation Interfaces

**Attach points:** Start with MLP up/down and attention output W_o. Later, gate Q/K/V chunkwise.
**Scope:** per-sequence (cheap), per-chunk (every N tokens), or per-token (max control).
**Composition:** Allow multiple BEMs to mount simultaneously; compose deltas with norm budgets and orthogonality constraints.

## Controller Design

The controller processes:
- Local hidden h_{ℓ,t}
- **Prefix summary** (mean/attention pooled over first 128 tokens)
- **Side signals**:
  - Retrieval embeddings from a domain memory
  - Explicit style/task tokens
  - Tool outputs or environment state (for agents)
  - Optional multimodal embeddings (projected)

Features:
- Small MLP with LayerNorm + residual
- Entropy regularizer to keep mixtures crisp
- Temporal smoothing (EMA) to stabilize decisions

## Multi-Model Support

Each BEM maintains a *canonical latent* z_t ∈ ℝ^r. For each supported model family m and layer ℓ, learn **per-family bases** U_ℓ^(m), V_ℓ^(m). The controller is shared; only the projection bases differ.

- Train across families with *weight-tying in latent space*
- Alignment loss to keep z_t distribution stable across models
- Ship one BEM that declares `supports=[llama3-*, mistral-*, qwen2-*]`

## Training Strategy

### Data Requirements
- Base instruction-following mix + domain pack(s) + style pack(s)
- If you want "expert that knows X," include retrieval over X so the expert learns to *steer into* evidence, not memorize it all

### Loss Functions
- Standard LM loss on next token
- **Behavior preservation**: KL to the frozen base on a calibration set to avoid personality drift
- **Delta budget**: λ∥ΔW∥_F^2 (layerwise), spectral clamp
- **Routing regularizers**: entropy (encourage low entropy for banked, moderate for generated), utilization penalty (avoid collapsed experts)
- **Orthogonality across experts**: ∥A_i^T A_j∥_F^2 + ∥B_i B_j^T∥_F^2 for i≠j
- Optional **distillation**: small teacher (static LoRA or previous BEM) to stabilize early training

### Curriculum
1. **Sequence-level codes only** (one code per prompt). Locks in coarse style/domain.
2. **Chunkwise for attention, token-level for MLPs.** Turn on EMA smoothing and top-k sparsity.
3. **Retrieval-aware**: add side embeddings; train the controller to flip experts when retrieved content changes.

## Deployment Mechanics

- **Hot-swap**: load/unload BEMs at runtime; they register hooks on specified layers
- **Composition**: sum multiple BEM deltas but enforce a **norm budget** per layer
- **Fallback**: if routing confidence < τ, fade to static LoRA (or base)
- **Quantization**: base in INT4/8; keep BEM in bf16
- **Telemetry**: track gate entropy, expert utilization by span type, base-divergence KL, and delta norms

## Expert Enhancement: Micro-Retriever

Add a **micro-retriever** to the BEM package:
- A tiny in-package index (e.g., 50k–2M text chunks, FAISS IVF/HNSW) with frozen encoder
- On prompt, retrieve k chunks → pool to a side embedding → feed the controller
- The BEM learns *how much* to trust retrieval per situation
- Offloads raw knowledge into the index while the adapter learns the *policy* to use it

## Evaluation Framework

### Automated Testing
- **No-regression harness**: instruction following, safety, long-context, and agentic subtasks vs frozen base
- **Expert value**: domain QA, style fidelity, latency/throughput hit (<15% target)
- **Ablations**: static LoRA vs seq-level BEM vs chunkwise vs token-level; banked vs generated; with/without retrieval
- **Interference test**: run two BEMs together; verify budgeted composition and orthogonality prevent collapse

## Strategic Recommendations

### Phase 1: Core Validation
- Target a small, weak base model (~270M parameters) to amplify BEM's impact
- Implement the "generated" variant (U diag(c) V^T) with per-sequence routing
- Train on a style/domain task and benchmark against static LoRA

### Phase 2: Expert Integration  
- Integrate micro-retriever with Hyde (Hypothetical Document Embeddings)
- Train the BEM to learn the policy of using retrieved evidence
- Implement asynchronous retrieval to avoid blocking token generation

### Phase 3: Performance Optimization
- Move to chunk-wise routing (every 32 tokens)
- Develop fused CUDA kernels for dynamic updates
- Implement chunk-wise gating for attention layers

### Phase 4: Advanced Features
- Experiment with the "banked" (MoE) variant
- Multi-BEM composition with norm budgeting
- Cross-layer tying and adaptive rank allocation

## Future Directions

- **Ada-rank**: let the controller allocate rank across layers online (dynamic r)
- **Cross-layer tying**: share codes across attention+MLP within a block to stabilize style
- **Programmatic rails**: reserve one expert slot for safety/tone; expose a scalar knob
- **Distill to static LoRA** for cheap devices, keep BEM for premium deployments

## Packaging Format

```
bem/
 ├─ manifest.json          # supported families, layers, r, K, gating schedule, norm budgets
 ├─ controller.safetensors
 ├─ generator.safetensors  # U/V or A/B
 ├─ shim/                  # per-family layer maps
 └─ optional_index.faiss   # retrieval memory (optional)
```

This vision provides a roadmap for creating true bolt-on experts: modular, composable, and situationally intelligent—able to add a new voice or specialty without mutating the core model.