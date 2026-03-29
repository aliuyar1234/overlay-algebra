# RESEARCH_BRIEF.md

## 1. Core problem statement
Post-training usually learns task semantics and output behavior together.
That is convenient for shipping one model, but bad for science and operations:
- every new output requirement can require another full or partial fine-tune,
- adapter merging becomes hard to reason about,
- behavior transfer is opaque,
- composition failures are difficult to attribute.

This project asks whether a narrow set of **operational behaviors** can be isolated from shared semantics and recomposed causally.

## 2. Why this problem matters
Many real deployments do not mainly need "better general intelligence".
They need reliable output contracts:
- valid structured outputs,
- traceable support references,
- exact grounding / quoting.

If those can be turned into reusable overlays, teams can potentially:
- reuse behavior patches,
- reduce interference from monolithic retraining,
- study composition with clearer failure analysis,
- make post-training changes more auditable.

## 3. Proposed contribution in plain language
We will keep the underlying QA task fixed and vary only how the model is required to answer.
We will:
1. train one adapter for plain QA semantics,
2. train one adapter each for JSON, citation, and quote outputs on the exact same examples,
3. subtract the shared semantic adapter from each behavior adapter in dense-delta space,
4. test whether the remaining residuals can be recomposed into unseen pairs/triples.

This is a **causal composition study**, not a benchmark chase.

## 4. Locked scope
These are locked unless the repo docs are explicitly revised.

### Locked facts
- **Paper title:** *Overlay Algebra: Causal Composition of Operational Behaviors in Frozen Language Models*
- **Method name:** SOAR (Semantic-Operational Adapter Residuals)
- **v1 overlay set:** J (JSON), C (citation), Q (quote)
- **v1 task family:** answerable evidence-grounded QA only
- **v1 core dataset:** filtered SQuAD 1.1 with one support sentence per example
- **Base path:** frozen base model + LoRA adapters
- **Core algebra object:** effective dense adapter deltas, not raw LoRA factors
- **Core evaluation philosophy:** deterministic parsers, no LLM judges for core claims

### Locked scope boundaries
- no abstention in v1
- no agent environments
- no benchmark-spanning claim
- no "universal behavior atoms" claim
- no open-ended safety claim
- no claim that SOAR fully replaces direct multi-behavior training

## 5. Non-goals
- SOTA on QA
- generic adapter-merging improvements
- broad benchmark leadership
- theorem-heavy guarantees of composability
- production serving optimization
- claims about all model families
- large-scale human evaluation
- retrieval, tools, or agent planning

## 6. Target users / readers / evaluators
Primary readers:
- LLM post-training researchers
- adapter / PEFT researchers
- reliability / infrastructure researchers
- reviewers who care about causal evidence and reproducibility

Secondary readers:
- engineering teams maintaining libraries of narrow behavior adapters

## 7. What must be true for this project to count as a success
Minimum success bar:
1. the dataset/compiler/parsers are deterministic and leak-controlled,
2. at least **two** single overlays are recovered reasonably well after residualization,
3. at least **one** unseen pair composition beats both prompt-only control and tuned whole-adapter merging under a semantic-retention guardrail,
4. the paper can say something honest and specific about which behaviors are more or less composable.

Paper-strong success:
- the strongest pair works cleanly,
- the triple is partially viable,
- JSON is clearly easier to compose than citation/quote,
- the transfer slice shows at least limited external relevance.

## 8. What would count as partial failure
Still publishable if the evidence is clean:
- single overlays recover but pair/triple composition fails,
- composition only works for JSON,
- dense-delta residualization is valid but benefits over tuned merging are small,
- the main result becomes a map of **limits of linear operational composition** rather than a positive modularity result.

Likely not worth a paper if:
- the compiler/parsers are unreliable,
- support sentence mapping is noisy,
- residualization does not recover any single overlay,
- the project collapses into generic prompt engineering or generic merging without a distinct insight.

## 9. Assumptions and open questions
Assumptions:
- a meaningful fraction of overlay behavior is low-rank and recoverable after subtracting shared semantics,
- sentence-labeled QA gives a clean enough substrate for citation and quote behaviors,
- prompt instructions for combined overlays are enough for the model to "know what to do" once the residuals are present.

Open questions:
- which module subset is best: `o_proj + down_proj` only, or all linear layers?
- does residual rank 16 suffice, or is rank 32 needed after subtraction?
- how much should beta (`β`) be tuned instead of fixed at 1?
- how strong should the semantic guardrail be during alpha tuning?
- do quote and citation remain too entangled for useful linear composition?

## 10. The difference between “we aim to show” and “we have shown”
This repo begins with **aims**, not validated results.

### We aim to show
- some operational behaviors are partially composable after residualization,
- dense-delta residualization is the right technical object for the method,
- whole-adapter merging is a weaker baseline in this semantics-fixed setting.

### We have shown
At build-pack creation time: **nothing empirical yet**.
Only the research direction, method specification, claim structure, and implementation plan are locked.

## 11. The core scientific stance
The project is allowed to succeed with a positive result or with a disciplined negative result.
What is not allowed is:
- vague novelty inflation,
- unsupported universality language,
- benchmark theater,
- silently changing the question after results arrive.
