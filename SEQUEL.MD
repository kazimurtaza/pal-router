From Classifier to Orchestrator: NVIDIA Just Validated Our Approach (And Took It Further)
TL;DR: Two months ago, I built a ternary router that sends queries to weak models, strong models, or code execution. Last week, NVIDIA published a paper showing an 8B "orchestrator" model that beats GPT-5 on Humanity's Last Exam — using the same core insight. Here's what they did differently, and what it means for the future of LLM systems.

Part 1 Recap: The Ternary Router
In Part 1, I argued that binary LLM routing (weak vs strong) misses a crucial category: queries that need tools, not bigger models.
The solution was a three-lane router:
FAST      → Weak model (simple facts)
REASONING → Strong model (complex analysis)  
AGENTIC   → Weak model + code execution (computation)
It worked. 91.7% routing accuracy. 85% accuracy on math problems. 45% cost reduction.
But the router was a classifier — an embedding model plus a 2-layer MLP trained on 457 queries. It couldn't adapt, couldn't reason about why to route somewhere, and couldn't handle tools it hadn't seen before.
Then NVIDIA dropped ToolOrchestra.

What NVIDIA Built
NVIDIA's Orchestrator-8B is a fine-tuned Qwen3-8B model trained specifically to orchestrate — to decide which tools and models to call, in what order, for any given query.
The headline result:
ModelHumanity's Last ExamCostGPT-535.1%1.0xClaude Opus 4.132.8%~1.2xOrchestrator-8B37.1%0.4x
An 8B model, routing to tools and other models, beats GPT-5 at 30% of the cost.
This isn't a benchmark quirk. They validated on FRAMES (multi-hop reasoning) and τ²-Bench (tool use) with consistent results.

The Key Differences
1. The Router is an LLM, Not a Classifier
My approach:
Query → Embedding → MLP → {FAST, REASONING, AGENTIC}
NVIDIA's approach:
Query → Orchestrator LLM → "I should call code_executor, then search, then GPT-4"
The orchestrator doesn't just classify — it reasons about the routing decision. It can chain tools, backtrack on failures, and explain its choices.
2. Multi-Objective Reinforcement Learning
I trained on accuracy alone. NVIDIA trained on three objectives simultaneously:
ObjectiveWhat It OptimizesAccuracyGet the right answerEfficiencyMinimize cost and latencyPreferenceFollow user constraints ("use only open-source models")
They used Group Relative Policy Optimization (GRPO) with a combined reward:
R = α·accuracy + β·efficiency + γ·preference_adherence
This means the orchestrator learns to make trade-offs. If a user says "prioritize speed," it routes differently than "prioritize accuracy."
3. Generalizes to Unseen Tools
My router was static — it knew about exactly three lanes. If I added a fourth tool (say, a web search API), I'd need to retrain.
NVIDIA's orchestrator generalizes. In their experiments, they tested on tools the model had never seen during training. It figured out how to use them from their descriptions alone.
This is the difference between a lookup table and actual reasoning.
4. Dynamic Pricing Awareness
Here's something I didn't even consider: API prices change.
NVIDIA trained their orchestrator to adapt to different pricing configurations. If GPT-4 gets cheaper, it should route more queries there. If a tool gets rate-limited, it should fall back.
My router had hardcoded cost assumptions. Theirs learns the trade-offs.

What This Validates
NVIDIA's paper confirms the core thesis from Part 1:

The answer to hard problems isn't always a bigger model — sometimes it's the right tool.

But they took it further:
InsightPart 1NVIDIATools beat scale for computation✓ Validated✓ ValidatedRouting is learnable✓ Classifier works✓ LLM works betterCost-aware routing mattersImplicitExplicit reward signalRouting should be dynamic✗ Static classifier✓ Adapts in real-time
The validation is gratifying. The extension is humbling.

The New Mental Model
After reading the ToolOrchestra paper, I think about routing differently.
Old mental model: Routing is classification
Query → Classifier → Fixed Lane → Execution
New mental model: Routing is planning
Query → Planner → [Tool₁ → Result₁ → Tool₂ → Result₂ → ...] → Answer
The orchestrator doesn't just pick a lane. It constructs an execution plan — potentially calling multiple tools in sequence, using intermediate results to decide next steps.
This is closer to how humans solve complex problems. We don't classify a math problem as "calculator-worthy" and blindly compute. We reason: "I need to break this into parts, compute each part, then combine."

What I'd Do Differently Now
If I were rebuilding the PAL-Router today:
1. Replace the Classifier with a Small LLM
Instead of embedding + MLP, use a fine-tuned 1-3B model that outputs structured routing decisions:
json{
  "plan": [
    {"tool": "code_executor", "reason": "needs arithmetic"},
    {"tool": "weak_model", "reason": "format the result"}
  ],
  "confidence": 0.87
}
This preserves the cost advantage (small model for routing) while gaining flexibility.
2. Add a Verification Step
NVIDIA's system can backtrack. If a tool call fails or returns garbage, the orchestrator can try a different approach.
My system had retry logic for code execution, but no ability to say "this answer looks wrong, let me try the strong model instead."
3. Make Cost a First-Class Citizen
Instead of implicit cost tiers (weak = cheap, strong = expensive), explicitly model:
pythoncost = Σ (tokens_in × price_in + tokens_out × price_out) for each call
Then optimize for accuracy/cost Pareto frontier, not just accuracy.
4. Train on Reasoning Traces
NVIDIA used RL on execution traces — the full sequence of tool calls and results. This teaches the model why certain routing decisions work.
My classifier only saw (query, correct_lane) pairs. It learned correlations, not causation.

The Bigger Picture: The Orchestration Era
I think we're entering a new phase of LLM development:
EraScaling StrategyExamples2020-2023Bigger modelsGPT-3 → GPT-42023-2024Longer context4K → 128K → 1M tokens2024-2025Reasoning chainso1, R1, chain-of-thought2025+OrchestrationTool use, multi-model routing
The insight: intelligence isn't monolithic.
A system that routes math to calculators, search to Google, and nuance to GPT-4 will beat a system that tries to do everything with one giant model.
NVIDIA proved this empirically. Orchestrator-8B beats GPT-5 not because it's smarter, but because it's better at knowing what it doesn't know.

Practical Implications
For Builders
If you're building LLM applications today:

Don't default to the strongest model — Route based on query type
Add tool use — Code execution, search, and retrieval are force multipliers
Consider a learned router — Even a small fine-tuned model beats heuristics
Measure cost/accuracy trade-offs — Not just accuracy

For Researchers
Open questions from the ToolOrchestra paper:

Minimum orchestrator size? — Can a 1B model orchestrate effectively?
Transfer across domains? — Does an orchestrator trained on benchmarks work for production traffic?
Online learning? — Can orchestrators improve from production feedback without full retraining?
Orchestrator orchestrators? — At what point do you need hierarchy?

For the Industry
The endgame is clear: the best AI system won't be one model.
It'll be:

A router/orchestrator (small, fast, cheap)
A toolbox (code execution, search, retrieval, specialized models)
A set of fallback models (weak → strong → frontier)
A verification layer (catch mistakes, trigger retries)

OpenAI, Anthropic, and Google are probably all building this internally. NVIDIA just published the blueprint.

What's Next
I'm planning to:

Replicate NVIDIA's results — Fine-tune a small model on ToolScale dataset
Benchmark against my classifier — Quantify the accuracy gain from learned routing
Test on production traffic — Does this generalize beyond benchmarks?

If you want to follow along, [subscribe / follow / star repo].

Conclusion
Two months ago, I built a ternary router based on a hunch: some queries need tools, not bigger models.
NVIDIA just validated that hunch at scale — and showed that the router itself should be a reasoning model, not a classifier.
The insight generalizes:

Stop asking "which model should answer this?"
Start asking "what tools and models, in what sequence, will solve this?"

The future isn't one big model. It's a small model that knows when to call for help.

Part 1: Beyond Binary: A Ternary Router for the LLM Reasoning Cliff
NVIDIA Paper: ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration
Code: [github.com/...]