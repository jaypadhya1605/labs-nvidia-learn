# NVIDIA Senior Solutions Architect, Generative AI (JR2017709)
## Fit Analysis, Prep Plan & Profile Repositioning — Jay Padhya

---

## 1. The Verdict

**This specific req is a reach, and not the door you should walk through.**

Read the two lines that define the actual job:

> *"You will also collaborate with a diverse set of internal teams on performance analysis and modeling of inference software."*
> *"Analyze performance and power efficiency of deep learning inference workloads."*

This is not a pre-sales architect role with a GenAI label. This is an **inference performance engineering role that happens to be customer-facing**. The people who get hired into it spend their days in Nsight profiles and TensorRT engine builds, and present the results.

You are an outstanding **application-layer** GenAI architect. This req hires **infrastructure-layer** GenAI architects. That's the gap — and it's a real one, not an ATS keyword problem.

Three additional flags:

- **Location: US-CA-Santa Clara.** Your stated goal is a full-time *remote* role. This req is HQ-based.
- **Posting age.** It states applications accepted "at least until May 15, 2026." Today is July 16, 2026. It may still be open ("at least" is a floor, not a ceiling), but you're two months past the stated window on a req that's been live a while.
- **Salary band $184K–$287.5K** — this is a Level 4 band, consistent with a genuinely senior technical bar.

---

## 2. Requirement-by-Requirement Map

### "What We Need To See" — the hard filter

| Requirement | Your reality | Verdict |
|---|---|---|
| BS/MS/PhD in CS, EE/CE, Physics, Math, Engineering | BE Electronics Engineering (Univ. of Mumbai, 2012) + MS Information Systems (Northeastern, 2017) | ✅ **Pass** — Electronics Engineering is a direct hit on "Electrical/Computer Engineering" |
| **8+ years hands-on with DL frameworks (PyTorch, TensorFlow)** | Microsoft (Jun 2022–present), Oshkosh (TensorFlow/PySpark, 2022), CVS Health. Realistically ~4–5 years of framework-level work, and your last 4 years are **API-layer GenAI** (Azure OpenAI, Foundry) — not PyTorch | ❌ **Hard gap. This is the screening filter that stops the resume.** |
| Strong Python fundamentals, optimization, software design | Strong. SDK-first, IaC-first, production patterns | ✅ **Pass** |
| Strong problem-solving and debugging | Strong — UPMC 500-error root-cause (context overflow / tool-call loop), Gemma eval confound catch | ✅ **Pass** |
| **Excellent knowledge of theory and practice of LLMs and *DL inference*** | LLM *practice* is strong. **DL inference theory is untested** — KV cache, in-flight batching, TTFT/ITL, quantization tradeoffs, memory-bandwidth bounds | ⚠️ **Partial — this is what kills you in the loop, not the resume** |
| Excellent presentation, communication, collaboration | Elite. Technical pre-sales, live Fortune-50 demos, adjunct professor, published author | ✅ **Your single biggest differentiator** |
| Desire for diverse, creative projects | Self-evident from your portfolio | ✅ **Pass** |

### "Ways To Stand Out" — the tiebreakers

| Differentiator | Your reality | Verdict |
|---|---|---|
| NeMo, Triton, TensorRT, TensorRT-LLM | Nemotron Omni + llama-server locally; NVIDIA certs in progress. **No production Triton/TRT-LLM.** | ❌ Gap |
| Excellent C/C++ (debugging, profiling, optimization, perf analysis) | Not on your resume — Python, R, Java, SQL | ❌ Gap |
| Parallel programming / distributed computing | Limited | ❌ Gap |
| DL training at scale; deploying/optimizing DL inference in production | Gemma 4B LoRA on A100s — but Chirag's EIM team owned the GPUs and dataset | ⚠️ Partial |

**Honest score: ~45% on this req.** You pass the human-facing bar decisively and miss the systems bar decisively.

---

## 3. The Strategic Redirect — The Right NVIDIA Door

NVIDIA runs a **Healthcare & Life Sciences vertical SA team**. Their reqs read completely differently. From the live HLS SA family:

- *"5+ years of experience in software development related to AI/ML in healthcare or life sciences"* — not 8+ years PyTorch
- *"Experience deploying and scaling agentic AI solutions in cloud environments (AWS Bedrock, **Azure AI Foundry**, Vertex AI, etc.)"* — **this is a literal description of your day job**
- *"Deep experience with end-to-end generative AI solutions: data ingestion, preprocessing, model training, agentic tool development, pipeline deployment and evaluation"* — Molina + UPMC, exactly
- *"Excellent communication skills with the ability to present complex technical concepts to both technical and non-technical audiences"* — your core skill
- *"Established thought leadership through publications or presentations on AI/ML applications in healthcare and life science"* — **you co-authored a Packt book and teach at MSU**
- *"Experience collaborating with pharma, techbio, and healthcare providers"* — Molina, UPMC, Providence, CommonSpirit, Bayer
- *"Familiarity with AI deployment/inference technologies such as TensorRT, TRT-LLM"* — listed under **nice-to-have**, not required
- Posted as **US Remote**, Level 3 ($148K–$235.7K) / Level 4 ($184K–$287.5K)

**Your fit on that family is ~75–80%, versus ~45% here.** Same company, same band at L4, same GenAI work, remote — and it plays to the domain expertise that took you four years to build instead of asking you to discard it.

**Action:** Set a saved search on NVIDIA's Workday board filtered to `Solutions Architect` + `Healthcare` + `Remote`. The HLS reqs (JR2005828-type) cycle regularly. Don't burn your one application on JR2017709.

> ⚠️ Note: the HLS reqs I found are from the Oct 2025 posting cycle and at least one shows as closed. Check the live board before applying. NVIDIA reposts this family several times a year.

---

## 4. The Highest-Leverage Move (Costs You Zero Study Hours)

**Work a joint Microsoft–NVIDIA co-sell opportunity through your day job.**

You are a Microsoft HLS solutions engineer with Molina, UPMC, Providence, and CommonSpirit. NVIDIA has an HLS vertical team that co-sells with Azure on exactly these accounts. Azure NC/ND GPU workloads, NIM on AKS, Clara — these are joint motions with real NVIDIA field engagement.

Engineering yourself into one of those rooms means:
- You do the NVIDIA SA job, with NVIDIA SAs watching, on Microsoft's clock
- You get a referral from someone who has seen you work, not from a cold LinkedIn message
- A referral is the **only** realistic path past the "8+ years PyTorch" filter on a req like JR2017709

Every hour here is worth ten hours of certification study. Referral is the mechanism; everything in §5 is what makes the referral convert.

---

## 5. The 90-Day Prep Plan

### The core diagnosis

Azure abstracts inference economics away behind a PTU. That abstraction is precisely what NVIDIA sells against. So you've never been *forced* to answer:

> *"Your Molina voice agent does 500K calls/month with a sub-second latency budget. How many GPUs? What's your TTFT p95? What batch size? What quantization? What's the cost per call?"*

**Here's the reframe that unlocks your whole candidacy: your Molina Voice AI project is already an inference economics problem.** 500K+ calls/month against a hard real-time latency budget *is* a throughput/latency/concurrency problem. You just expressed it in PTU units instead of tokens/sec/GPU. Same work. Different currency. Learn the currency.

### One project. Not five.

**"Prior-Auth Agent on the NVIDIA Stack: An Inference Economics Teardown."**

Rebuild the thing you already know cold — a prior-auth agent — on NVIDIA's inference stack, with real measured numbers.

**Phase 1 — Weeks 1–3: GPU access + vocabulary**
- Free NIM endpoints at `build.nvidia.com` (zero setup, works from your M4)
- Spin an Azure `NC24ads A100 v4` — your Azure job *is* your GPU bridge; use it
- Master the vocabulary cold: TTFT, ITL/TPOT, throughput vs. goodput, concurrency, KV cache, PagedAttention, in-flight/continuous batching, tensor vs. pipeline parallel
- **Acceptance test:** you can explain from memory why LLM decode is *memory-bandwidth-bound, not compute-bound*, and what that implies for batching. This is the single most-asked NVIDIA inference question. If you can't answer it, you fail the loop regardless of everything else.

**Phase 2 — Weeks 4–8: Build + benchmark (this is the credibility move)**
- Deploy Llama 3.1 8B (or a Nemotron) via **NIM** on the A100
- Then deploy the *same model* via **TRT-LLM + Triton**, hand-building the engine. Doing it the hard way once is what separates you from every other pre-sales resume
- Sweep with `genai-perf`: concurrency 1 / 4 / 16 / 64 / 128 → TTFT p50/p95, ITL, req/s, tok/s/GPU
- Quantize: FP16 → FP8 → INT4 AWQ. Measure the **quality delta on a healthcare eval set** — you already know how to build these
- Produce two charts: (a) tokens/sec/GPU vs. latency, (b) **$ per 1,000 prior-auth decisions**

**Phase 3 — Weeks 9–12: Wrap in domain + publish**
- **NeMo Guardrails** for PHI redaction — a healthcare differentiator almost nobody demos
- **NeMo Retriever** over a policy-PDF corpus (your UPMC pattern)
- **NeMo Agent Toolkit** for orchestration
- Ship: public repo + blog post + a talk. Title it something like *"What a 500K-call/month healthcare voice agent actually costs: an inference teardown."*

**What this single artifact buys you:** NeMo ✓ Triton ✓ TensorRT-LLM ✓ production DL inference ✓ healthcare domain ✓ thought leadership ✓ — every "way to stand out" except C/C++, in one project, in your own domain, honestly claimable.

### On the C/C++ gap: don't chase it

It's under "ways to stand out," not "what we need to see." Ninety days of C++ instead of inference benchmarking is a bad trade for an SA role. If you want a token: learn to **read** CUDA kernels and profile with **Nsight Systems**. For an SA, reading > writing.

### On the 8+ years PyTorch gap: it can't be fixed

Two honest options, in order:
1. **Target the HLS SA family** (5+ years AI/ML) — real, remote, and a genuine fit today
2. **Get a referral** for the GenAI reqs — see §4

Do not pad the resume to fake eight years of PyTorch. NVIDIA interviews go deep and fast, and a fabricated claim will collapse in the first technical screen.

---

## 6. Profile Repositioning

> **Verify before use:** I couldn't reach your Drive (search errored), so this is built from your resume backbone in our prior conversations and your project history. Confirm all dates and titles against your live resume — especially total years of experience, which I've left as a placeholder.

### LinkedIn headline

> **GenAI Solutions Architect — Healthcare & Life Sciences | Agentic AI, LLM Inference & GPU-Accelerated Deployment | Author, *Agentic AI with Microsoft Foundry* | Adjunct Professor, Michigan State**

### LinkedIn "About" — opening paragraph

> I help healthcare payors and providers get generative AI into production — not into slide decks. Over the last four years I've architected and delivered multi-agent LLM systems, real-time voice AI, and RAG platforms for Fortune 50 health plans and integrated delivery networks, including prior authorization, utilization management, and member-facing voice agents operating at 500K+ calls per month.
>
> My work sits where clinical workflow meets inference economics: what a model costs per decision, what latency a clinician or member will actually tolerate, and how you evaluate a system honestly before it touches PHI. I co-authored *Agentic AI with Microsoft Foundry* (Packt, 2026) and teach AI/ML as an Adjunct Professor at Michigan State University.

### Resume summary (NVIDIA-facing)

> Customer-facing GenAI solutions architect specializing in production LLM and multi-agent systems for healthcare payors and providers. **[X]+ years** building and deploying ML/AI systems; 4+ years as a trusted technical advisor to Fortune 50 health plans and IDNs. Depth in agentic orchestration, retrieval, evaluation methodology, and inference capacity planning across Azure AI Foundry, with hands-on LoRA fine-tuning on NVIDIA A100 GPUs. Published author on agentic AI; Adjunct Professor of AI/ML.

### Bullet translations — Azure currency → NVIDIA currency

These are all things you actually did. Only the framing changes.

| What you'd write today | What NVIDIA needs to read |
|---|---|
| Built Azure OpenAI cost model for Molina (PTU vs. PAYG) | **Built the inference capacity and cost model for a Fortune 50 health plan's GenAI platform — modeling token throughput, concurrency, and per-request cost across provisioned vs. on-demand serving to size GPU capacity at production scale** |
| Molina Voice AI: Genesys → APIM → Azure Voice Live, GPT Realtime | **Architected a real-time conversational AI system targeting 500K+ calls/month under a sub-second end-to-end latency budget, decomposing the ASR → LLM → TTS pipeline against a hard real-time SLA** |
| Gemma 4B LoRA fine-tuning POC with EIM team | **Led a LoRA fine-tuning program for a 4B-parameter model on NVIDIA A100 GPUs for browser-control/form automation; designed the evaluation methodology and identified a serving-stack confound — baseline scored through vLLM vs. fine-tuned through a custom inference path — that invalidated the comparison** |
| UPMC utilization management multi-agent POC | **Built a multi-agent LLM system (policy + claims agents) over a real clinical-policy corpus with vector retrieval; root-caused a production failure to context overflow driving a tool-call loop** |
| Co-authored Packt book, Chapter 8 | **Published author on advanced agent capabilities and tool integration (MCP), with healthcare case studies on prior authorization and IT service agents** |
| Adjunct professor, MSU | **Adjunct Professor of AI/ML — teach, and build curriculum for, applied deep learning and agentic systems** |

**Your best interview story is the Gemma eval confound.** Catching that a base model was scored through Foundry's vLLM while the fine-tuned model ran through a custom inference script — and that this invalidated the whole comparison — is an *inference-serving* insight, not an application-layer one. It proves you understand that the serving stack changes the result. Lead with it. NVIDIA will recognize it immediately.

---

## 7. What To Do This Week

1. **Don't apply to JR2017709 cold.** It won't clear the filter and it burns the req.
2. **Set the saved search** — NVIDIA Workday: Solutions Architect + Healthcare + Remote.
3. **Open the co-sell door.** Find NVIDIA's HLS field team through your Azure GPU/co-sell motion on Molina or Providence. One meeting is worth a hundred applications.
4. **Start Phase 1.** Free NIM endpoint on `build.nvidia.com` tonight. It takes 20 minutes and it's the first honest line of NVIDIA experience on your resume.
5. **Rewrite the LinkedIn headline and About.** Ten minutes, and recruiters for the HLS family search on exactly those terms.
