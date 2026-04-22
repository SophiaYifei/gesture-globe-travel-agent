# WanderMind — AI Vacation Planning Agent
### Project 2 · Perception · Planning · Control

An interactive vacation planner that lets users click anywhere on a world map and instantly generates a full itinerary. Built in two versions: one using classical CV + rule-based planning (Non-DL), and one using CLIP + LLM (DL).

---

## 📁 Project Structure

```
vacation-agent/
├── app.py                   # Flask server (entry point)
├── requirements.txt
├── agents/
│   ├── non_dl_agent.py      # Classical CV + rule-based planner
│   └── dl_agent.py          # CLIP perception + LLM planner
├── evaluation/
│   └── evaluator.py         # Evaluation framework
└── templates/
    └── index.html           # Full frontend UI
```

---

## 🚀 Setup & Run

### 1. Install Python dependencies

```bash
cd vacation-agent
pip install -r requirements.txt
```

### 2. Set your Anthropic API key (for DL agent planning)

**Mac/Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-...
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

> **Note:** The DL agent uses the Anthropic API for itinerary generation. Get a key at https://console.anthropic.com. Without a key, the DL agent gracefully falls back to the rule-based planner.

### 3. Run the app

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🎮 How to Use

1. **Click anywhere on the map** to drop a pin and trigger the agent
2. **Toggle between agents** using the top-right buttons:
   - `⚙ Rule-based` — Non-DL agent (classical CV + templates)
   - `✦ AI (DL)` — DL agent (CLIP + Claude LLM)
3. **Browse tabs** in the side panel:
   - **Itinerary** — Day-by-day plan with morning/afternoon/evening activities
   - **Food & Tips** — Local cuisine and practical advice
   - **Refine** — Adjust duration, style, and budget then regenerate
   - **Info** — Technical details about what the agent detected and how
4. **Run Evaluation** (bottom-left button) — tests both agents on 12 ground-truth locations

---

## 🧠 Architecture

### Non-DL Agent
| Stage | Method |
|-------|--------|
| Perception | PIL color histogram analysis + lat/lng coordinate lookup against a 35-region geographic database |
| Planning | Rule-based template system: biome → activity/food/tip lookup tables, day themes pre-defined |
| Control | Structured itinerary renderer, deterministic output |

### DL Agent
| Stage | Method |
|-------|--------|
| Perception | CLIP (openai/clip-vit-base-patch32) zero-shot classifies map screenshot against 12 terrain descriptions |
| Planning | Anthropic Claude LLM: structured JSON prompt with location context, preferences, and conversation history for refinement |
| Control | Dynamic output with feedback loop — user can refine preferences and regenerate |

---

## 📊 Evaluation

The evaluator tests both agents on 12 ground-truth locations (Nepal, Bali, Paris, Mexico, Iceland, Kenya, Greece, Amazon, India, Japan, Australia, London).

**Metrics:**
- **Region accuracy** — Did the agent correctly identify the destination name and biome?
- **Itinerary relevance** — Do the generated activities match known keywords for that environment?
- **Response time** — Milliseconds per request

Run via the UI button, or directly:
```bash
python -c "from evaluation.evaluator import run_evaluation; import json; print(json.dumps(run_evaluation(), indent=2))"
```

---

## 🔧 Optional: Enable CLIP (DL Perception)

CLIP requires ~1.5GB of model weights and PyTorch. Install if you want true visual perception:

```bash
pip install torch transformers
```

The first run will download the CLIP model automatically. Without it, the DL agent falls back to coordinate lookup for perception but still uses the LLM for planning.

---

## 📹 Video Demo Tips

- Use the **DL agent** for the impressive demo (rich LLM-generated itineraries)
- Click exotic locations: Bali, Iceland, Maldives, Patagonia, Kenya
- Use **Refine tab** to show the feedback loop — change style to "Adventure" and regenerate
- Run **Evaluation** live to show the comparison metrics
- Switch between agents on the same location to show the architectural difference
