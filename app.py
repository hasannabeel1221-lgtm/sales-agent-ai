import streamlit as st
import time
import re
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════
#  🔑 API KEYS — loaded from .env, fallback to hardcoded
# ══════════════════════════════════════════════════════════
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "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")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY",  "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")
# ══════════════════════════════════════════════════════════

from groq import Groq
from tavily import TavilyClient


@dataclass
class AgentOutput:
    agent_name: str
    agent_icon: str
    status: str = "pending"
    output: str = ""
    metadata: dict = field(default_factory=dict)
    elapsed: float = 0.0

@dataclass
class PipelineState:
    target_company: str
    product_name: str
    research: AgentOutput = None
    persona: AgentOutput = None
    strategy: AgentOutput = None
    script: AgentOutput = None
    feedback: AgentOutput = None


class SalesScriptOrchestrator:

    def __init__(self, groq_key: str, tavily_key: str):
        self.groq = Groq(api_key=groq_key)
        # Active Groq models as of 2025 — no deprecated ones
        self._models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
        ]
        self.tavily = TavilyClient(api_key=tavily_key)

    def _generate(self, prompt: str) -> str:
        last_err = None
        for model_name in self._models:
            for attempt in range(4):
                try:
                    response = self.groq.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2048,
                        temperature=0.7,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    last_err = e
                    s = str(e)
                    if any(x in s for x in ["404", "not found", "not supported", "deprecated", "decommissioned", "model_not_active", "model_decommissioned"]):
                        break  # skip to next model
                    if "429" in s or "quota" in s.lower() or "rate" in s.lower():
                        m = re.search(r"seconds:\s*(\d+)", s)
                        wait = int(m.group(1)) + 3 if m else 30 * (attempt + 1)
                        time.sleep(min(wait, 90))
                        continue
                    raise
        raise RuntimeError(f"Groq failed on all models: {last_err}\n\nCheck your API key or quota at https://console.groq.com/")

    def run_research_agent(self, state: PipelineState) -> AgentOutput:
        agent = AgentOutput("Research Agent", "🔍", "running")
        t0 = time.time()
        try:
            queries = [
                f"{state.target_company} latest news 2025",
                f"{state.target_company} business challenges problems",
                f"{state.target_company} strategic priorities growth",
            ]
            all_results = []
            for q in queries:
                r = self.tavily.search(query=q, search_depth="advanced", max_results=3, include_answer=True)
                if r.get("answer"):
                    all_results.append(f"**Query:** {q}\n**Answer:** {r['answer']}")
                for item in r.get("results", []):
                    all_results.append(f"**Source:** {item.get('url','')}\n**Title:** {item.get('title','')}\n**Snippet:** {item.get('content','')[:400]}")
            raw = "\n\n---\n\n".join(all_results)
            prompt = f"""You are a B2B Sales Research Analyst. Using ONLY the Tavily data below (no hallucination), summarize:
1. Recent news about {state.target_company}
2. Key business challenges
3. Strategic priorities
4. Leadership or funding news
5. Competitive signals

TAVILY DATA:
{raw}

Use clear headings and bullets. Cite real facts. Say Unknown if missing."""
            agent.output = self._generate(prompt)
            agent.metadata = {"sources_scraped": len(all_results)}
            agent.status = "done"
        except Exception as e:
            agent.status = "error"
            agent.output = f"Error: {e}"
        agent.elapsed = time.time() - t0
        return agent

    def run_persona_agent(self, state: PipelineState) -> AgentOutput:
        agent = AgentOutput("ICP & Persona Agent", "👤", "running")
        t0 = time.time()
        try:
            prompt = f"""You are a B2B sales strategist. Define the ideal buyer persona at {state.target_company} for {state.product_name}.

Research: {state.research.output}

## Primary Buyer Persona
- Job Title, Department, Seniority

## Goals & KPIs
- Metrics owned, biggest worries

## Pain Points (from research)
- 3-5 specific pains

## How They Buy
- Decision style, buying committee, channels

## What Hooks Them
- Resonant messaging, proof points that matter"""
            agent.output = self._generate(prompt)
            agent.status = "done"
        except Exception as e:
            agent.status = "error"
            agent.output = f"Error: {e}"
        agent.elapsed = time.time() - t0
        return agent

    def run_strategy_agent(self, state: PipelineState) -> AgentOutput:
        agent = AgentOutput("Copy Strategy Agent", "🧠", "running")
        t0 = time.time()
        try:
            prompt = f"""You are a world-class B2B copywriting strategist.

Company: {state.target_company} | Product: {state.product_name}
Research: {state.research.output[:1200]}
Persona: {state.persona.output[:1200]}

## Recommended Tone
Pick ONE: Aggressive/Bold | Empathetic/Consultative | Professional/Credibility. Explain why.

## Core Messaging Pillars
3 key messages for this persona.

## Hook Strategy
The #1 opening angle.

## Unique Value Framing
How to frame {state.product_name} for {state.target_company} specifically.

## What to AVOID

## Script Format
Email/LinkedIn/Phone — pick best, state length and CTA style."""
            agent.output = self._generate(prompt)
            agent.status = "done"
        except Exception as e:
            agent.status = "error"
            agent.output = f"Error: {e}"
        agent.elapsed = time.time() - t0
        return agent

    def run_script_agent(self, state: PipelineState) -> AgentOutput:
        agent = AgentOutput("Script Generation Agent", "✍️", "running")
        t0 = time.time()
        try:
            prompt = f"""You are an elite B2B sales copywriter. Write the final outreach script.

Company: {state.target_company} | Product: {state.product_name}
Research: {state.research.output[:1000]}
Persona: {state.persona.output[:800]}
Strategy: {state.strategy.output[:1000]}

Write using Hook → Value → CTA. Use REAL details from the research.

## THE SALES SCRIPT

### SUBJECT LINE (3 options A/B/C):

---
### HOOK (2-3 sentences, reference something specific from research):

---
### VALUE BRIDGE (4-6 sentences, connect their pain to your solution):

---
### SOCIAL PROOF (1-2 sentences):

---
### CALL TO ACTION (1-2 sentences, soft and specific):

---
### P.S. LINE:

---
## DELIVERY NOTES
- Best time to send, follow-up cadence, tone reminder"""
            agent.output = self._generate(prompt)
            agent.status = "done"
        except Exception as e:
            agent.status = "error"
            agent.output = f"Error: {e}"
        agent.elapsed = time.time() - t0
        return agent

    def run_feedback_agent(self, state: PipelineState) -> AgentOutput:
        agent = AgentOutput("Optimization & Feedback Agent", "⚖️", "running")
        t0 = time.time()
        try:
            prompt = f"""You are an expert B2B sales coach acting as an impartial judge.

SCRIPT:
{state.script.output}

Context: {state.target_company} | {state.product_name}
Persona: {state.persona.output[:500]}

## SCORECARD
| Criterion | Score (1-10) | Reasoning |
|-----------|-------------|-----------|
| Personalization | /10 | Real specific details? |
| Hook Strength | /10 | Exec keeps reading? |
| Value Clarity | /10 | ROI clear in 30s? |
| Tone Alignment | /10 | Matches persona? |
| CTA Effectiveness | /10 | Clear, low-friction? |
| Credibility | /10 | Proof points solid? |
| Conciseness | /10 | Any filler? |

## OVERALL SCORE: [X / 10]

## TOP 3 STRENGTHS
## TOP 3 IMPROVEMENTS
## REWRITE SUGGESTIONS
## FINAL VERDICT"""
            agent.output = self._generate(prompt)
            m = re.search(r'OVERALL SCORE[:\s]*(\d+(?:\.\d+)?)\s*/\s*10', agent.output, re.IGNORECASE)
            agent.metadata["score"] = float(m.group(1)) if m else None
            agent.status = "done"
        except Exception as e:
            agent.status = "error"
            agent.output = f"Error: {e}"
        agent.elapsed = time.time() - t0
        return agent


def render_agent_card(agent: AgentOutput, expanded: bool = True):
    icons = {"done": "🟢", "running": "🟡", "error": "🔴", "pending": "⚪"}
    label = f"{agent.agent_icon} {agent.agent_name} {icons.get(agent.status,'⚪')}"
    if agent.elapsed > 0:
        label += f"  _(⏱ {agent.elapsed:.1f}s)_"
    with st.expander(label, expanded=expanded):
        if agent.status == "error":
            st.error(agent.output)
        elif agent.status == "done":
            st.markdown(agent.output)
            if agent.metadata.get("score") is not None:
                st.metric("⚖️ Judge Score", f"{agent.metadata['score']}/10")
            if agent.metadata.get("sources_scraped"):
                st.metric("🔍 Sources Scraped", agent.metadata["sources_scraped"])
        else:
            st.info("⚙️ Processing…")


def pipeline_bar(done: list, active: str = None) -> str:
    steps = [("research","🔍","Research"),("persona","👤","Persona"),("strategy","🧠","Strategy"),("script","✍️","Script"),("feedback","⚖️","Judge")]
    html = '<div style="display:flex;align-items:center;margin:1.5rem 0;font-family:monospace;font-size:0.75rem;">'
    for i,(key,icon,label) in enumerate(steps):
        bg,border,color = ("#0d2015","#10b981","#10b981") if key in done else (("#1e1040","#7c3aed","#a78bfa") if key==active else ("#12121a","#1e1e2e","#64748b"))
        r = "8px 0 0 8px" if i==0 else ("0 8px 8px 0" if i==4 else "0")
        html += f'<div style="flex:1;background:{bg};border:1px solid {border};padding:0.6rem 0.3rem;text-align:center;color:{color};border-radius:{r};">{icon}<br>{label}</div>'
        if i<4: html += '<div style="color:#64748b;padding:0 3px;">→</div>'
    return html + '</div>'


def main():
    st.set_page_config(page_title="Sales Script Generator", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Mono&family=Inter:wght@400;500&display=swap');
    html,body,[data-testid="stApp"]{background:#0a0a0f!important;color:#e2e8f0!important;font-family:'Inter',sans-serif;}
    .hero{background:linear-gradient(135deg,#0f0520,#0a1628);border:1px solid #2d1b69;border-radius:16px;padding:2.5rem 3rem;margin-bottom:2rem;}
    .hero h1{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#a78bfa,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 0.4rem 0;}
    .hero p{font-family:'IBM Plex Mono',monospace;color:#64748b;font-size:0.82rem;margin:0;}
    .stTextInput>div>div>input{background:#12121a!important;border:1px solid #1e1e2e!important;color:#e2e8f0!important;border-radius:8px!important;}
    .stButton>button{background:linear-gradient(135deg,#7c3aed,#4f46e5)!important;color:white!important;border:none!important;border-radius:10px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:1rem!important;width:100%!important;padding:0.75rem!important;}
    [data-testid="stSidebar"]{background:#12121a!important;border-right:1px solid #1e1e2e!important;}
    [data-testid="stMetric"]{background:#0a0a0f!important;border:1px solid #1e1e2e!important;border-radius:8px!important;padding:0.8rem!important;}
    #MainMenu,footer,header{visibility:hidden;}
    .block-container{padding-top:2rem!important;}
    </style>""", unsafe_allow_html=True)

    st.markdown('<div class="hero"><h1>🎯 Multi-Agent Sales Script Generator</h1><p>RESEARCH → PERSONA → STRATEGY → SCRIPT → JUDGE &nbsp;|&nbsp; GROQ (Llama 3.3) + TAVILY</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        groq_key   = st.text_input("Groq API Key",   value=GROQ_API_KEY,   type="password")
        tavily_key = st.text_input("Tavily API Key", value=TAVILY_API_KEY, type="password")
        st.markdown("---")
        st.markdown("### 🤖 Agent Pipeline")
        st.markdown("1. 🔍 **Research** — Tavily web intel\n2. 👤 **Persona** — ICP definition\n3. 🧠 **Strategy** — Copy direction\n4. ✍️ **Script** — Final draft\n5. ⚖️ **Judge** — LLM-as-Judge score")
        st.markdown("---")
        st.caption("Powered by **Groq** (llama-3.3-70b-versatile) + **Tavily** real-time search.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        target_company = st.text_input("🏢 Target Company", placeholder="e.g. Tesla, Salesforce…")
    with c2:
        product_name = st.text_input("🚀 Your Product / Service", placeholder="e.g. AI analytics platform…")
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚡ Generate Sales Script", use_container_width=True)

    if "agent_outputs" not in st.session_state:
        st.session_state.agent_outputs = {}
    if "pipeline_state" not in st.session_state:
        st.session_state.pipeline_state = None

    KEYS = ["research","persona","strategy","script","feedback"]

    if run_btn:
        if not target_company or not product_name:
            st.error("⚠️ Fill in both fields.")
            st.stop()
        gkey = groq_key.strip()
        tkey = tavily_key.strip()
        if not gkey or len(gkey) < 10:
            st.error("⚠️ Invalid Groq key. Check your .env file or sidebar.")
            st.stop()
        if not tkey or len(tkey) < 10:
            st.error("⚠️ Invalid Tavily key. Check your .env file or sidebar.")
            st.stop()

        st.session_state.agent_outputs = {}
        st.markdown("---")
        st.markdown("### 🔄 Agent Pipeline Running…")
        bar_ph = st.empty()
        phs = {k: st.empty() for k in KEYS}
        done_steps = []

        STEPS = [
            ("research", "Research"),
            ("persona",  "Persona"),
            ("strategy", "Strategy"),
            ("script",   "Script"),
            ("feedback", "Judge"),
        ]

        try:
            orch = SalesScriptOrchestrator(gkey, tkey)
            state = PipelineState(target_company=target_company, product_name=product_name)
            fns = [
                orch.run_research_agent,
                orch.run_persona_agent,
                orch.run_strategy_agent,
                orch.run_script_agent,
                orch.run_feedback_agent,
            ]

            for idx, key in enumerate(KEYS):
                label = STEPS[idx][1]
                bar_ph.markdown(pipeline_bar(done_steps, key), unsafe_allow_html=True)
                with phs[key].container():
                    st.info(f"⚙️ **{label} Agent** is running…")
                out = fns[idx](state)
                setattr(state, key, out)
                st.session_state.agent_outputs[key] = out
                done_steps.append(key)
                with phs[key].container():
                    render_agent_card(out, expanded=True)
                if key != "feedback":
                    time.sleep(3)  # Groq is fast — 3s buffer is enough

            st.session_state.pipeline_state = state
            bar_ph.markdown(pipeline_bar(KEYS), unsafe_allow_html=True)
            score = state.feedback.metadata.get("score") if state.feedback else None
            total = sum(getattr(state, k).elapsed for k in KEYS if getattr(state, k))
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("✅ Agents Done", "5 / 5")
            col2.metric("⏱ Total Time", f"{total:.0f}s")
            col3.metric("⚖️ Judge Score", f"{score}/10" if score else "N/A")
            st.success("🎉 Done! Scroll up to review each agent's output.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    elif st.session_state.agent_outputs:
        st.markdown("---")
        st.markdown("### 📋 Last Run Results")
        st.markdown(pipeline_bar(KEYS), unsafe_allow_html=True)
        for key in KEYS:
            out = st.session_state.agent_outputs.get(key)
            if out:
                render_agent_card(out, expanded=(key=="script"))
    else:
        st.markdown("---")
        st.markdown(pipeline_bar([]), unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;padding:3rem;color:#475569;font-family:monospace;">Enter a company and product above, then click Generate.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()