"""
BlackClaw Exploration — Dive + Pattern Extraction
Searches a seed domain and extracts abstract patterns via LLM.
"""
import json
import google.generativeai as genai
from tavily import TavilyClient
from config import GEMINI_API_KEY, TAVILY_API_KEY, MODEL
from sanitize import sanitize, check_llm_output
from store import increment_tavily_calls, increment_llm_calls
genai.configure(api_key=GEMINI_API_KEY)
_gemini_model = genai.GenerativeModel(MODEL)
_tavily = TavilyClient(api_key=TAVILY_API_KEY)
EXTRACT_PROMPT = """You are a pattern extraction engine. Your job is to find abstract, transferable patterns in a domain of knowledge.
You are analyzing the domain of: {domain}
Based on the following research material, extract 3-5 core patterns, structures, or principles that are FUNDAMENTAL to this domain. Focus specifically on:
- Mathematical relationships (ratios, scaling laws, distributions)
- Dynamic behaviors (feedback loops, oscillations, phase transitions)
- Structural principles (hierarchies, networks, symmetries)
- Information patterns (encoding, compression, signal propagation)
- Emergence (how simple rules produce complex behavior)
For each pattern, you MUST provide:
- pattern_name: 2-5 word name
- description: 1-2 sentences explaining the pattern in this domain
- abstract_structure: Describe the SAME pattern using ZERO domain-specific language. Someone from a completely different field should understand the structure.
- search_query: A 3-6 word search query designed to find this SAME abstract pattern in UNRELATED domains. Use no terms from the original domain.
Respond ONLY with valid JSON. No markdown. No explanation. Format:
{{
  "patterns": [
    {{
      "pattern_name": "...",
      "description": "...",
      "abstract_structure": "...",
      "search_query": "..."
    }}
  ]
}}"""
def _search_seed(seed: dict) -> str:
    """Run Tavily searches for the seed domain and return combined content."""
    combined = []
    for query in seed["seed_queries"][:2]:  # Max 2 searches per seed
        try:
            results = _tavily.search(
                query=query,
                max_results=3,
                include_answer=False,
                search_depth="basic",
            )
            increment_tavily_calls(1)
            for result in results.get("results", []):
                content = result.get("content", "")
                if content:
                    # Sanitize BEFORE collecting
                    clean = sanitize(content)
                    if clean:
                        combined.append(f"Source: {result.get('title', 'Unknown')}")
                        combined.append(clean)
                        combined.append("")
        except Exception as e:
            print(f"  [!] Tavily search failed for '{query}': {e}")
            continue
    return "\n".join(combined)
def dive(seed: dict) -> list[dict]:
    """
    Dive into a seed domain:
    1. Search the web for information
    2. Extract abstract patterns via LLM
    3. Return list of pattern dicts
    Returns empty list on failure.
    """
    # Step 1: Search
    research = _search_seed(seed)
    if not research.strip():
        print(f"  [!] No search results for {seed['name']}")
        return []
    # Step 2: Extract patterns via LLM
    prompt = EXTRACT_PROMPT.format(domain=seed["name"])
    try:
        response = _gemini_model.generate_content(
            f"{prompt}\n\n--- RESEARCH MATERIAL ---\n\n{research}",
            generation_config={"max_output_tokens": 1500},
        )
        increment_llm_calls(1)
        raw_output = response.text if getattr(response, "text", None) else ""
        # Safety check on LLM output
        checked = check_llm_output(raw_output)
        if checked is None:
            print("  [!] LLM output failed safety check — skipping")
            return []
        # Parse JSON
        # Strip markdown code fences if present
        cleaned = checked.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        data = json.loads(cleaned)
        patterns = data.get("patterns", [])
        # Validate each pattern has required fields
        valid = []
        required = {"pattern_name", "description", "abstract_structure", "search_query"}
        for p in patterns:
            if required.issubset(p.keys()):
                valid.append(p)
        return valid
    except json.JSONDecodeError as e:
        print(f"  [!] Failed to parse LLM response as JSON: {e}")
        return []
    except Exception as e:
        print(f"  [!] LLM call failed: {e}")
        return []
