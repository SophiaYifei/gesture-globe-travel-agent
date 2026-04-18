"""
Evaluation module
Compares Non-DL vs DL agent across:
1. Perception accuracy (did we identify the region correctly?)
2. Itinerary relevance (do activities match the destination?)
3. Response time
4. Human-rated quality score simulation
"""

import time, json
from agents.non_dl_agent import NonDLAgent
from agents.dl_agent import DLAgent

# Ground truth test cases: (lat, lng, true_region, true_biome)
TEST_CASES = [
    (27.9881,  86.9250, "Nepal",        "mountain",      "alpine"),
    (-8.4095, 115.1889, "Bali",         "island",        "tropical"),
    (48.8566,   2.3522, "France",       "mixed",         "temperate"),
    (23.6345, -102.5528,"Mexico",       "coastal",       "diverse"),
    (64.9631, -19.0208, "Iceland",      "tundra",        "subarctic"),
    (-1.2921,  36.8219, "Kenya",        "savanna",       "tropical"),
    (37.9838,  23.7275, "Greece",       "coastal",       "mediterranean"),
    (-3.4653, -62.2159, "Brazil",       "rainforest",    "tropical"),
    (27.1751,  78.0421, "India",        "diverse",       "tropical"),
    (35.6762, 139.6503, "Japan",        "island",        "temperate"),
    (-33.8688, 151.2093,"Australia",    "outback",       "diverse"),
    (51.5074,  -0.1278, "USA Northeast","forest",        "temperate"),  # London — tests edge
]

ACTIVITY_KEYWORDS = {
    "mountain":     ["trek", "hike", "mountain", "monastery", "altitude", "glacier"],
    "tropical":     ["beach", "snorkel", "jungle", "coral", "dive", "palm"],
    "temperate":    ["museum", "castle", "cycling", "market", "countryside", "wine"],
    "mediterranean":["ruins", "boat", "vineyard", "old town", "island", "swim"],
    "subarctic":    ["aurora", "glacier", "whale", "hot spring", "dog sled", "midnight"],
    "arid":         ["desert", "camel", "dune", "oasis", "ruin", "canyon"],
    "rainforest":   ["amazon", "canopy", "jungle", "piranha", "river", "wildlife"],
    "savanna":      ["safari", "game drive", "lion", "balloon", "maasai", "wildlife"],
    "island":       ["island hop", "snorkel", "ferry", "surf", "beach", "dive"],
    "alpine":       ["trek", "cable car", "monastery", "acclimatize", "pass"],
}


def region_accuracy(predicted_name, predicted_biome, true_name, true_biome):
    name_match = true_name.lower() in predicted_name.lower() or predicted_name.lower() in true_name.lower()
    biome_match = predicted_biome.lower() == true_biome.lower()
    partial = any(word in predicted_biome.lower() for word in true_biome.lower().split())
    return {
        "name_match": name_match,
        "biome_exact": biome_match,
        "biome_partial": partial or biome_match,
        "score": (1.0 if name_match else 0.0) * 0.5 + (1.0 if biome_match else 0.5 if partial else 0.0) * 0.5,
    }


def itinerary_relevance(plan, true_biome):
    keywords = ACTIVITY_KEYWORDS.get(true_biome, [])
    if not keywords:
        return {"score": 0.5, "matched_keywords": [], "note": "No keywords for biome"}
    
    # Collect all text from plan
    all_text = ""
    if isinstance(plan.get("itinerary"), list):
        for day in plan["itinerary"]:
            all_text += " ".join(str(v) for v in day.values()).lower() + " "
    all_text += " ".join(plan.get("food", [])).lower()
    all_text += " ".join(plan.get("tips", [])).lower()

    matched = [kw for kw in keywords if kw in all_text]
    score = len(matched) / len(keywords)
    return {"score": score, "matched_keywords": matched, "total_keywords": len(keywords)}


def run_evaluation():
    non_dl = NonDLAgent()
    dl = DLAgent()

    results = {"non_dl": [], "dl": [], "summary": {}}

    for lat, lng, true_name, true_biome, true_climate in TEST_CASES:
        # Non-DL
        t0 = time.time()
        nd_result = non_dl.run(lat, lng)
        nd_time = time.time() - t0

        nd_region = nd_result["region"]
        nd_acc = region_accuracy(nd_region["name"], nd_region["biome"], true_name, true_biome)
        nd_rel = itinerary_relevance(nd_result["plan"], true_biome)

        results["non_dl"].append({
            "location": f"{true_name} ({lat:.1f}, {lng:.1f})",
            "predicted_region": nd_region["name"],
            "predicted_biome": nd_region["biome"],
            "true_region": true_name,
            "true_biome": true_biome,
            "accuracy": nd_acc,
            "relevance": nd_rel,
            "response_time_ms": round(nd_time * 1000),
        })

        # DL (without CLIP since it needs GPU + model download; uses geo fallback)
        t0 = time.time()
        dl_result = dl.run(lat, lng)
        dl_time = time.time() - t0

        dl_region = dl_result["region"]
        dl_acc = region_accuracy(dl_region["name"], dl_region["biome"], true_name, true_biome)
        dl_rel = itinerary_relevance(dl_result["plan"], true_biome)

        results["dl"].append({
            "location": f"{true_name} ({lat:.1f}, {lng:.1f})",
            "predicted_region": dl_region["name"],
            "predicted_biome": dl_region["biome"],
            "true_region": true_name,
            "true_biome": true_biome,
            "accuracy": dl_acc,
            "relevance": dl_rel,
            "response_time_ms": round(dl_time * 1000),
        })

    # Summary stats
    for agent_key in ["non_dl", "dl"]:
        agent_results = results[agent_key]
        avg_acc = sum(r["accuracy"]["score"] for r in agent_results) / len(agent_results)
        avg_rel = sum(r["relevance"]["score"] for r in agent_results) / len(agent_results)
        avg_time = sum(r["response_time_ms"] for r in agent_results) / len(agent_results)
        name_acc = sum(1 for r in agent_results if r["accuracy"]["name_match"]) / len(agent_results)
        results["summary"][agent_key] = {
            "avg_accuracy_score": round(avg_acc, 3),
            "name_match_rate": round(name_acc, 3),
            "avg_relevance_score": round(avg_rel, 3),
            "avg_response_time_ms": round(avg_time),
            "n_cases": len(agent_results),
        }

    return results
