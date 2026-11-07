"""Materialize assistant review annotations for the completed development trials.

These are explicit judgments made after reading each candidate, not a semantic
classifier. No API calls, new candidates, or training datasets are produced.
"""
import json
from pathlib import Path

from Scripts.DataGeneration.finetuning_pipeline import ROOT, canonical

# Candidate IDs are local to each preserved run. Unlisted candidates were read
# and judged to have defensible labels; mechanical exclusions are independent.
DECISIONS = {
    "contrastive-v2_20261107_111104_411342": {
        "semantic": {2: "Affordability, not minimum lookup", 4: "Affordability, not minimum lookup",
                     16: "Score lacks a credit or financial referent", 33: "Assistant's username, not user's name"},
        "quality": {42: "Unnatural old name Assistant in this clean pilot", 44: "Awkward redundant name-setting and name-query combination"}},
    "contrastive-v2_20261107_111125_628284": {
        "semantic": {2: "Minimum amount lacks a financial referent", 41: "Renames assistant, not user", 42: "Queries assistant name, not changing user name"},
        "quality": {4: "Assertion about minimum, not the requested clean lookup form"}},
    "contrastive-v3_20261107_111251_148642": {
        "semantic": {39: "Suggests a new assistant name rather than retrieving it; placeholder"},
        "quality": {25: "Invented artist attribution for a known song"}},
    "contrastive-v3_20261107_111311_624896": {
        "semantic": {17: "Generic limit lacks financial referent", 18: "Spending ceiling lacks credit/card specificity",
                     33: "Placeholder", 34: "Placeholder", 35: "Placeholder", 36: "Placeholder",
                     38: "Requests user form of address, not assistant name", 39: "Asks about user's name", 40: "Asks about user's name"},
        "quality": {4: "Unnatural utility card", 44: "Conflicting preferred names in one request"}},
    "contrastive-v4_20261107_111427_166454": {
        "semantic": {}, "quality": {39: "Awkward what-is-your-name-called construction"}},
    "contrastive-v4_20261107_111523_604425": {
        "semantic": {39: "Title can mean position rather than personal name"}, "quality": {}},
    "contrastive-v4_20261107_111630_987565": {
        "semantic": {38: "Approved account limit lacks credit/card referent"},
        "quality": {56: "Ambient sounds do not necessarily request music", 75: "Awkward name-called construction"}},
    "contrastive-v5_20261107_111919_691391": {"semantic": {}, "quality": {}},
    "contrastive-v5_20261107_112007_867916": {"semantic": {}, "quality": {}},
    "contrastive-v6_20261107_112830_013649": {"semantic": {}, "quality": {}},
    "contrastive-v6_20261107_112929_439400": {
        "semantic": {4: "How low is a payment does not clearly ask for the required minimum"}, "quality": {}},
    "contrastive-v6_20261107_113014_352351": {"semantic": {}, "quality": {}},
    "contrastive-v6_20261107_113057_997640": {
        "semantic": {4: "How low is a payment does not clearly ask for the required minimum",
                     95: "Asks the kind of music rather than title or performer"},
        "quality": {128: "Old and replacement name are identical; contradictory request"}},
    "contrastive-v5_20261107_112500_650638": {
        "semantic": {53: "Store-card credit does not specify credit limit", 89: "No clear music referent",
                     104: "What do you use for me lacks a name referent"},
        "quality": {115: "Unnatural go-by name construction"}},
    "contrastive-v5_20261107_112054_073069": {
        "semantic": {37: "Store-card credit can ask about balance/availability rather than limit",
                     61: "Playing in background has no clear music referent"},
        "quality": {8: "Minimum payment execution is benchmark-supported but excluded from the clean query pilot"}},
}


def main():
    summaries = []
    for name, decisions in DECISIONS.items():
        output = ROOT / "Data/CLINC150/finetuning_trials" / name
        rows = json.loads((output / "candidates.json").read_text(encoding="utf-8"))
        reviews = []
        for row in rows:
            cid = row["candidate_id"]
            reasons = []
            semantic = cid not in decisions["semantic"]
            if not semantic:
                reasons.append(decisions["semantic"][cid])
            if cid in decisions["quality"]:
                reasons.append(decisions["quality"][cid])
            reasons.extend(row["mechanical_flags"])
            reviews.append({"candidate_id": cid, "label_defensible": semantic,
                "provisionally_usable": not reasons,
                "reasons": reasons or ["Defensible target action and referent; no mechanical flags"],
                "reviewer": "assistant_first_pass_not_independent"})
        summary = {"run": name, "generated": len(rows),
            "label_defensible": sum(r["label_defensible"] for r in reviews),
            "provisionally_usable": sum(r["provisionally_usable"] for r in reviews),
            "mechanically_flagged": sum(bool(r["mechanical_flags"]) for r in rows),
            "short_at_most_8_words": sum(r["word_count"] <= 8 for r in rows),
            "training_approved": False}
        (output / "assistant_review.json").write_text(json.dumps({"summary": summary, "decisions": reviews}, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(summary)
    # Report cross-seed overlap without counting it as within-run duplication.
    v5 = [ROOT / "Data/CLINC150/finetuning_trials" / name / "candidates.json"
          for name in DECISIONS if name.startswith("contrastive-v6")]
    sets = [{canonical(r["text"]) for r in json.loads(p.read_text(encoding="utf-8"))} for p in v5]
    overlaps = [{"run_a": v5[i].parent.name, "run_b": v5[j].parent.name,
                 "shared_canonical_utterances": len(sets[i] & sets[j])}
                for i in range(len(v5)) for j in range(i + 1, len(v5))]
    result = {"review_type": "single assistant, not independent annotation",
              "trials": summaries, "final_version_cross_run_overlap": overlaps}
    (ROOT / "Experiments/finetuning_trial_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Cross-run overlap:", overlaps)


if __name__ == "__main__":
    main()
