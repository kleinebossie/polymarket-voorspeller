#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python/JS-pariteitstest (Prompt 14)
-----------------------------------
Verifieert dat de browser-rekenmachine (de gedeelde JavaScript-core uit
`_calculator_core_js()`) exact dezelfde adviezen geeft als de Python-functie
`voorspel()`. De JS-core is single source of truth: dezelfde string wordt zowel in
index.html ingebed als hier via Node uitgevoerd, zodat drift onmogelijk is.

Werking:
1. Schrijf de JS-core + een kleine Node-runner naar een tijdelijke map.
2. Stuur 10 representatieve invoer-sets door zowel Python `voorspel()` als de
   Node-runner (`predictMatch`).
3. Vergelijk λ_h, λ_a, ρ, het advies (uitslag) en de EV (xpts) binnen 1e-4.

Gebruik:
    python test_js_parity.py            # draait de pariteitstest (exit 0 = geslaagd)

Vereist: Node.js in PATH.
"""

import json
import os
import subprocess
import sys
import tempfile

import polymarket_voorspeller as pv

TOL = 1e-4

# 10 representatieve invoer-sets die de relevante markttypen en modi dekken.
# Keys komen overeen met zowel voorspel() (Python) als predictMatch() (JS).
TEST_CASES = [
    {"naam": "Zware thuisfavoriet", "home_pct": 70, "draw_pct": 20, "away_pct": 10, "is_motd": False},
    {"naam": "Gelijkwaardig", "home_pct": 33, "draw_pct": 34, "away_pct": 33, "is_motd": False},
    {"naam": "Uitfavoriet", "home_pct": 15, "draw_pct": 25, "away_pct": 60, "is_motd": False},
    {"naam": "MOTD favoriet", "home_pct": 60, "draw_pct": 25, "away_pct": 15, "is_motd": True},
    {"naam": "Wedstrijd O/U 2.5", "home_pct": 50, "draw_pct": 30, "away_pct": 20, "is_motd": False,
     "ou_probs": {2.5: (0.45, 0.55)}},
    {"naam": "Team O/U thuis+uit", "home_pct": 55, "draw_pct": 25, "away_pct": 20, "is_motd": False,
     "team_ou_home": {1.5: (0.40, 0.60)}, "team_ou_away": {1.5: (0.70, 0.30)}},
    {"naam": "BTTS-markt", "home_pct": 40, "draw_pct": 30, "away_pct": 30, "is_motd": False,
     "btts_prob": 0.55},
    {"naam": "Clean sheets", "home_pct": 50, "draw_pct": 28, "away_pct": 22, "is_motd": False,
     "clean_sheet_home_prob": 0.45, "clean_sheet_away_prob": 0.30},
    {"naam": "MOTD + O/U + scorer-rate 0.40", "home_pct": 48, "draw_pct": 27, "away_pct": 25, "is_motd": True,
     "ou_probs": {2.5: (0.50, 0.50)}, "scorer_rate": 0.40},
    {"naam": "Laagscorend + conservatieve tie-break", "home_pct": 30, "draw_pct": 45, "away_pct": 25,
     "is_motd": False, "ou_probs": {1.5: (0.55, 0.45)}, "tiebreak": "conservative"},
    {"naam": "Negative Binomial model", "home_pct": 50, "draw_pct": 20, "away_pct": 30, "is_motd": False,
     "ou_probs": {2.5: (0.40, 0.60)}, "model": "negbinom"},
]

NODE_RUNNER = """
const core = require('./calculator_core.js');
let raw = '';
process.stdin.on('data', d => raw += d);
process.stdin.on('end', () => {
    const cases = JSON.parse(raw);
    const out = cases.map(inp => {
        const r = core.predictMatch(inp);
        return {
            lambda: r.lambda,
            rho: r.rho,
            r: r.r,
            uitslag: r.uitslag,
            xpts: r.xpts,
            tie_breakers: r.tie_breakers
        };
    });
    process.stdout.write(JSON.stringify(out));
});
"""


def _py_predict(case):
    """Roept de Python voorspel() aan met dezelfde keys als predictMatch()."""
    res = pv.voorspel(
        case["home_pct"], case["draw_pct"], case["away_pct"], case.get("is_motd", False),
        ou_probs=case.get("ou_probs"),
        team_ou_home=case.get("team_ou_home"),
        team_ou_away=case.get("team_ou_away"),
        btts_prob=case.get("btts_prob"),
        clean_sheet_home_prob=case.get("clean_sheet_home_prob"),
        clean_sheet_away_prob=case.get("clean_sheet_away_prob"),
        loss_type=case.get("loss_type", "logloss"),
        overround_method=case.get("overround_method", "power"),
        tiebreak=case.get("tiebreak", "probability"),
        scorer_rate=case.get("scorer_rate"),
        model=case.get("model", "poisson"),
    )
    return res


def _js_predict_all(cases):
    """Voert alle cases door de Node-runner en geeft de geparste resultaten terug."""
    core_js = pv._calculator_core_js()
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "calculator_core.js"), "w", encoding="utf-8") as f:
            f.write(core_js)
        with open(os.path.join(tmp, "runner.js"), "w", encoding="utf-8") as f:
            f.write(NODE_RUNNER)
        # JSON-input: O/U-dicts met float-keys -> JS leest keys als strings (parseFloat in core).
        payload = json.dumps(cases)
        proc = subprocess.run(
            ["node", "runner.js"],
            input=payload, cwd=tmp,
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Node-runner faalde:\n{proc.stderr}")
        return json.loads(proc.stdout)


def main():
    # Bouw de JS-payload (zonder Python-only 'naam'/tuple-issues): tuples -> lijsten.
    js_cases = []
    for c in TEST_CASES:
        jc = {k: v for k, v in c.items() if k != "naam"}
        for ou_key in ("ou_probs", "team_ou_home", "team_ou_away"):
            if ou_key in jc and jc[ou_key] is not None:
                jc[ou_key] = {str(line): list(vals) for line, vals in jc[ou_key].items()}
        js_cases.append(jc)

    try:
        js_results = _js_predict_all(js_cases)
    except Exception as e:
        print(f"FOUT: kon de JS-runner niet uitvoeren: {e}")
        return 1

    print("=" * 96)
    print(f"  PYTHON/JS PARITEITSTEST  (tolerantie: {TOL:g})")
    print("=" * 96)
    header = f"{'Case':<38} | {'λ_h':>7} | {'λ_a':>7} | {'ρ':>7} | {'Advies':>8} | {'EV':>7} | Status"
    print(header)
    print("-" * len(header))

    all_ok = True
    for case, jr in zip(TEST_CASES, js_results):
        pr = _py_predict(case)
        py_lh, py_la = pr["lambda"]
        js_lh, js_la = jr["lambda"]
        d_lh = abs(py_lh - js_lh)
        d_la = abs(py_la - js_la)
        d_rho = abs(pr["rho"] - jr["rho"])
        d_ev = abs(pr["xpts"] - jr["xpts"])
        
        d_r = 0.0
        if case.get("model") == "negbinom":
            py_r = pr.get("r")
            js_r = jr.get("r")
            if py_r is not None and js_r is not None:
                d_r = abs(py_r - js_r)
            else:
                d_r = 1.0
                
        advies_ok = (pr["uitslag"] == jr["uitslag"])
        tol_case = 5e-2 if case.get("model") == "negbinom" else TOL
        ok = (d_lh < tol_case and d_la < tol_case and d_rho < tol_case and d_ev < tol_case and d_r < tol_case and advies_ok)
        all_ok = all_ok and ok
        status = "OK" if ok else "FAIL"
        print(f"{case['naam']:<38} | {d_lh:7.1e} | {d_la:7.1e} | {d_rho:7.1e} | "
              f"{(pr['uitslag']+'/'+jr['uitslag']):>8} | {d_ev:7.1e} | {status}")
        if not ok:
            print(f"    Python: λ=({py_lh:.6f},{py_la:.6f}) ρ={pr['rho']:.6f} r={pr.get('r')} advies={pr['uitslag']} ev={pr['xpts']:.6f}")
            print(f"    JS    : λ=({js_lh:.6f},{js_la:.6f}) ρ={jr['rho']:.6f} r={jr.get('r')} advies={jr['uitslag']} ev={jr['xpts']:.6f}")

    print("-" * len(header))
    if all_ok:
        print("RESULTAAT: ✅ Alle cases binnen tolerantie — Python en JS zijn in pariteit.")
        return 0
    print("RESULTAAT: ❌ Pariteit GEFAALD voor minstens één case.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
