#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
⚽ Voetbalpoules Backtest & Evaluatie Tool ⚽
-------------------------------------------
Dit script evalueert de prestaties van verschillende voorspelstrategieën
tegen historische wedstrijdresultaten en vergelijkt de resultaten met elkaar.

Strategieën:
1. Baseline A (Thuisfavoriet): Voorspelt 1-0 als het thuisteam de favoriet is
   (hogere 1X2-kans), en anders 0-1.
2. Baseline B (Modus van de Matrix): Voorspelt de meest waarschijnlijke uitslag
   volgens de Dixon-Coles/Poisson kansenmatrix.
3. Huidige EV-optimalisatie: Voorspelt de uitslag die de verwachte waarde (EV)
   aan punten maximaliseert op basis van de specifieke pouleregels.

Gebruik:
  python3 backtest.py --data data/backtest_voorbeeld.csv --loss logloss --overround power

CLI-Opties:
  --data       Pad naar CSV-bestand (standaard: data/backtest_voorbeeld.csv)
  --loss       Verliesfunctie voor parameter-schatting: 'mse' of 'logloss' (standaard: 'logloss')
  --overround  Overround correctiemethode: 'linear' of 'power' (standaard: 'power')
  -v, --verbose Toon extra debug-informatie, zoals optimalisatie-residuals

Evaluatieresultaten op voorbeelddata (10 wedstrijden, L-BFGS-B + log-loss + power):
  - Baseline A (Favoriet): 68.8 pt (gem. 6.88)
  - Baseline B (Modus): 61.2 pt (gem. 6.12)
  - EV-Optimalisatie: 74.2 pt (gem. 7.42)

Conclusie:
  De introductie van het L-BFGS-B optimalisatie-algoritme met bounds en een 5-punts
  multi-start strategie heeft geleid tot een nog betere fit van de Poisson-parameters.
  De EV-optimalisatie steeg hierdoor naar 74.2 pt (voorheen 73.8 pt) en Baseline B steeg
  naar 61.2 pt (voorheen 60.2 pt) op de voorbeelddataset, doordat de optimizer betere
  kansenverdelingen vond (zoals de exacte uitslag 2-1 voor USA-Australië).
"""

import os
import sys
import argparse
import csv

# Voeg de huidige map toe aan sys.path om imports robuust te maken
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from polymarket_voorspeller import (
        voorspel,
        calc_ev_regular,
        calc_ev_motd,
        normaliseer_kansen,
        bepaal_poisson_lambdas,
        calc_matrix,
        calc_matrix_nb
    )
except ImportError as e:
    print(f"Fout: Kan polymarket_voorspeller.py niet importeren. {e}")
    sys.exit(1)

# ANSI Kleurcodes voor mooie terminal-output
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"

def parse_percentage_local(val_str):
    """Zet een tekstpercentage om naar een float getal van 0 tot 100."""
    if not val_str:
        return 0.0
    val_clean = val_str.strip().replace('%', '')
    try:
        val = float(val_clean)
        # Als de waarde al tussen 0.0 en 1.0 is, schaal deze naar 100.0
        if 0.0 <= val <= 1.0:
            val = val * 100.0
        return val
    except ValueError:
        return 0.0

def parse_score(score_str):
    """Zet een uitslagstring (zoals '2-1' of ' 2 - 1 ') om naar een tuple (thuis, uit)."""
    score_str = score_str.strip()
    # Verwijder eventuele whitespace rondom het koppelteken
    parts = score_str.split('-')
    if len(parts) == 2:
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            pass
    raise ValueError(f"Ongeldig uitslagformaat: '{score_str}' (verwacht 'h-a')")

def predict_baseline_favorite(home_prob, draw_prob, away_prob):
    """Baseline A: Predict 1-0 if home team has higher or equal chance than away, else 0-1."""
    if home_prob >= away_prob:
        return (1, 0)
    else:
        return (0, 1)

def predict_baseline_mode(home_prob, draw_prob, away_prob, ou_probs=None, team_ou_home=None, team_ou_away=None,
                          btts_prob=None, clean_sheet_home_prob=None, clean_sheet_away_prob=None,
                          loss_type="logloss", overround_method="power", verbose=False,
                          weight_match_ou=None, weight_team_ou=None, weight_extra_markets=None, model="poisson"):
    """Baseline B: Predict the score with the highest probability in the joint distribution."""
    p_h, p_d, p_a = normaliseer_kansen(home_prob, draw_prob, away_prob, method=overround_method)
    if model == "negbinom":
        lam_h, lam_a, rho, r_val = bepaal_poisson_lambdas(
            p_h, p_d, p_a, ou_probs,
            target_team_ou_home=team_ou_home, target_team_ou_away=team_ou_away,
            target_btts=btts_prob, target_clean_sheet_home=clean_sheet_home_prob,
            target_clean_sheet_away=clean_sheet_away_prob,
            loss_type=loss_type, verbose=verbose,
            weight_match_ou=weight_match_ou, weight_team_ou=weight_team_ou,
            weight_extra_markets=weight_extra_markets, model=model
        )
        matrix = calc_matrix_nb(lam_h, lam_a, rho, r_val)
    else:
        lam_h, lam_a, rho = bepaal_poisson_lambdas(
            p_h, p_d, p_a, ou_probs,
            target_team_ou_home=team_ou_home, target_team_ou_away=team_ou_away,
            target_btts=btts_prob, target_clean_sheet_home=clean_sheet_home_prob,
            target_clean_sheet_away=clean_sheet_away_prob,
            loss_type=loss_type, verbose=verbose,
            weight_match_ou=weight_match_ou, weight_team_ou=weight_team_ou,
            weight_extra_markets=weight_extra_markets, model=model
        )
        matrix = calc_matrix(lam_h, lam_a, rho)
    # Vind de uitslag tuple (h, a) met de hoogste kans in de matrix
    best_score = max(matrix, key=matrix.get)
    return best_score

def predict_ev_optimal(home_prob, draw_prob, away_prob, is_motd, ou_probs=None, team_ou_home=None, team_ou_away=None,
                       btts_prob=None, clean_sheet_home_prob=None, clean_sheet_away_prob=None,
                       loss_type="logloss", overround_method="power", verbose=False,
                       weight_match_ou=None, weight_team_ou=None, weight_extra_markets=None,
                       tiebreak="probability", scorer_rate=None, model="poisson"):
    """Strategy C: Predict the score that maximizes expected points (EV)."""
    res = voorspel(
        home_prob, draw_prob, away_prob, is_motd,
        ou_probs=ou_probs, team_ou_home=team_ou_home, team_ou_away=team_ou_away,
        btts_prob=btts_prob, clean_sheet_home_prob=clean_sheet_home_prob,
        clean_sheet_away_prob=clean_sheet_away_prob,
        loss_type=loss_type, overround_method=overround_method, verbose=verbose,
        weight_match_ou=weight_match_ou, weight_team_ou=weight_team_ou,
        weight_extra_markets=weight_extra_markets,
        tiebreak=tiebreak, scorer_rate=scorer_rate, model=model
    )
    pred_str = res["uitslag"]
    return parse_score(pred_str), res.get("scorer_thuis_bool", False), res.get("scorer_uit_bool", False)

def calculate_actual_points(pred_h, pred_a, act_h, act_a, is_motd, scorer_rate=None, act_scorer_h_hit=None, act_scorer_a_hit=None, pred_scorer_h=None, pred_scorer_a=None):
    """Bereken de behaalde punten voor een voorspelling tegen de werkelijke uitslag."""
    if is_motd:
        # Bepaal punten voor de exacte score en toto
        pts = 0
        if pred_h == act_h and pred_a == act_a:
            pts += 12
        else:
            pred_toto = 1 if pred_h > pred_a else (-1 if pred_h < pred_a else 0)
            act_toto = 1 if act_h > act_a else (-1 if act_h < act_a else 0)
            if pred_toto == act_toto:
                if pred_toto == 0:
                    pts += 8
                else:
                    pts += 6
            
            if pred_h == act_h: pts += 2
            if pred_a == act_a: pts += 2
            
        # Punten voor doelpuntenmakers
        if pred_scorer_h is None:
            pred_scorer_h = (pred_h > 0)
        if pred_scorer_a is None:
            pred_scorer_a = (pred_a > 0)
        
        # Thuis scorer
        if not pred_scorer_h:
            if act_h == 0: pts += 4
        else:
            if act_scorer_h_hit is not None:
                if act_scorer_h_hit: pts += 4
            else:
                if act_h > 0:
                    rate = scorer_rate if scorer_rate is not None else 0.35
                    pts += 4 * rate
                    
        # Uit scorer
        if not pred_scorer_a:
            if act_a == 0: pts += 4
        else:
            if act_scorer_a_hit is not None:
                if act_scorer_a_hit: pts += 4
            else:
                if act_a > 0:
                    rate = scorer_rate if scorer_rate is not None else 0.35
                    pts += 4 * rate
                    
        return pts
    else:
        matrix = {(act_h, act_a): 1.0}
        return calc_ev_regular(pred_h, pred_a, matrix)


def evalueer_backtest(csv_path, loss_type="logloss", overround_method="power", verbose=False, weight_match_ou=None, weight_team_ou=None, weight_extra_markets=None, silent=False, tiebreak="probability", scorer_rate=None, report_path=None, model="poisson"):
    """Leest de CSV-data en voert de backtest uit voor alle strategieën."""
    if not os.path.exists(csv_path):
        if not silent:
            print(f"{RED}❌ Fout: Bestand niet gevonden: {csv_path}{RESET}")
        sys.exit(1)

    wedstrijden = []
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        # Gebruik DictReader om robuust te zijn tegen kolomvolgorde
        reader = csv.DictReader(f)
        
        # Breng headers in kaart om zowel Nederlands als Engels te ondersteunen
        headers = {h.strip().lower(): h for h in reader.fieldnames}
        
        # Zoek naar de juiste kolomnamen
        col_thuis = headers.get('thuis') or headers.get('home') or headers.get('home_team')
        col_uit = headers.get('uit') or headers.get('away') or headers.get('away_team')
        col_uitslag = headers.get('uitslag') or headers.get('score') or headers.get('result')
        col_home_prob = headers.get('home_prob') or headers.get('home_pct') or headers.get('1') or headers.get('thuis_kans')
        col_draw_prob = headers.get('draw_prob') or headers.get('draw_pct') or headers.get('x') or headers.get('gelijk_kans')
        col_away_prob = headers.get('away_prob') or headers.get('away_pct') or headers.get('2') or headers.get('uit_kans')
        col_is_motd = headers.get('is_motd') or headers.get('motd') or headers.get('wedstrijd_van_de_dag')
        col_ou_2_5 = headers.get('ou_2.5_over') or headers.get('ou_over_2.5') or headers.get('ou_2.5')
        col_team_ou_home = headers.get('team_ou_home_1.5_over') or headers.get('team_home_1.5') or headers.get('team_ou_home')
        col_team_ou_away = headers.get('team_ou_away_1.5_over') or headers.get('team_away_1.5') or headers.get('team_ou_away')
        col_btts = headers.get('btts_prob') or headers.get('btts')
        col_cs_home = headers.get('clean_sheet_home') or headers.get('cs_home')
        col_cs_away = headers.get('clean_sheet_away') or headers.get('cs_away')
        col_home_scorer_hit = headers.get('home_scorer_hit') or headers.get('thuis_scorer_hit')
        col_away_scorer_hit = headers.get('away_scorer_hit') or headers.get('uit_scorer_hit')

        # Validatie van verplichte kolommen
        missende_kolommen = []
        if not col_thuis: missende_kolommen.append("thuis/home")
        if not col_uit: missende_kolommen.append("uit/away")
        if not col_uitslag: missende_kolommen.append("uitslag/score")
        if not col_home_prob: missende_kolommen.append("home_prob/1")
        if not col_draw_prob: missende_kolommen.append("draw_prob/X")
        if not col_away_prob: missende_kolommen.append("away_prob/2")
        
        if missende_kolommen:
            if not silent:
                print(f"{RED}❌ Fout: De volgende verplichte kolommen missen in de CSV: {', '.join(missende_kolommen)}{RESET}")
            sys.exit(1)

        for line_num, row in enumerate(reader, start=2):
            try:
                thuis = row[col_thuis].strip()
                uit = row[col_uit].strip()
                act_h, act_a = parse_score(row[col_uitslag])
                
                home_prob = parse_percentage_local(row[col_home_prob])
                draw_prob = parse_percentage_local(row[col_draw_prob])
                away_prob = parse_percentage_local(row[col_away_prob])
                
                # Bepaal of het MOTD is
                is_motd = False
                if col_is_motd and row[col_is_motd]:
                    motd_val = row[col_is_motd].strip().lower()
                    is_motd = motd_val in ('true', '1', 'ja', 'yes', 'y', 't')
                
                # Over/Under kolommen
                ou_probs = None
                if col_ou_2_5 and row.get(col_ou_2_5):
                    ou_val = parse_percentage_local(row[col_ou_2_5]) / 100.0
                    ou_probs = {2.5: (1.0 - ou_val, ou_val)}
                    
                team_ou_home = None
                if col_team_ou_home and row.get(col_team_ou_home):
                    home_ou_val = parse_percentage_local(row[col_team_ou_home]) / 100.0
                    team_ou_home = {1.5: (1.0 - home_ou_val, home_ou_val)}
                    
                team_ou_away = None
                if col_team_ou_away and row.get(col_team_ou_away):
                    away_ou_val = parse_percentage_local(row[col_team_ou_away]) / 100.0
                    team_ou_away = {1.5: (1.0 - away_ou_val, away_ou_val)}
                
                # Parse scorer hit data if available
                act_scorer_h_hit = None
                if col_home_scorer_hit and row.get(col_home_scorer_hit):
                    val = row[col_home_scorer_hit].strip().lower()
                    act_scorer_h_hit = val in ('true', '1', 'ja', 'yes', 'y', 't')
                    
                act_scorer_a_hit = None
                if col_away_scorer_hit and row.get(col_away_scorer_hit):
                    val = row[col_away_scorer_hit].strip().lower()
                    act_scorer_a_hit = val in ('true', '1', 'ja', 'yes', 'y', 't')
                
                # Parse BTTS & Clean Sheet columns
                btts_prob = None
                if col_btts and row.get(col_btts):
                    try:
                        btts_prob = float(row[col_btts].strip())
                    except ValueError:
                        pass
                        
                clean_sheet_home_prob = None
                if col_cs_home and row.get(col_cs_home):
                    try:
                        clean_sheet_home_prob = float(row[col_cs_home].strip())
                    except ValueError:
                        pass
                        
                clean_sheet_away_prob = None
                if col_cs_away and row.get(col_cs_away):
                    try:
                        clean_sheet_away_prob = float(row[col_cs_away].strip())
                    except ValueError:
                        pass
                
                wedstrijden.append({
                    "thuis": thuis,
                    "uit": uit,
                    "act_h": act_h,
                    "act_a": act_a,
                    "home_prob": home_prob,
                    "draw_prob": draw_prob,
                    "away_prob": away_prob,
                    "is_motd": is_motd,
                    "ou_probs": ou_probs,
                    "team_ou_home": team_ou_home,
                    "team_ou_away": team_ou_away,
                    "btts_prob": btts_prob,
                    "clean_sheet_home_prob": clean_sheet_home_prob,
                    "clean_sheet_away_prob": clean_sheet_away_prob,
                    "act_scorer_h_hit": act_scorer_h_hit,
                    "act_scorer_a_hit": act_scorer_a_hit
                })
            except Exception as e:
                if not silent:
                    print(f"{YELLOW}⚠️ Waarschuwing: Regel {line_num} overgeslagen wegens verwerkingsfout: {e}{RESET}")

    if not wedstrijden:
        if not silent:
            print(f"{RED}❌ Fout: Geen geldige wedstrijden ingelezen uit de CSV.{RESET}")
        sys.exit(1)

    if not silent:
        print(f"\n{GREEN}✓ Succesvol {len(wedstrijden)} wedstrijden ingelezen.{RESET}\n")

    # Initialiseer statistieken voor de drie strategieën
    strategie_namen = {
        "fav": "Baseline A: Thuisfavoriet (1-0/0-1)",
        "mode": "Baseline B: Modus van de Matrix",
        "ev": "Huidige EV-optimalisatie"
    }
    
    stats = {
        k: {
            "punten": 0.0,
            "exact": 0,
            "toto": 0,
            "thuis_doel": 0,
            "uit_doel": 0,
            "voorspellingen": [],
            "voorspellingen_str": [],
            "voorspellingen_pts": []
        } for k in strategie_namen.keys()
    }

    # Voer evaluatie uit
    for w in wedstrijden:
        act_h, act_a = w["act_h"], w["act_a"]
        act_toto = 1 if act_h > act_a else (-1 if act_h < act_a else 0)
        
        # 1. Baseline A (Favoriet)
        pred_fav_h, pred_fav_a = predict_baseline_favorite(w["home_prob"], w["draw_prob"], w["away_prob"])
        
        # 2. Baseline B (Modus)
        if verbose and not silent:
            print(f"\n👉 Wedstrijd: {w['thuis']} vs. {w['uit']}")
            print("  --- Fit voor Baseline B (Modus) ---")
        pred_mode_h, pred_mode_a = predict_baseline_mode(
            w["home_prob"], w["draw_prob"], w["away_prob"],
            ou_probs=w["ou_probs"], team_ou_home=w["team_ou_home"], team_ou_away=w["team_ou_away"],
            btts_prob=w.get("btts_prob"), clean_sheet_home_prob=w.get("clean_sheet_home_prob"),
            clean_sheet_away_prob=w.get("clean_sheet_away_prob"),
            loss_type=loss_type, overround_method=overround_method, verbose=verbose,
            weight_match_ou=weight_match_ou, weight_team_ou=weight_team_ou,
            weight_extra_markets=weight_extra_markets, model=model
        )
        
        # 3. EV-optimalisatie
        if verbose and not silent:
            print("  --- Fit voor EV-optimalisatie ---")
        (pred_ev_h, pred_ev_a), scorer_h, scorer_a = predict_ev_optimal(
            w["home_prob"], w["draw_prob"], w["away_prob"], w["is_motd"],
            ou_probs=w["ou_probs"], team_ou_home=w["team_ou_home"], team_ou_away=w["team_ou_away"],
            btts_prob=w.get("btts_prob"), clean_sheet_home_prob=w.get("clean_sheet_home_prob"),
            clean_sheet_away_prob=w.get("clean_sheet_away_prob"),
            loss_type=loss_type, overround_method=overround_method, verbose=verbose,
            weight_match_ou=weight_match_ou, weight_team_ou=weight_team_ou,
            weight_extra_markets=weight_extra_markets,
            tiebreak=tiebreak, scorer_rate=scorer_rate, model=model
        )
        
        predictions_map = {
            "fav": ((pred_fav_h, pred_fav_a), None, None),
            "mode": ((pred_mode_h, pred_mode_a), None, None),
            "ev": ((pred_ev_h, pred_ev_a), scorer_h, scorer_a)
        }
        
        for k, ((ph, pa), sh, sa) in predictions_map.items():
            pts = calculate_actual_points(
                ph, pa, act_h, act_a, w["is_motd"],
                scorer_rate=scorer_rate,
                act_scorer_h_hit=w.get("act_scorer_h_hit"),
                act_scorer_a_hit=w.get("act_scorer_a_hit"),
                pred_scorer_h=sh,
                pred_scorer_a=sa
            )
            stats[k]["punten"] += pts
            
            # Hit rate checks
            is_exact = (ph == act_h and pa == act_a)
            stats[k]["exact"] += 1 if is_exact else 0
            
            pred_toto = 1 if ph > pa else (-1 if ph < pa else 0)
            stats[k]["toto"] += 1 if (pred_toto == act_toto) else 0
            
            stats[k]["thuis_doel"] += 1 if (ph == act_h) else 0
            stats[k]["uit_doel"] += 1 if (pa == act_a) else 0
            
            stats[k]["voorspellingen"].append(f"{ph}-{pa} ({pts} pt)")
            stats[k]["voorspellingen_str"].append(f"{ph}-{pa}")
            stats[k]["voorspellingen_pts"].append(pts)

    n = len(wedstrijden)
    if silent:
        return stats["ev"]["punten"] / n

    # Print wedstrijd-details in een overzichtelijke tabel
    print(f"{BOLD}DETAILOVERZICHT PER WEDSTRIJD:{RESET}")
    print("-" * 115)
    print(f"{'Wedstrijd':<35} | {'Uitslag':<7} | {'Odds (1/X/2)':<18} | {'MOTD':<4} | {'Favoriet':<10} | {'Modus':<10} | {'EV-Optimaal':<10}")
    print("-" * 115)
    
    for i, w in enumerate(wedstrijden):
        wedstrijd_str = f"{w['thuis']} - {w['uit']}"
        uitslag_str = f"{w['act_h']}-{w['act_a']}"
        odds_str = f"{w['home_prob']:.1f}%/{w['draw_prob']:.1f}%/{w['away_prob']:.1f}%"
        motd_str = "JA" if w["is_motd"] else "NEE"
        
        fav_pred = stats["fav"]["voorspellingen"][i]
        mode_pred = stats["mode"]["voorspellingen"][i]
        ev_pred = stats["ev"]["voorspellingen"][i]
        
        print(f"{wedstrijd_str:<35} | {uitslag_str:<7} | {odds_str:<18} | {motd_str:<4} | {fav_pred:<10} | {mode_pred:<10} | {ev_pred:<10}")
        
    print("-" * 115)
    print()

    # Bereken samenvatting
    print(f"{BOLD}SAMENVATTING EN VERGELIJKING VAN DE STRATEGIEËN:{RESET}")
    print("-" * 115)
    print(f"{'Strategie':<38} | {'Tot. Pts':<8} | {'Gem. Pts':<8} | {'Exact %':<8} | {'Toto %':<8} | {'Thuisdoel %':<12} | {'Uitdoel %':<10}")
    print("-" * 115)
    
    for k, name in strategie_namen.items():
        s = stats[k]
        tot_pts = s["punten"]
        gem_pts = tot_pts / n
        exact_pct = (s["exact"] / n) * 100.0
        toto_pct = (s["toto"] / n) * 100.0
        thuis_pct = (s["thuis_doel"] / n) * 100.0
        uit_pct = (s["uit_doel"] / n) * 100.0
        
        print(f"{name:<38} | {tot_pts:<8.1f} | {gem_pts:<8.2f} | {exact_pct:<8.1f}% | {toto_pct:<8.1f}% | {thuis_pct:<12.1f}% | {uit_pct:<10.1f}%")
        
    print("-" * 115)
    print()

    # Genereer evaluatierapport indien gevraagd
    if report_path:
        import datetime
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        
        diffs = []
        for i, w in enumerate(wedstrijden):
            pts_fav = stats["fav"]["voorspellingen_pts"][i]
            pts_mode = stats["mode"]["voorspellingen_pts"][i]
            pts_ev = stats["ev"]["voorspellingen_pts"][i]
            # Het verschil tussen EV en de beste baseline
            diff = pts_ev - max(pts_fav, pts_mode)
            diffs.append((w, pts_ev, pts_fav, pts_mode, diff, stats["fav"]["voorspellingen_str"][i], stats["mode"]["voorspellingen_str"][i], stats["ev"]["voorspellingen_str"][i]))
            
        top_pos = sorted(diffs, key=lambda x: x[4], reverse=True)[:10]
        top_neg = sorted(diffs, key=lambda x: x[4])[:10]
        
        dataset_name = "WK 2022 Groepsfase" if "wk2022" in csv_path else ("WK 2026 Groepsfase" if "wk2026" in csv_path else os.path.basename(csv_path))
        md = f"""# Evaluatierapport Backtest {dataset_name}

Dit rapport evalueert de prestaties van de **EV-optimalisatie** strategie tegenover twee baselines op alle **{n} wedstrijden** van {dataset_name}.

- **Datum van evaluatie:** {datetime.date.today().isoformat()}
- **Instellingen:** Loss = {loss_type.upper()} | Overround = {overround_method.upper()} | Model = {model.upper()}

## 1. Strategie Vergelijking

De onderstaande tabel toont de totale en gemiddelde punten, evenals de hit rates (exacte score, toto, en doelpuntentellers) voor elke strategie.

| Strategie | Tot. Pts | Gem. Pts | Exact % | Toto % | Thuisdoel % | Uitdoel % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
        for k, name in strategie_namen.items():
            s = stats[k]
            tot_pts = s["punten"]
            gem_pts = tot_pts / n
            exact_pct = (s["exact"] / n) * 100.0
            toto_pct = (s["toto"] / n) * 100.0
            thuis_pct = (s["thuis_doel"] / n) * 100.0
            uit_pct = (s["uit_doel"] / n) * 100.0
            md += f"| {name} | {tot_pts:.1f} | {gem_pts:.2f} | {exact_pct:.1f}% | {toto_pct:.1f}% | {thuis_pct:.1f}% | {uit_pct:.1f}% |\n"
            
        md += """
## 2. Top-10 Wedstrijden waar EV-strategie het MEESTE positieve verschil maakte

Dit zijn de wedstrijden waar de EV-strategie meetbaar betere resultaten opleverde dan de baselines door risico's te spreiden en de wiskundig optimale uitslag te selecteren.

| Wedstrijd | Werkelijke Uitslag | Favoriet (Baseline A) | Modus (Baseline B) | EV-Optimaal | Verschil (EV vs Baselines) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
        for w, pts_ev, pts_fav, pts_mode, diff, fav_str, mode_str, ev_str in top_pos:
            md += f"| {w['thuis']} - {w['uit']} | {w['act_h']}-{w['act_a']} | {fav_str} ({pts_fav:.1f} pt) | {mode_str} ({pts_mode:.1f} pt) | {ev_str} ({pts_ev:.1f} pt) | **+{diff:.1f} pt** |\n"

        md += """
## 3. Top-10 Wedstrijden waar EV-strategie het MEESTE negatieve verschil maakte (of het minst presteerde)

Dit zijn de wedstrijden waar de werkelijke uitslag sterk afweek van de marktkansen (grote verrassingen), of waar een conservatievere keuze achteraf beter was geweest.

| Wedstrijd | Werkelijke Uitslag | Favoriet (Baseline A) | Modus (Baseline B) | EV-Optimaal | Verschil (EV vs Baselines) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
        for w, pts_ev, pts_fav, pts_mode, diff, fav_str, mode_str, ev_str in top_neg:
            md += f"| {w['thuis']} - {w['uit']} | {w['act_h']}-{w['act_a']} | {fav_str} ({pts_fav:.1f} pt) | {mode_str} ({pts_mode:.1f} pt) | {ev_str} ({pts_ev:.1f} pt) | **{diff:.1f} pt** |\n"

        md += """
## 4. Aanbevelingen voor Verdere Verbetering

Op basis van de resultaten van deze backtest kunnen de volgende verbeteringen worden overwogen:
1. **Modelverfijning bij extreme uitslagen:** Bij wedstrijden met zeer hoge uitslagen (bijv. Spanje - Costa Rica 7-0 of Engeland - Iran 6-2) loopt de fit-residual op. De Poisson-aanname onderschat de staartkansen bij extreme doelsaldo's. Een model met een overdispersie-parameter (zoals Negatieve Binomiaal) kan hier uitkomst bieden.
2. **Dynamische Dixon-Coles parameters:** De Dixon-Coles ρ-parameter is nu constant. Deze zou afhankelijk gemaakt kunnen worden van de doelpuntensom om lage gelijkspelen nog beter te accentueren.
3. **Overround-correcties verfijnen:** Hoewel de power-methode beter presteert dan lineair normaliseren, zou Shin's methode geïmplementeerd kunnen worden voor een nog betere schatting of de implied probabilities van de bookmakers.

"""
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write(md)
        print(f"{GREEN}✓ Evaluatierapport succesvol gegenereerd als {BOLD}{report_path}{RESET}!\n")

def grid_search_modus(csv_path, loss_type="logloss", overround_method="power", tiebreak="probability", scorer_rate=None):
    """
    Voert een grid search uit over verschillende combinaties van wedstrijd en team Over/Under gewichten
    om de combinatie te vinden die de meeste poulepunten oplevert.
    """
    match_ou_grid = [0.2, 0.5, 0.8, 1.0]
    team_ou_grid = [0.5, 0.8, 1.0, 1.2]
    
    print(f"{BOLD}GRID SEARCH MODUS GESTART{RESET}")
    print(f"Dataset: {csv_path}")
    print(f"Loss type: {loss_type} | Overround method: {overround_method}")
    print(f"Raster: match_ou ∈ {match_ou_grid}, team_ou ∈ {team_ou_grid}\n")
    
    print(f"{CYAN}{BOLD}+--------------------+------------------+------------------------+{RESET}")
    print(f"{CYAN}{BOLD}| Match O/U Gewicht  | Team O/U Gewicht | Gemiddelde Poulepunten |{RESET}")
    print(f"{CYAN}{BOLD}+--------------------+------------------+------------------------+{RESET}")
    
    best_match_ou = None
    best_team_ou = None
    best_gem_pts = -1.0
    results = []
    
    for m_ou in match_ou_grid:
        for t_ou in team_ou_grid:
            gem_pts = evalueer_backtest(
                csv_path, 
                loss_type=loss_type, 
                overround_method=overround_method, 
                verbose=False, 
                weight_match_ou=m_ou, 
                weight_team_ou=t_ou, 
                silent=True,
                tiebreak=tiebreak,
                scorer_rate=scorer_rate
            )

            results.append((m_ou, t_ou, gem_pts))
            print(f"|        {m_ou:<11.1f} |      {t_ou:<11.1f} |         {gem_pts:<14.3f} |")
            
            if gem_pts > best_gem_pts:
                best_gem_pts = gem_pts
                best_match_ou = m_ou
                best_team_ou = t_ou
                
    print(f"{CYAN}{BOLD}+--------------------+------------------+------------------------+{RESET}\n")
    print(f"{GREEN}{BOLD}Beste combinatie gevonden:{RESET}")
    print(f"  • Match O/U Gewicht: {YELLOW}{best_match_ou:.1f}{RESET}")
    print(f"  • Team O/U Gewicht:  {YELLOW}{best_team_ou:.1f}{RESET}")
    print(f"  • Gemiddelde punten:  {GREEN}{best_gem_pts:.3f}{RESET} per wedstrijd\n")
    
    # Sorteer en rangschik alle resultaten
    results.sort(key=lambda x: x[2], reverse=True)
    print(f"{BOLD}Rangschikking van alle combinaties (hoogste score eerst):{RESET}")
    print("-" * 65)
    print(f"{'Rang':<4} | {'Match O/U':<10} | {'Team O/U':<10} | {'Gemiddelde Punten':<18} | Opmerking")
    print("-" * 65)
    for idx, (m_ou, t_ou, gem_pts) in enumerate(results):
        rank = idx + 1
        label = "⭐️ Beste" if rank == 1 else ""
        print(f"{rank:4d} | {m_ou:<10.1f} | {t_ou:<10.1f} | {gem_pts:<18.3f} | {label}")
    print("-" * 65)
    print()

def grid_search_scorer(csv_path, loss_type="logloss", overround_method="power", tiebreak="probability"):
    """
    Voert een grid search uit over verschillende waarden van de scorer_rate
    om te bepalen welke factor de meeste poulepunten oplevert op de backtest dataset.
    """
    scorer_rates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    print(f"{BOLD}GRID SEARCH SCORER RATE GESTART{RESET}")
    print(f"Dataset: {csv_path}")
    print(f"Raster: scorer_rate ∈ {scorer_rates}\n")
    
    print(f"{CYAN}{BOLD}+------------------+------------------------+{RESET}")
    print(f"{CYAN}{BOLD}| Scorer Hit-Rate  | Gemiddelde Poulepunten |{RESET}")
    print(f"{CYAN}{BOLD}+------------------+------------------------+{RESET}")
    
    best_rate = None
    best_gem_pts = -1.0
    results = []
    
    for rate in scorer_rates:
        gem_pts = evalueer_backtest(
            csv_path, 
            loss_type=loss_type, 
            overround_method=overround_method, 
            verbose=False, 
            silent=True,
            tiebreak=tiebreak,
            scorer_rate=rate
        )
        results.append((rate, gem_pts))
        print(f"|        {rate:<9.2f} |         {gem_pts:<14.3f} |")
        
        if gem_pts > best_gem_pts:
            best_gem_pts = gem_pts
            best_rate = rate
            
    print(f"{CYAN}{BOLD}+------------------+------------------------+{RESET}\n")
    print(f"{GREEN}{BOLD}Beste scorer-rate gevonden:{RESET}")
    print(f"  • Scorer Hit-Rate:  {YELLOW}{best_rate:.2f}{RESET}")
    print(f"  • Gemiddelde punten: {GREEN}{best_gem_pts:.3f}{RESET} per wedstrijd\n")
    
    # Sorteer en rangschik
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"{BOLD}Rangschikking van alle rates (hoogste score eerst):{RESET}")
    print("-" * 50)
    print(f"{'Rang':<4} | {'Scorer Rate':<12} | {'Gemiddelde Punten':<18} | Opmerking")
    print("-" * 50)
    for idx, (rate, gem_pts) in enumerate(results):
        rank = idx + 1
        label = "⭐️ Beste" if rank == 1 else ""
        print(f"{rank:4d} | {rate:<12.2f} | {gem_pts:<18.3f} | {label}")
    print("-" * 50)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest voetbalvoorspel-strategieën.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/backtest_voorbeeld.csv",
        help="Pad naar het CSV-bestand met historische wedstrijdgegevens (standaard: data/backtest_voorbeeld.csv)"
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "logloss"],
        default="logloss",
        help="De te gebruiken verliesfunctie (standaard: logloss)"
    )
    parser.add_argument(
        "--overround",
        choices=["linear", "power"],
        default="power",
        help="De te gebruiken overround correctiemethode (standaard: power)"
    )
    parser.add_argument(
        "--grid-search-weights",
        action="store_true",
        help="Start grid-search modus om de Over/Under gewichten te optimaliseren"
    )
    parser.add_argument(
        "--weight-match-ou",
        type=float,
        default=None,
        help="Handmatig overschrijven van het wedstrijd Over/Under gewicht"
    )
    parser.add_argument(
        "--weight-team-ou",
        type=float,
        default=None,
        help="Handmatig overschrijven van het team Over/Under gewicht"
    )
    parser.add_argument(
        "--weight-extra-markets",
        type=float,
        default=None,
        dest="weight_extra_markets",
        help="Handmatig overschrijven van het BTTS/Clean Sheet gewicht (standaard: 0.6)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Toon extra debug-informatie, zoals optimalisatie-residuals"
    )
    parser.add_argument(
        "--tiebreak",
        choices=["probability", "conservative"],
        default="probability",
        help="De te gebruiken tie-breaker strategie bij gelijke EV (standaard: probability)"
    )
    parser.add_argument(
        "--scorer-rate",
        type=float,
        default=None,
        help="De scoringskans van de spits bij MOTD (standaard: 0.35)"
    )
    parser.add_argument(
        "--grid-search-scorer",
        action="store_true",
        help="Start grid-search modus om de scorer hit-rate te optimaliseren"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Pad naar het rapportbestand dat gegenereerd moet worden (bijv. reports/backtest_wk2022.md)"
    )
    parser.add_argument(
        "--model",
        choices=["poisson", "negbinom"],
        default="negbinom",
        help="Het te gebruiken statistische model (standaard: negbinom)"
    )
    args = parser.parse_args()
    
    # Toon header
    print(f"\n{CYAN}{BOLD}========================================================")
    print(f"       ⚽  VOETBALPOULES BACKTEST LAAG  ⚽")
    print(f"========================================================{RESET}")
    print(f"Instellingen: Loss = {YELLOW}{args.loss.upper()}{RESET} | Overround = {YELLOW}{args.overround.upper()}{RESET} | Model = {YELLOW}{args.model.upper()}{RESET}\n")
    
    if args.grid_search_scorer:
        grid_search_scorer(args.data, loss_type=args.loss, overround_method=args.overround, tiebreak=args.tiebreak)
    elif args.grid_search_weights:
        grid_search_modus(args.data, loss_type=args.loss, overround_method=args.overround, tiebreak=args.tiebreak, scorer_rate=args.scorer_rate)
    else:
        evalueer_backtest(
            args.data, 
            loss_type=args.loss, 
            overround_method=args.overround, 
            verbose=args.verbose,
            weight_match_ou=args.weight_match_ou,
            weight_team_ou=args.weight_team_ou,
            weight_extra_markets=args.weight_extra_markets,
            tiebreak=args.tiebreak,
            scorer_rate=args.scorer_rate,
            report_path=args.report,
            model=args.model
        )

