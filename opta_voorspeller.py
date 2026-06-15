#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voetbalpoules Opta Voorspeller
------------------------------
Deze tool berekent de wiskundig optimale uitslag voor voetbalpoules op basis van
de Opta Analyst 1X2 winstkansen, volgens het onderzoeksrapport:
'Optimalisatie van Expected Points (xPts) in Voetbalpoules'.
"""

import sys
import argparse
import requests
import json
import re
import math
import math

# ANSI Kleurcodes voor een mooie vormgeving in de terminal (zonder extra pakketten!)
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"

def print_header():
    print(f"\n{CYAN}{BOLD}========================================================")
    print(f"       ⚽  VOETBALPOULES OPTA VOORSPELLER  ⚽")
    print(f"========================================================{RESET}")
    print("Dit programma berekent de wiskundig beste uitslag om de")
    print("meeste punten (Expected Points) te behalen in je poule.")
    print("--------------------------------------------------------\n")


def parse_percentage(val_str):
    val_str = val_str.strip().replace('%', '')
    try:
        val = float(val_str)
        if 0.0 <= val <= 1.0:
            val = val * 100.0
        return val
    except ValueError:
        raise ValueError(f"Ongeldig getal: '{val_str}'")

def normaliseer_kansen(home, draw, away):
    totaal = home + draw + away
    if totaal == 0:
        return 0.0, 0.0, 0.0
    return home / totaal, draw / totaal, away / totaal

def converteer_utc_naar_nl(utc_str):
    if not utc_str:
        return ""
    try:
        from zoneinfo import ZoneInfo
        import datetime
        clean_str = utc_str.replace('Z', '+00:00')
        dt_utc = datetime.datetime.fromisoformat(clean_str)
        dt_nl = dt_utc.astimezone(ZoneInfo("Europe/Amsterdam"))
        return dt_nl.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return utc_str.replace('T', ' ')[:16]


def poisson(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def calc_matrix(lam_h, lam_a):
    matrix = {}
    for h in range(10):
        for a in range(10):
            matrix[(h, a)] = poisson(lam_h, h) * poisson(lam_a, a)
    return matrix

def get_1x2_and_ou(matrix):
    h_win = sum(p for (h,a), p in matrix.items() if h > a)
    d = sum(p for (h,a), p in matrix.items() if h == a)
    a_win = sum(p for (h,a), p in matrix.items() if h < a)
    return h_win, d, a_win

def bepaal_poisson_lambdas(target_h, target_d, target_a, target_ou=None):
    best_lam_h, best_lam_a = 0.1, 0.1
    best_error = float('inf')
    
    for lh in [x/100.0 for x in range(10, 400, 5)]:
        for la in [x/100.0 for x in range(10, 400, 5)]:
            matrix = calc_matrix(lh, la)
            h, d, a = get_1x2_and_ou(matrix)
            error = (h - target_h)**2 + (d - target_d)**2 + (a - target_a)**2
            
            if target_ou:
                for line, (t_u, t_o) in target_ou.items():
                    u = sum(p for (sc_h,sc_a), p in matrix.items() if sc_h+sc_a < line)
                    o = sum(p for (sc_h,sc_a), p in matrix.items() if sc_h+sc_a > line)
                    error += ((u - t_u)**2 + (o - t_o)**2) * 0.5
                    
            if error < best_error:
                best_error = error
                best_lam_h = lh
                best_lam_a = la
                
    return best_lam_h, best_lam_a

def calc_ev_regular(pred_h, pred_a, matrix):
    ev = 0
    for (act_h, act_a), prob in matrix.items():
        pts = 0
        if pred_h == act_h and pred_a == act_a:
            pts += 10
        else:
            pred_toto = 1 if pred_h > pred_a else (-1 if pred_h < pred_a else 0)
            act_toto = 1 if act_h > act_a else (-1 if act_h < act_a else 0)
            if pred_toto == act_toto:
                if pred_toto == 0:
                    pts += 7
                else:
                    pts += 5
                
                if pred_h == act_h: pts += 2
                if pred_a == act_a: pts += 2
        ev += prob * pts
    return ev

def calc_ev_motd(pred_h, pred_a, matrix):
    pred_scorer_h = (pred_h > 0)
    pred_scorer_a = (pred_a > 0)
    
    ev = 0
    for (act_h, act_a), prob in matrix.items():
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
                
        if not pred_scorer_h:
            if act_h == 0: pts += 4
        else:
            if act_h > 0: pts += 4 * 0.35
            
        if not pred_scorer_a:
            if act_a == 0: pts += 4
        else:
            if act_a > 0: pts += 4 * 0.35
            
        ev += prob * pts
        
    return ev, (pred_scorer_h, pred_scorer_a)

def voorspel(home_pct, draw_pct, away_pct, is_motd, ou_probs=None):
    p_h, p_d, p_a = normaliseer_kansen(home_pct, draw_pct, away_pct)
    lam_h, lam_a = bepaal_poisson_lambdas(p_h, p_d, p_a, ou_probs)
    matrix = calc_matrix(lam_h, lam_a)
    
    best_ev = -1
    best_pred = (0, 0)
    best_scorers = (False, False)
    
    for h in range(7):
        for a in range(7):
            if is_motd:
                ev, scorers = calc_ev_motd(h, a, matrix)
            else:
                ev = calc_ev_regular(h, a, matrix)
                scorers = (False, False)
                
            if ev > best_ev:
                best_ev = ev
                best_pred = (h, a)
                best_scorers = scorers
                
    uitslag = f"{best_pred[0]}-{best_pred[1]}"
    
    if is_motd:
        scorer_thuis = "Spits (of penaltynemer)" if best_scorers[0] else "Geen score"
        scorer_uit = "Spits (of penaltynemer)" if best_scorers[1] else "Geen score"
        uitleg = f"Maximale EV: {best_ev:.2f} verwachte punten (incl. doelpuntenmakers)."
    else:
        scorer_thuis = ""
        scorer_uit = ""
        uitleg = f"Maximale EV: {best_ev:.2f} verwachte punten."

    return {
        "genormaliseerd": (p_h, p_d, p_a),
        "lambda": (lam_h, lam_a),
        "uitslag": uitslag,
        "scorer_thuis": scorer_thuis,
        "scorer_uit": scorer_uit,
        "uitleg": uitleg,
        "xpts": best_ev
    }

def print_resultaat(res, is_motd, toon_extra=False):
    p_h, p_d, p_a = res["genormaliseerd"]
    lam_h, lam_a = res["lambda"]
    ev_val = res.get("xpts", 0.0)
    
    print(f"\n{BOLD}📊  GEANALYSEERDE GEGEVENS (POISSON MODEL):{RESET}")
    print(f"  • Implied Kansen: Thuis: {p_h*100:.1f}% | Gelijk: {p_d*100:.1f}% | Uit: {p_a*100:.1f}%")
    print(f"  • Berekende xG: Thuis: {lam_h:.2f} | Uit: {lam_a:.2f}")
    
    print(f"\n{GREEN}{BOLD}🏆  MAXIMALE EXPECTED VALUE (EV) ADVIES:{RESET}")
    print(f"  • {BOLD}Voorspelde uitslag:{RESET} {GREEN}{BOLD}{res['uitslag']}{RESET}")
    
    if is_motd:
        print(f"  • {BOLD}Doelpuntenmaker Thuis:{RESET} {YELLOW}{res['scorer_thuis']}{RESET}")
        print(f"  • {BOLD}Doelpuntenmaker Uit:{RESET} {YELLOW}{res['scorer_uit']}{RESET}")
    
    print(f"\n{BOLD}💡  BEREKENING:{RESET}")
    print(f"  {res['uitleg']}")
    
    if toon_extra:
        print(f"\n{BOLD}⏱️  TIE-BREAKER EXTRA VRAGEN:{RESET}")
        print(f"  • {BOLD}Minuut van het 1e toernooidoelpunt:{RESET} {YELLOW}31e minuut{RESET} (Mediaan)")
        print(f"  • {BOLD}Minuut van de 1e gele kaart:{RESET} {YELLOW}36e minuut{RESET}")
        print(f"  • {BOLD}Minuut van de 1e rode kaart:{RESET} {YELLOW}411e minuut{RESET}\n")

def interactieve_modus(toon_extra=False):
    print_header()
    
    while True:
        wedstrijd_type = input(f"{BOLD}Is dit de 'Wedstrijd van de Dag' (MOTD)? (ja/nee): {RESET}").strip().lower()
        if wedstrijd_type in ['ja', 'j', 'yes', 'y']:
            is_motd = True
            break
        elif wedstrijd_type in ['nee', 'n', 'no']:
            is_motd = False
            break
        else:
            print(f"{RED}Vul alstublieft 'ja' of 'nee' in.{RESET}")
            
    print(f"\n{BOLD}Voer de winstkansen in (bijv. 45 of 45% of 0.45):{RESET}")
    while True:
        try:
            h_in = input("  1. Kans op Thuiswinst: ").strip()
            home = parse_percentage(h_in)
            break
        except ValueError as e:
            print(f"  {RED}❌ {e}. Probeer het opnieuw.{RESET}")
            
    while True:
        try:
            d_in = input("  2. Kans op Gelijkspel: ").strip()
            draw = parse_percentage(d_in)
            break
        except ValueError as e:
            print(f"  {RED}❌ {e}. Probeer het opnieuw.{RESET}")
            
    while True:
        try:
            a_in = input("  3. Kans op Uitwinst:   ").strip()
            away = parse_percentage(a_in)
            break
        except ValueError as e:
            print(f"  {RED}❌ {e}. Probeer het opnieuw.{RESET}")
            
    res = voorspel(home, draw, away, is_motd)
    print_resultaat(res, is_motd, toon_extra=toon_extra)


MOTD_LIST = [
    {"nederland", "netherlands", "japan"},
    {"belgie", "belgium", "egypte", "egypt"},
    {"engeland", "england", "kroatie", "croatia"},
    {"vs", "usa", "united", "states", "australie", "australia"},
    {"nederland", "netherlands", "zweden", "sweden"},
    {"nieuw", "zeeland", "new", "egypte", "egypt"},
    {"jordanie", "jordan", "algerije", "algeria"},
    {"colombia", "dr", "congo"},
    {"schotland", "scotland", "brazilie", "brazil"},
    {"japan", "zweden", "sweden"},
    {"noorwegen", "norway", "frankrijk", "france"},
    {"uruguay", "spanje", "spain"}
]

def is_motd_match(home, away):
    home_lower = home.lower()
    away_lower = away.lower()
    
    home_words = set(re.findall(r'\w+', home_lower))
    away_words = set(re.findall(r'\w+', away_lower))
    
    # Map "vs", "usa", "united states" to same keywords
    if any(w in home_words for w in ["united", "states", "usa"]):
        home_words.add("vs")
    if any(w in away_words for w in ["united", "states", "usa"]):
        away_words.add("vs")
        
    for motd in MOTD_LIST:
        has_home = any(w in motd for w in home_words)
        has_away = any(w in motd for w in away_words)
        if has_home and has_away:
            return True
            
    return False

def haal_polymarket_wedstrijden():
    url = "https://gamma-api.polymarket.com/events"
    params = {
        "tag_id": 100350,
        "active": "true",
        "closed": "false",
        "limit": 100
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None, f"Fout bij ophalen Polymarket data (statuscode: {r.status_code})"
        
        events = r.json()
        parsed_matches = []
        
        for e in events:
            title = e.get("title", "")
            slug = e.get("slug", "")
            
            if not (slug.startswith("fifwc-") or "world-cup" in slug.lower() or "world cup" in title.lower()):
                continue
                
            markets = e.get("markets", [])
            team_part = title
            if ":" in title:
                team_part = title.split(":")[-1].strip()
                
            teams = re.split(r'\s+vs\.?\s+', team_part, flags=re.IGNORECASE)
            if len(teams) != 2:
                continue
                
            home_team = teams[0].strip()
            away_team = teams[1].strip()
            
            home_prob = draw_prob = away_prob = None
            ou_probs = {}
            non_draw_markets = []
            
            for m in markets:
                q = m.get("question", "").lower()
                prices_str = m.get("outcomePrices")
                if not prices_str:
                    continue
                prices = json.loads(prices_str)
                if len(prices) < 1:
                    continue
                yes_price = float(prices[0])
                
                match_ou = re.search(r'(over|under) (\d+\.5) goals', q)
                if match_ou:
                    type_ou = match_ou.group(1)
                    line = float(match_ou.group(2))
                    if line not in ou_probs:
                        ou_probs[line] = [None, None]
                    if type_ou == 'under':
                        ou_probs[line][0] = yes_price
                    else:
                        ou_probs[line][1] = yes_price
                elif "draw" in q:
                    draw_prob = yes_price
                else:
                    non_draw_markets.append((q, yes_price))
                    
            if len(non_draw_markets) >= 2:
                home_words = set(re.findall(r'\w+', home_team.lower()))
                away_words = set(re.findall(r'\w+', away_team.lower()))
                
                best_m_home = best_m_away = None
                max_score_h = max_score_a = 0
                
                for mq, mp in non_draw_markets:
                    mq_words = set(re.findall(r'\w+', mq))
                    sc_h = len(mq_words.intersection(home_words))
                    sc_a = len(mq_words.intersection(away_words))
                    
                    if sc_h > max_score_h:
                        max_score_h = sc_h
                        best_m_home = mp
                    if sc_a > max_score_a:
                        max_score_a = sc_a
                        best_m_away = mp
                        
                if best_m_home and best_m_away:
                    home_prob = best_m_home
                    away_prob = best_m_away

            if home_prob is not None and draw_prob is not None and away_prob is not None:
                final_ou = {}
                for line, (u, o) in ou_probs.items():
                    if u is not None and o is not None:
                        tot = u + o
                        final_ou[line] = (u/tot, o/tot)
                    elif o is not None:
                        final_ou[line] = (1-o, o)
                    elif u is not None:
                        final_ou[line] = (u, 1-u)
                
                parsed_matches.append({
                    "title": team_part,
                    "home": home_team,
                    "away": away_team,
                    "home_prob": home_prob * 100.0,
                    "draw_prob": draw_prob * 100.0,
                    "away_prob": away_prob * 100.0,
                    "ou_probs": final_ou,
                    "date": e.get("endDate", ""),
                    "is_motd": is_motd_match(home_team, away_team)
                })
                
        parsed_matches.sort(key=lambda x: x["date"])
        return parsed_matches, None
    except Exception as err:
        return None, f"Fout bij verbinding met Polymarket: {err}"

def exporteer_naar_bestand(alle_res, bestandsnaam):
    """Exporteert alle berekende uitslagen chronologisch naar een tekstbestand met xG en MOTD scorer tips."""
    try:
        with open(bestandsnaam, "w", encoding="utf-8") as f:
            f.write("======================================================================================================================================\n")
            f.write("                                                   WK VOORSPELLINGEN (OPTA & POLYMARKET)\n")
            f.write("======================================================================================================================================\n\n")
            f.write(f"{'Datum/Tijd':<17} | {'Thuisploeg':<20} vs. {'Uitploeg':<20} | {'Odds (1/X/2)':<18} | {'xG (Thuis-Uit)':<15} | {'EV (pts)':<8} | {'Advies':<12} | {'Doelpuntenmaker Tips (MOTD)':<30}\n")
            f.write("-" * 145 + "\n")
            for m, res in alle_res:
                kansen_str = f"{m['home_prob']:.1f}% / {m['draw_prob']:.1f}% / {m['away_prob']:.1f}%"
                lam_h, lam_a = res["lambda"]
                ev_val = res.get("xpts", 0.0)
                xg_str = f"{lam_h:.2f} - {lam_a:.2f}"
                datum_str = converteer_utc_naar_nl(m['date'])
                
                advies_str = res["uitslag"]
                if m["is_motd"]:
                    advies_str += " [MOTD]"
                
                scorer_str = ""
                if m["is_motd"]:
                    thuis_tip = "Spits" if "spits" in res["scorer_thuis"].lower() else "Geen"
                    uit_tip = "Spits" if "spits" in res["scorer_uit"].lower() else "Geen"
                    scorer_str = f"Thuis: {thuis_tip} | Uit: {uit_tip}"
                    
                f.write(f"{datum_str:<17} | {m['home']:<20} vs. {m['away']:<20} | {kansen_str:<18} | {xg_str:<15} | {ev_val:<8.2f} | {advies_str:<12} | {scorer_str:<30}\n")
            f.write("\n======================================================================================================================================\n")
            f.write("Gegenereerd door de Voetbalpoules Opta Voorspeller CLI.\n")
            
        print(f"{GREEN}✓ Voorspellingen succesvol opgeslagen in {BOLD}{bestandsnaam}{RESET}!\n")
    except Exception as e:
        print(f"{RED}❌ Fout bij opslaan van bestand: {e}{RESET}\n")

def exporteer_naar_html(alle_res, bestandsnaam):
    """Genereert een prachtige, mobielvriendelijke HTML-pagina (index.html) met de voorspellingen."""
    from zoneinfo import ZoneInfo
    import datetime
    nu_nl = datetime.datetime.now(ZoneInfo("Europe/Amsterdam"))
    nu_str = nu_nl.strftime("%d-%m-%Y %H:%M")
    
    html_content = f"""<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WK 2026 Voorspellingen - Opta Analyst & Polymarket</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-blue: #06b6d4;
            --accent-red: #ef4444;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(6, 182, 212, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.15) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-primary);
            padding: 20px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 30px;
            margin-top: 10px;
            max-width: 600px;
            width: 100%;
            animation: fadeInDown 0.8s ease-out;
        }}
        
        h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            background: linear-gradient(135deg, #06b6d4, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 0.95rem;
            line-height: 1.5;
        }}
        
        .last-updated {{
            display: inline-block;
            margin-top: 8px;
            font-size: 0.8rem;
            color: var(--accent-blue);
            background: rgba(6, 182, 212, 0.1);
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 600;
        }}
        
        .container {{
            max-width: 600px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 16px;
            animation: fadeInUp 0.8s ease-out;
        }}
        
        .match-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        
        .match-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }}
        
        .match-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: transparent;
            transition: background 0.3s;
        }}
        
        .match-card.is-motd::before {{
            background: linear-gradient(90deg, var(--accent-yellow), transparent);
        }}
        
        .match-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 12px;
            font-weight: 600;
        }}
        
        .motd-badge {{
            background: rgba(245, 158, 11, 0.15);
            color: var(--accent-yellow);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        
        .match-teams {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }}
        
        .team {{
            font-size: 1.15rem;
            font-weight: 600;
            width: 40%;
        }}
        
        .team.home {{
            text-align: right;
        }}
        
        .team.away {{
            text-align: left;
        }}
        
        .vs-text {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.05);
            padding: 4px 8px;
            border-radius: 8px;
            font-weight: 600;
        }}
        
        .match-details {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            border-top: 1px solid var(--border-color);
            padding-top: 14px;
        }}
        
        .detail-item {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        
        .detail-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .detail-value {{
            font-size: 0.9rem;
            font-weight: 600;
        }}
        
        .prediction-box {{
            grid-column: span 2;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 12px;
            padding: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .match-card.is-motd .prediction-box {{
            background: rgba(245, 158, 11, 0.07);
            border-color: rgba(245, 158, 11, 0.15);
        }}
        
        .pred-score {{
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--accent-green);
        }}
        
        .match-card.is-motd .pred-score {{
            color: var(--accent-yellow);
        }}
        
        .scorer-tips {{
            grid-column: span 2;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 10px;
            font-size: 0.8rem;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        
        .scorer-row {{
            display: flex;
            justify-content: space-between;
        }}
        
        .scorer-team {{
            color: var(--text-secondary);
        }}
        
        .scorer-name {{
            font-weight: 600;
            color: var(--accent-yellow);
        }}
        
        @keyframes fadeInDown {{
            from {{
                opacity: 0;
                transform: translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        @media (max-width: 480px) {{
            body {{
                padding: 12px;
            }}
            h1 {{
                font-size: 1.8rem;
            }}
            .team {{
                font-size: 1rem;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>🏆 WK 2026 Voorspeller</h1>
        <div class="subtitle">Wiskundig optimale uitslagen berekend op basis van live winstkansen van Polymarket en Opta Analyst data.</div>
        <div class="last-updated">Geüpdatet: {nu_str} CEST</div>
    </header>
    
    <div class="container">
"""
    
    for m, res in alle_res:
        motd_badge = '<span class="motd-badge">Wedstrijd van de Dag</span>' if m["is_motd"] else ''
        card_class = 'is-motd' if m["is_motd"] else ''
        datum_str = converteer_utc_naar_nl(m['date'])
        kansen_str = f"{m['home_prob']:.0f}% / {m['draw_prob']:.0f}% / {m['away_prob']:.0f}%"
        lam_h, lam_a = res["lambda"]
        ev_val = res.get("xpts", 0.0)
        
        # Build card html
        html_content += f"""
        <div class="match-card {card_class}">
            <div class="match-header">
                <span>📅 {datum_str}</span>
                {motd_badge}
            </div>
            <div class="match-teams">
                <span class="team home">{m['home']}</span>
                <span class="vs-text">VS</span>
                <span class="team away">{m['away']}</span>
            </div>
            <div class="match-details">
                <div class="detail-item">
                    <span class="detail-label">Odds (1/X/2)</span>
                    <span class="detail-value">{kansen_str}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">xG (Verwacht)</span>
                    <span class="detail-value">{lam_h:.2f} - {lam_a:.2f}</span>
                </div>
                
                <div class="prediction-box">
                    <div class="detail-item">
                        <span class="detail-label" style="color: var(--text-primary);">Aanbevolen Uitslag</span>
                        <span class="subtitle" style="font-size: 0.75rem;">Verwachte Punten (EV): <strong style="color: var(--accent-blue);">{res.get('xpts', 0.0):.2f} pts</strong></span>
                    </div>
                    <span class="pred-score">{res['uitslag']}</span>
                </div>
        """
        
        # Scorer tips if MOTD
        if m["is_motd"]:
            thuis_tip = "Spits (of penaltynemer)" if "spits" in res["scorer_thuis"].lower() else "Geen score"
            uit_tip = "Spits (of penaltynemer)" if "spits" in res["scorer_uit"].lower() else "Geen score"
            
            html_content += f"""
                <div class="scorer-tips">
                    <span class="detail-label" style="color: var(--accent-yellow);">Doelpuntenmaker Tips (MOTD)</span>
                    <div class="scorer-row">
                        <span class="scorer-team">{m['home']}:</span>
                        <span class="scorer-name">{thuis_tip}</span>
                    </div>
                    <div class="scorer-row">
                        <span class="scorer-team">{m['away']}:</span>
                        <span class="scorer-name">{uit_tip}</span>
                    </div>
                </div>
            """
            
        html_content += """
            </div>
        </div>
        """
        
    html_content += """
    </div>
    
    <footer style="margin-top: 40px; margin-bottom: 20px; font-size: 0.8rem; color: var(--text-secondary); text-align: center;">
        <p>Berekend met de Voetbalpoules Opta Voorspeller. Data ververst dagelijks om 17:00 CEST.</p>
    </footer>
</body>
</html>
"""

    try:
        with open(bestandsnaam, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"{GREEN}✓ Mobiele website succesvol gegenereerd als {BOLD}{bestandsnaam}{RESET}!\n")
    except Exception as e:
        print(f"{RED}❌ Fout bij genereren HTML-bestand: {e}{RESET}\n")

def polymarket_modus(toon_extra=False, output_file=None):
    print_header()
    print(f"{CYAN}{BOLD}Bezig met ophalen van actieve WK-wedstrijden en odds van Polymarket...{RESET}")
    matches, error = haal_polymarket_wedstrijden()
    if error:
        print(f"{RED}❌ {error}{RESET}")
        return
        
    if not matches:
        print(f"{YELLOW}Geen actieve WK-wedstrijden gevonden op Polymarket op dit moment.{RESET}")
        return
        
    print(f"\n{GREEN}✓ {len(matches)} actieve WK-wedstrijden succesvol opgehaald!{RESET}\n")
    
    while True:
        print(f"{BOLD}Beschikbare wedstrijden (chronologisch):{RESET}")
        for idx, m in enumerate(matches):
            datum_str = converteer_utc_naar_nl(m['date'])
            motd_label = " [MOTD]" if m['is_motd'] else ""
            print(f"  {idx+1:2d}. [{datum_str}] {m['home']} vs. {m['away']}{motd_label} (Odds: {m['home_prob']:.1f}% / {m['draw_prob']:.1f}% / {m['away_prob']:.1f}%)")
        print(f"  {len(matches)+1:2d}. [Voorspel ALLE wedstrijden]")
        
        keuze = input(f"\n{BOLD}Kies een nummer (1 t/m {len(matches)+1}) of typ 'exit': {RESET}").strip().lower()
        if keuze in ['exit', 'quit', 'q']:
            print(f"\n{YELLOW}Tot ziens! 👋{RESET}\n")
            return
            
        try:
            val = int(keuze)
            if 1 <= val <= len(matches):
                # Voorspel één wedstrijd
                m = matches[val-1]
                print(f"\nJe koos: {BOLD}{m['home']} vs. {m['away']}{RESET}")
                
                suggestie = "ja" if m['is_motd'] else "nee"
                while True:
                    motd_in = input(f"{BOLD}Is dit de Wedstrijd van de Dag (MOTD)? (ja/nee) [standaard: {suggestie}]: {RESET}").strip().lower()
                    if not motd_in:
                        is_motd = m['is_motd']
                        break
                    elif motd_in in ['ja', 'j', 'yes', 'y']:
                        is_motd = True
                        break
                    elif motd_in in ['nee', 'n', 'no']:
                        is_motd = False
                        break
                    else:
                        print(f"{RED}Vul alstublieft 'ja' of 'nee' in.{RESET}")
                
                res = voorspel(m['home_prob'], m['draw_prob'], m['away_prob'], is_motd, ou_probs=m.get('ou_probs'))
                print_resultaat(res, is_motd, toon_extra=toon_extra)
                break
                
            elif val == len(matches) + 1:
                # Voorspel ALLE wedstrijden
                print(f"\n{BOLD}Berekent voorspellingen voor alle {len(matches)} wedstrijden...{RESET}\n")
                
                alle_res = []
                for m in matches:
                    res = voorspel(m['home_prob'], m['draw_prob'], m['away_prob'], is_motd=m['is_motd'], ou_probs=m['ou_probs'])
                    alle_res.append((m, res))
                    
                # Toon tabel
                print(f"{CYAN}{BOLD}================================================================================================================================================={RESET}")
                print(f"                                                               OVERZICHT ALLE VOORSPELDE WEDSTRIJDEN")
                print(f"{CYAN}{BOLD}================================================================================================================================================={RESET}")
                print(f"{BOLD}{'Datum/Tijd':<17} | {'Thuisploeg':<20} vs. {'Uitploeg':<20} | {'Odds (1/X/2)':<18} | {'xG (Thuis-Uit)':<15} | {'EV (pts)':<8} | {'Uitslag':<12} | {'Doelpuntenmaker Tips (MOTD)':<30}{RESET}")
                print("-" * 145)
                for m, res in alle_res:
                    kansen_str = f"{m['home_prob']:.0f}% / {m['draw_prob']:.0f}% / {m['away_prob']:.0f}%"
                    lam_h, lam_a = res["lambda"]
                    ev_val = res.get("xpts", 0.0)
                    xg_str = f"{lam_h:.2f} - {lam_a:.2f}"
                    datum_str = converteer_utc_naar_nl(m['date'])
                    
                    advies_str = res["uitslag"]
                    if m["is_motd"]:
                        advies_str += " [MOTD]"
                    
                    scorer_str = ""
                    if m["is_motd"]:
                        thuis_tip = "Spits" if "spits" in res["scorer_thuis"].lower() else "Geen"
                        uit_tip = "Spits" if "spits" in res["scorer_uit"].lower() else "Geen"
                        scorer_str = f"Thuis: {thuis_tip} | Uit: {uit_tip}"
                        
                    print(f"{datum_str:<17} | {m['home']:<20} vs. {m['away']:<20} | {kansen_str:<18} | {xg_str:<15} | {ev_val:<8.2f} | {GREEN}{BOLD}{advies_str:<12}{RESET} | {YELLOW}{scorer_str:<30}{RESET}")
                print(f"{CYAN}{BOLD}======================================================================================================================================{RESET}\n")
                
                # Exporteren
                if not output_file:
                    opslaan = input(f"{BOLD}Wil je deze voorspellingen opslaan in een tekstbestand? (ja/nee) [standaard: ja]: {RESET}").strip().lower()
                    if opslaan not in ['nee', 'n', 'no']:
                        output_file = "voorspellingen.txt"
                
                if output_file:
                    exporteer_naar_bestand(alle_res, output_file)
                break
            else:
                print(f"{RED}Ongeldig nummer. Kies een getal tussen 1 en {len(matches)+1}.{RESET}\n")
        except ValueError:
            print(f"{RED}Vul een geldig nummer in.{RESET}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Berekent de wiskundig optimale uitslag voor voetbalpoules op basis van Opta 1X2 kansen."
    )
    parser.add_argument("-t", "--home", type=str, help="Kans op winst voor het thuisteam (bijv. 45 of 45%%)")
    parser.add_argument("-g", "--draw", type=str, help="Kans op een gelijkspel (bijv. 28 of 28%%)")
    parser.add_argument("-u", "--away", type=str, help="Kans op winst voor het uitteam (bijv. 27 of 27%%)")
    parser.add_argument("-m", "--motd", action="store_true", help="Stel in als dit de Wedstrijd van de Dag (MOTD) is")
    parser.add_argument("-e", "--extra", action="store_true", help="Toon extra toernooi tie-breaker voorspellingen")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start de interactieve vragengids")
    parser.add_argument("-p", "--polymarket", action="store_true", help="Haal actieve WK-kansen op van Polymarket")
    parser.add_argument("-o", "--output", type=str, help="Exporteer alle Polymarket voorspellingen naar dit bestand")
    parser.add_argument("-w", "--web", type=str, help="Genereer een prachtige HTML-pagina (index.html) naar dit bestand")
    
    args = parser.parse_args()
    
    # Als polymarket-modus is gekozen:
    if args.polymarket:
        if args.output or args.web:
            print_header()
            print(f"{CYAN}{BOLD}Bezig met batch-verwerking van alle WK-kansen van Polymarket...{RESET}")
            matches, error = haal_polymarket_wedstrijden()
            if error:
                print(f"{RED}❌ {error}{RESET}")
                sys.exit(1)
            if not matches:
                print(f"{YELLOW}Geen actieve WK-wedstrijden gevonden op Polymarket.{RESET}")
                sys.exit(0)
                
            alle_res = []
            for m in matches:
                res = voorspel(m['home_prob'], m['draw_prob'], m['away_prob'], is_motd=m['is_motd'], ou_probs=m['ou_probs'])
                alle_res.append((m, res))
                
            if args.output:
                exporteer_naar_bestand(alle_res, args.output)
            if args.web:
                exporteer_naar_html(alle_res, args.web)
            sys.exit(0)
        else:
            try:
                polymarket_modus(toon_extra=args.extra)
            except (KeyboardInterrupt, SystemExit):
                print(f"\n\n{YELLOW}Programma afgebroken. Tot ziens! 👋{RESET}\n")
            sys.exit(0)
            
    # Als er geen argumenten zijn opgegeven of specifiek --interactive is meegegeven:
    if args.interactive or (args.home is None and args.draw is None and args.away is None):
        try:
            interactieve_modus(toon_extra=args.extra)
        except (KeyboardInterrupt, SystemExit):
            print(f"\n\n{YELLOW}Programma afgebroken. Tot ziens! 👋{RESET}\n")
    else:
        # Controleer of alle drie de kansen zijn ingevoerd in argument-modus
        if args.home is None or args.draw is None or args.away is None:
            print(f"{RED}Fout: Voer alle drie de kansen in (--home, --draw en --away) of start zonder argumenten voor de interactieve gids.{RESET}")
            sys.exit(1)
            
        try:
            home = parse_percentage(args.home)
            draw = parse_percentage(args.draw)
            away = parse_percentage(args.away)
        except ValueError as e:
            print(f"{RED}Fout bij verwerken invoer: {e}{RESET}")
            sys.exit(1)
            
        res = voorspel(home, draw, away, args.motd)
        print_header()
        print_resultaat(res, args.motd, toon_extra=args.extra)

if __name__ == "__main__":
    main()
