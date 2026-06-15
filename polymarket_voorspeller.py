#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voetbalpoules Polymarket Voorspeller
------------------------------------
Deze tool berekent de wiskundig optimale uitslag voor voetbalpoules op basis van
de Polymarket 1X2 winstkansen, volgens het onderzoeksrapport:
'Optimalisatie van Expected Points (xPts) in Voetbalpoules'.

Dit model maakt gebruik van Nelder-Mead optimalisatie voor het schatten van
de Poisson-parameters en past de Dixon-Coles correctie toe om gelijkspelen
beter te voorspellen bij lage scores.
"""

import sys
import argparse
import json
import re
import math
import datetime
from zoneinfo import ZoneInfo
import requests
from scipy.optimize import minimize

# ANSI Kleurcodes voor een mooie vormgeving in de terminal (zonder extra pakketten!)
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"

def print_header():
    """
    Toont de start-header in de terminal met informatie over de voetbalvoorspeller.
    """
    print(f"\n{CYAN}{BOLD}========================================================")
    print(f"       ⚽  VOETBALPOULES POLYMARKET VOORSPELLER  ⚽")
    print(f"========================================================{RESET}")
    print("Dit programma berekent de wiskundig beste uitslag om de")
    print("meeste punten (Expected Points) te behalen in je poule.")
    print("Model: Nelder-Mead optimalisatie + Dixon-Coles correctie")
    print("       voor betere gelijkspelschattingen.")
    print("--------------------------------------------------------\n")


def parse_percentage(val_str):
    """
    Zet een procentteken-tekst (zoals '45%') of kommagetal om naar een getal van 0 tot 100.
    
    Parameters:
    val_str (str): De invoertekst die de kans representeert.
    
    Returns:
    float: De kans als percentage (tussen 0.0 en 100.0).
    """
    val_str = val_str.strip().replace('%', '')
    try:
        val = float(val_str)
        if 0.0 <= val <= 1.0:
            val = val * 100.0
        return val
    except ValueError:
        raise ValueError(f"Ongeldig getal: '{val_str}'")

def normaliseer_kansen(home, draw, away):
    """
    Zorgt ervoor dat de drie kansen (thuis, gelijk, uit) samen exact 100% (of 1.0) worden.
    
    Parameters:
    home (float): De ingevoerde kans op thuiswinst.
    draw (float): De ingevoerde kans op een gelijkspel.
    away (float): De ingevoerde kans op uitwinst.
    
    Returns:
    tuple: Een drietal met de genormaliseerde kansen (thuis, gelijk, uit) die optellen tot 1.0.
    """
    totaal = home + draw + away
    if totaal == 0:
        return 0.0, 0.0, 0.0
    return home / totaal, draw / totaal, away / totaal

def converteer_utc_naar_nl(utc_str):
    """
    Zet een UTC-tijdstip om naar de Nederlandse tijdzone en formatteert dit als leesbare tekst.
    
    Parameters:
    utc_str (str): De datum/tijd-tekenreeks in UTC-formaat.
    
    Returns:
    str: De geformatteerde Nederlandse datum en tijd (JJJJ-MM-DD UU:MM).
    """
    if not utc_str:
        return ""
    try:
        clean_str = utc_str.replace('Z', '+00:00')
        dt_utc = datetime.datetime.fromisoformat(clean_str)
        dt_nl = dt_utc.astimezone(ZoneInfo("Europe/Amsterdam"))
        return dt_nl.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return utc_str.replace('T', ' ')[:16]


def poisson(lam, k):
    """
    Berekent de kans op exact k doelpunten met een bepaald gemiddelde aantal doelpunten (lambda).
    
    Parameters:
    lam (float): Het verwachte gemiddelde aantal doelpunten (lambda).
    k (int): Het aantal doelpunten waarvoor de kans berekend moet worden.
    
    Returns:
    float: De kans op exact k doelpunten volgens de Poisson-verdeling.
    """
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def dixon_coles_tau(h, a, lam_h, lam_a, rho):
    """
    Berekent de Dixon-Coles correctiefactor voor uitslagen met lage scores (0 en 1 doelpunten).
    Dit helpt om de kans op gelijkspelen en nipte overwinningen beter te schatten.
    
    Parameters:
    h (int): Het aantal doelpunten van het thuisteam.
    a (int): Het aantal doelpunten van het uitteam.
    lam_h (float): Het verwachte gemiddelde aantal doelpunten van het thuisteam.
    lam_a (float): Het verwachte gemiddelde aantal doelpunten van het uitteam.
    rho (float): De Dixon-Coles correctieparameter (ρ).
    
    Returns:
    float: De vermenigvuldigingsfactor voor de kans op deze specifieke uitslag.
    """
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    elif h == 1 and a == 0:
        return 1.0 + lam_a * rho
    elif h == 0 and a == 1:
        return 1.0 + lam_h * rho
    elif h == 1 and a == 1:
        return 1.0 - rho
    else:
        return 1.0

def calc_matrix(lam_h, lam_a, rho=0.0):
    """
    Berekent de kansenmatrix voor uitslagen van 0-0 tot 9-9 op basis van de Poisson-verdelingen
    en de Dixon-Coles correctieparameter.
    
    Parameters:
    lam_h (float): Het verwachte gemiddelde aantal doelpunten van het thuisteam.
    lam_a (float): Het verwachte gemiddelde aantal doelpunten van het uitteam.
    rho (float, optioneel): De Dixon-Coles correctieparameter (ρ). Standaard 0.0.
    
    Returns:
    dict: Een woordenboek met uitslagen (thuis, uit) als sleutels en hun kansen als waarden.
    """
    matrix = {}
    totaal = 0.0
    for h in range(10):
        for a in range(10):
            prob = poisson(lam_h, h) * poisson(lam_a, a) * dixon_coles_tau(h, a, lam_h, lam_a, rho)
            prob = max(0.0, prob)
            matrix[(h, a)] = prob
            totaal += prob
            
    if totaal > 0.0:
        for key in matrix:
            matrix[key] /= totaal
    return matrix

def get_1x2_and_ou(matrix):
    """
    Berekent de totale kansen op thuiswinst (1), gelijkspel (X) en uitwinst (2) uit de kansenmatrix.
    
    Parameters:
    matrix (dict): De berekende kansenmatrix voor alle uitslagen.
    
    Returns:
    tuple: Een drietal met de totale kans op (thuiswinst, gelijkspel, uitwinst).
    """
    h_win = sum(p for (h,a), p in matrix.items() if h > a)
    d = sum(p for (h,a), p in matrix.items() if h == a)
    a_win = sum(p for (h,a), p in matrix.items() if h < a)
    return h_win, d, a_win

def bepaal_poisson_lambdas(target_h, target_d, target_a, target_ou=None):
    """
    Vindt de optimale Poisson-lambda's en de Dixon-Coles rho-waarde die het beste aansluiten
    bij de gewenste winst-, gelijkspel- en verlieskansen (en eventuele over/under kansen).
    Dit gebeurt met behulp van de Nelder-Mead optimalisatie.
    
    Parameters:
    target_h (float): De gewenste (genormaliseerde) kans op thuiswinst.
    target_d (float): De gewenste (genormaliseerde) kans op gelijkspel.
    target_a (float): De gewenste (genormaliseerde) kans op uitwinst.
    target_ou (dict, optioneel): Kansen voor over/under doelpuntengrenzen.
    
    Returns:
    tuple: Een drietal met de berekende parameters (lambda_thuis, lambda_uit, rho).
    """
    def objective(params):
        lam_h, lam_a, rho = params
        # Zorg dat de lambda's en rho binnen het geldige bereik liggen tijdens de optimalisatie
        lh = max(0.05, min(lam_h, 5.0))
        la = max(0.05, min(lam_a, 5.0))
        r = max(-0.25, min(rho, 0.10))
        
        matrix = calc_matrix(lh, la, r)
        h, d, a = get_1x2_and_ou(matrix)
        error = (h - target_h)**2 + (d - target_d)**2 + (a - target_a)**2
        
        if target_ou:
            for line, (t_u, t_o) in target_ou.items():
                u = sum(p for (sc_h,sc_a), p in matrix.items() if sc_h+sc_a < line)
                o = sum(p for (sc_h,sc_a), p in matrix.items() if sc_h+sc_a > line)
                error += ((u - t_u)**2 + (o - t_o)**2) * 0.5
        return error

    res = minimize(objective, [1.3, 1.0, -0.05], method='Nelder-Mead')
    
    # Zorg dat de definitieve resultaten worden afgekapt (clipped) tot het bereik [0.05, 5.0] en [-0.25, 0.10]
    lam_h_opt = max(0.05, min(res.x[0], 5.0))
    lam_a_opt = max(0.05, min(res.x[1], 5.0))
    rho_opt = max(-0.25, min(res.x[2], 0.10))
    
    return lam_h_opt, lam_a_opt, rho_opt

def calc_ev_regular(pred_h, pred_a, matrix):
    """
    Berekent de verwachte waarde (Expected Value, EV) in punten voor een voorspelde uitslag in een normale poule.
    
    Parameters:
    pred_h (int): Het voorspelde aantal doelpunten van het thuisteam.
    pred_a (int): Het voorspelde aantal doelpunten van het uitteam.
    matrix (dict): De berekende kansenmatrix voor alle uitslagen.
    
    Returns:
    float: Het verwachte aantal punten voor deze voorspelling.
    """
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
    """
    Berekent de verwachte waarde (Expected Value, EV) in punten voor de Wedstrijd van de Dag (MOTD),
    waarbij extra punten voor doelpuntenmakers (spitsen) worden meegerekend.
    
    Parameters:
    pred_h (int): Het voorspelde aantal doelpunten van het thuisteam.
    pred_a (int): Het voorspelde aantal doelpunten van het uitteam.
    matrix (dict): De berekende kansenmatrix voor alle uitslagen.
    
    Returns:
    tuple: Een duo met (de verwachte punten, (thuis_scorer_tip, uit_scorer_tip)).
    """
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
    """
    Berekent de optimale voorspelling door de uitslag te zoeken die de verwachte waarde (EV) maximaliseert.
    
    Parameters:
    home_pct (float): De kans op thuiswinst (percentage).
    draw_pct (float): De kans op gelijkspel (percentage).
    away_pct (float): De kans op uitwinst (percentage).
    is_motd (bool): Geeft aan of dit de Wedstrijd van de Dag (MOTD) is.
    ou_probs (dict, optioneel): Kansen voor over/under grenzen.
    
    Returns:
    dict: Een woordenboek met alle resultaten, zoals genormaliseerde kansen, lambda's, rho,
          de geadviseerde uitslag, tips voor doelpuntenmakers en de maximale verwachte punten.
    """
    p_h, p_d, p_a = normaliseer_kansen(home_pct, draw_pct, away_pct)
    lam_h, lam_a, rho = bepaal_poisson_lambdas(p_h, p_d, p_a, ou_probs)
    matrix = calc_matrix(lam_h, lam_a, rho)
    
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
        "rho": rho,
        "uitslag": uitslag,
        "scorer_thuis": scorer_thuis,
        "scorer_uit": scorer_uit,
        "uitleg": uitleg,
        "xpts": best_ev
    }

def print_resultaat(res, is_motd, toon_extra=False):
    """
    Toont de geanalyseerde gegevens en het voorspellingsadvies op een overzichtelijke manier in de terminal.
    
    Parameters:
    res (dict): Het resultaatwoordenboek uit de voorspel-functie.
    is_motd (bool): Geeft aan of dit de Wedstrijd van de Dag (MOTD) is.
    toon_extra (bool, optioneel): Of er extra tie-breaker statistieken getoond moeten worden. Standaard False.
    """
    p_h, p_d, p_a = res["genormaliseerd"]
    lam_h, lam_a = res["lambda"]
    rho = res.get("rho", 0.0)
    ev_val = res.get("xpts", 0.0)
    
    print(f"\n{BOLD}📊  GEANALYSEERDE GEGEVENS (POISSON MODEL):{RESET}")
    print(f"  • Implied Kansen: Thuis: {p_h*100:.1f}% | Gelijk: {p_d*100:.1f}% | Uit: {p_a*100:.1f}%")
    print(f"  • Berekende xG: Thuis: {lam_h:.2f} | Uit: {lam_a:.2f} | ρ: {rho:.2f}")
    
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
    """
    Start een interactief vraag-en-antwoordscherm in de terminal om een voorspelling voor één wedstrijd te berekenen.
    
    Parameters:
    toon_extra (bool, optioneel): Of er extra tie-breaker statistieken getoond moeten worden. Standaard False.
    """
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
    """
    Controleert of de gegeven teams overeenkomen met een van de 'Wedstrijden van de Dag' (MOTD) uit de lijst.
    
    Parameters:
    home (str): De naam van de thuisploeg.
    away (str): De naam van de uitploeg.
    
    Returns:
    bool: True als het een MOTD-wedstrijd is, anders False.
    """
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
    """
    Haalt live WK-wedstrijden en bijbehorende kansen op via de Polymarket API.
    
    Returns:
    tuple: Een duo met (een lijst van gevonden wedstrijden, een eventuele foutmelding).
    """
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
    """
    Exporteert alle berekende uitslagen chronologisch naar een tekstbestand met xG, rho en MOTD scorer tips.
    
    Parameters:
    alle_res (list): Een lijst met paren van (wedstrijd_data, voorspelling_resultaat).
    bestandsnaam (str): Het pad naar het uit te voeren tekstbestand.
    """
    try:
        with open(bestandsnaam, "w", encoding="utf-8") as f:
            f.write("=================================================================================================================================================\n")
            f.write("                                                   WK VOORSPELLINGEN (POLYMARKET)\n")
            f.write("=================================================================================================================================================\n\n")
            f.write(f"{'Datum/Tijd':<17} | {'Thuisploeg':<20} vs. {'Uitploeg':<20} | {'Odds (1/X/2)':<18} | {'xG (Thuis-Uit)':<15} | {'rho':<6} | {'EV (pts)':<8} | {'Advies':<12} | {'Doelpuntenmaker Tips (MOTD)':<30}\n")
            f.write("-" * 154 + "\n")
            for m, res in alle_res:
                kansen_str = f"{m['home_prob']:.1f}% / {m['draw_prob']:.1f}% / {m['away_prob']:.1f}%"
                lam_h, lam_a = res["lambda"]
                rho = res.get("rho", 0.0)
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
                    
                f.write(f"{datum_str:<17} | {m['home']:<20} vs. {m['away']:<20} | {kansen_str:<18} | {xg_str:<15} | {rho:<6.2f} | {ev_val:<8.2f} | {advies_str:<12} | {scorer_str:<30}\n")
            f.write("\n=================================================================================================================================================\n")
            f.write("Gegenereerd door de Voetbalpoules Polymarket Voorspeller CLI.\n")
            
        print(f"{GREEN}✓ Voorspellingen succesvol opgeslagen in {BOLD}{bestandsnaam}{RESET}!\n")
    except Exception as e:
        print(f"{RED}❌ Fout bij opslaan van bestand: {e}{RESET}\n")

def exporteer_naar_html(alle_res, bestandsnaam):
    """
    Genereert een prachtige, mobielvriendelijke HTML-pagina (index.html) met de voorspellingen.
    
    Parameters:
    alle_res (list): Een lijst met paren van (wedstrijd_data, voorspelling_resultaat).
    bestandsnaam (str): Het pad naar het te genereren HTML-bestand.
    """
    nu_nl = datetime.datetime.now(ZoneInfo("Europe/Amsterdam"))
    nu_str = nu_nl.strftime("%d-%m-%Y %H:%M")
    
    html_content = f"""<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WK 2026 Voorspellingen - Polymarket</title>
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
        <div class="subtitle">Wiskundig optimale uitslagen berekend op basis van live winstkansen van Polymarket data.</div>
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
        rho = res.get("rho", 0.0)
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
                    <span class="detail-value">{lam_h:.2f} - {lam_a:.2f} (ρ: {rho:.2f})</span>
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
        <p>Berekend met de Voetbalpoules Polymarket Voorspeller. Data ververst dagelijks om 17:00 CEST.</p>
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
    """
    Start de Polymarket-modus waarin de gebruiker live wedstrijden kan bekijken en voorspellen via de terminal.
    
    Parameters:
    toon_extra (bool, optioneel): Of er extra tie-breaker statistieken getoond moeten worden. Standaard False.
    output_file (str, optioneel): Bestand om voorspellingen naar te exporteren.
    """
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
                print(f"{CYAN}{BOLD}================================================================================================================================================================={RESET}")
                print(f"                                                               OVERZICHT ALLE VOORSPELDE WEDSTRIJDEN")
                print(f"{CYAN}{BOLD}================================================================================================================================================================={RESET}")
                print(f"{BOLD}{'Datum/Tijd':<17} | {'Thuisploeg':<20} vs. {'Uitploeg':<20} | {'Odds (1/X/2)':<18} | {'xG (Thuis-Uit)':<15} | {'rho':<6} | {'EV (pts)':<8} | {'Uitslag':<12} | {'Doelpuntenmaker Tips (MOTD)':<30}{RESET}")
                print("-" * 154)
                for m, res in alle_res:
                    kansen_str = f"{m['home_prob']:.0f}% / {m['draw_prob']:.0f}% / {m['away_prob']:.0f}%"
                    lam_h, lam_a = res["lambda"]
                    rho = res.get("rho", 0.0)
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
                        
                    print(f"{datum_str:<17} | {m['home']:<20} vs. {m['away']:<20} | {kansen_str:<18} | {xg_str:<15} | {rho:<6.2f} | {ev_val:<8.2f} | {GREEN}{BOLD}{advies_str:<12}{RESET} | {YELLOW}{scorer_str:<30}{RESET}")
                print(f"{CYAN}{BOLD}================================================================================================================================================================={RESET}\n")
                
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
    """
    Hoofdfunctie van het programma die de argumenten verwerkt en de juiste modus start.
    """
    parser = argparse.ArgumentParser(
        description="Berekent de wiskundig optimale uitslag voor voetbalpoules op basis van Polymarket 1X2 kansen."
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
