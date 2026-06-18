#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polymarket Data Fetcher en Teamnaam-normalisatie Module
------------------------------------------------------
Handelt het ophalen van wedstrijden en kansen af van de Polymarket API.
Inclusief robuuste foutafhandeling, retry-logica en teamnaam-normalisatie.
"""

import os
import re
import json
import math
import time
import requests
import difflib

from model.overround import normaliseer_kansen, normaliseer_kansen_power
from model.poisson import PARSING_WARNINGS

RESET = "\033[0m"
YELLOW = "\033[93m"

TEAM_NAME_MAP = {
    "nederland": "Netherlands",
    "netherlands": "Netherlands",
    "holland": "Netherlands",
    "engeland": "England",
    "england": "England",
    "kroatie": "Croatia",
    "kroatië": "Croatia",
    "croatia": "Croatia",
    "belgie": "Belgium",
    "belgië": "Belgium",
    "belgium": "Belgium",
    "duitsland": "Germany",
    "germany": "Germany",
    "frankrijk": "France",
    "france": "France",
    "spanje": "Spain",
    "spain": "Spain",
    "italie": "Italy",
    "italië": "Italy",
    "italy": "Italy",
    "portugal": "Portugal",
    "argentinie": "Argentina",
    "argentinië": "Argentina",
    "argentina": "Argentina",
    "brazilie": "Brazil",
    "brazilië": "Brazil",
    "brazil": "Brazil",
    "vs": "United States",
    "usa": "United States",
    "verenigde staten": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "saudi arabia": "Saudi Arabia",
    "saoedi-arabië": "Saudi Arabia",
    "saoedi arabie": "Saudi Arabia",
    "saoedi arabië": "Saudi Arabia",
    "saudi-arabia": "Saudi Arabia",
    "zuid-korea": "South Korea",
    "zuid korea": "South Korea",
    "south korea": "South Korea",
    "korea republic": "South Korea",
    "korea": "South Korea",
    "zuid-afrika": "South Africa",
    "zuid afrika": "South Africa",
    "south africa": "South Africa",
    "tsjechie": "Czechia",
    "tsjechië": "Czechia",
    "czechia": "Czechia",
    "czech republic": "Czechia",
    "tsjechische republiek": "Czechia",
    "nieuw-zeeland": "New Zealand",
    "nieuw zeeland": "New Zealand",
    "new zealand": "New Zealand",
    "egypte": "Egypt",
    "egypt": "Egypt",
    "algerije": "Algeria",
    "algeria": "Algeria",
    "colombia": "Colombia",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "congo-kinshasa": "DR Congo",
    "schotland": "Scotland",
    "scotland": "Scotland",
    "zweden": "Sweden",
    "sweden": "Sweden",
    "noorwegen": "Norway",
    "norway": "Norway",
    "uruguay": "Uruguay",
    "polen": "Poland",
    "poland": "Poland",
    "denemarken": "Denmark",
    "denmark": "Denmark",
    "servie": "Serbia",
    "servië": "Serbia",
    "serbia": "Serbia",
    "kameroen": "Cameroon",
    "cameroon": "Cameroon",
    "costa rica": "Costa Rica",
    "wales": "Wales",
    "marokko": "Morocco",
    "morocco": "Morocco",
    "australie": "Australia",
    "australië": "Australia",
    "australia": "Australia",
    "turkije": "Turkey",
    "turkey": "Turkey",
    "türkiye": "Turkey",
    "cabo verde": "Cape Verde",
    "cape verde": "Cape Verde",
    "kaapverdie": "Cape Verde",
    "kaapverdië": "Cape Verde",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "ivoorkust": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "ecuador": "Ecuador",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "tunesie": "Tunisia",
    "tunesië": "Tunisia",
    "tunisia": "Tunisia",
    "japan": "Japan",
    "iran": "Iran",
    "ir iran": "Iran",
    "islamic republic of iran": "Iran",
    "oostenrijk": "Austria",
    "austria": "Austria",
    "irak": "Iraq",
    "iraq": "Iraq",
    "senegal": "Senegal",
    "jordanie": "Jordan",
    "jordan": "Jordan",
    "oezbekistan": "Uzbekistan",
    "uzbekistan": "Uzbekistan",
    "ghana": "Ghana",
    "panama": "Panama",
    "haiti": "Haiti",
    "haïti": "Haiti",
    "switzerland": "Switzerland",
    "zwitserland": "Switzerland",
    "bosnie-herzegovina": "Bosnia and Herzegovina",
    "bosnië-herzegovina": "Bosnia and Herzegovina",
    "bosnie en herzegovina": "Bosnia and Herzegovina",
    "bosnië en herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "canada": "Canada",
    "qatar": "Qatar",
    "mexico": "Mexico"
}

def normalize_team_name(name):
    """
    Centraliseert teamnaam-normalisatie met een mapping voor alle WK-2026 en 2022 teams.
    """
    if not name:
        return ""
    n = name.strip().lower()
    
    # Simpele accentverwijdering
    for a, b in [("ë", "e"), ("é", "e"), ("è", "e"), ("ï", "i"), ("ç", "c"), ("ü", "u"), ("ä", "a"), ("ö", "o")]:
        n = n.replace(a, b)
        
    n = n.replace(".", "")
    
    if n in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[n]
        
    # fallback loop voor keys na accent-opschoning
    cleaned_map = {}
    for k, v in TEAM_NAME_MAP.items():
        ck = k
        for a, b in [("ë", "e"), ("é", "e"), ("è", "e"), ("ï", "i"), ("ç", "c"), ("ü", "u"), ("ä", "a"), ("ö", "o")]:
            ck = ck.replace(a, b)
        ck = ck.replace(".", "")
        cleaned_map[ck] = v
        
    if n in cleaned_map:
        return cleaned_map[n]
        
    return name.strip()

def match_team(candidate_name, home_team, away_team):
    """
    Matches candidate_name against home_team and away_team using exact match
    after normalization and fuzzy match fallback.
    """
    cand_norm = normalize_team_name(candidate_name)
    home_norm = normalize_team_name(home_team)
    away_norm = normalize_team_name(away_team)
    
    if cand_norm == home_norm:
        return "home"
    if cand_norm == away_norm:
        return "away"
        
    # Fallback fuzzy matching
    ratio_home = difflib.SequenceMatcher(None, cand_norm.lower(), home_norm.lower()).ratio()
    ratio_away = difflib.SequenceMatcher(None, cand_norm.lower(), away_norm.lower()).ratio()
    
    if ratio_home >= 0.8 or ratio_away >= 0.8:
        if ratio_home > ratio_away:
            return "home"
        else:
            return "away"
            
    return None

def is_motd_match(home, away, motd_pairs):
    """
    Controleert of de gegeven teams overeenkomen met een van de 'Wedstrijden van de Dag' (MOTD) uit de lijst.
    """
    home_norm = normalize_team_name(home)
    away_norm = normalize_team_name(away)
    
    for team_a, team_b in motd_pairs:
        ta_norm = normalize_team_name(team_a)
        tb_norm = normalize_team_name(team_b)
        if (home_norm == ta_norm and away_norm == tb_norm) or (home_norm == tb_norm and away_norm == ta_norm):
            return True
            
    return False

def selecteer_en_normaliseer_lijn(raw_data, est_total, overround_method, type_label, home_team, away_team, verbose=False):
    """
    Selecteert maximaal 1 lijn uit raw_data op basis van liquiditeit en doelpuntenverwachting.
    """
    if not raw_data:
        return {}
        
    candidates = []
    for line, info in raw_data.items():
        u = info['under']
        o = info['over']
        has_both = (u is not None) and (o is not None)
        combined_liq = info['under_liq'] + info['over_liq']
        diff = abs(line - est_total)
        candidates.append((line, u, o, has_both, combined_liq, diff, info))
        
    candidates.sort(key=lambda x: (-int(x[3]), -x[4], x[5]))
    
    best = candidates[0]
    best_line = best[0]
    best_u = best[1]
    best_o = best[2]
    best_has_both = best[3]
    best_liq = best[4]
    best_diff = best[5]
    best_info = best[6]
    
    if verbose:
        print(f"  [Verbose O/U] Selectie voor '{type_label}' ({home_team} vs. {away_team}):")
        print(f"    -> Geselecteerd: Lijn {best_line} (beide kanten: {best_has_both}, liquiditeit: {best_liq:.2f}, diff: {best_diff:.4f})")
        for c in candidates[1:]:
            print(f"       Genegeerd: Lijn {c[0]} (beide kanten: {c[3]}, liquiditeit: {c[4]:.2f}, diff: {c[5]:.4f})")
            
    if best_u is not None and best_o is not None:
        if overround_method == "power":
            norm = normaliseer_kansen_power([best_u, best_o])
            u_norm, o_norm = norm[0], norm[1]
        else:
            tot = best_u + best_o
            u_norm, o_norm = best_u/tot, best_o/tot
    elif best_o is not None:
        u_norm, o_norm = 1.0 - best_o, best_o
    else:  # best_u is not None
        u_norm, o_norm = best_u, 1.0 - best_u
        
    spreads = []
    if best_info['under_spread'] is not None:
        spreads.append(best_info['under_spread'])
    if best_info['over_spread'] is not None:
        spreads.append(best_info['over_spread'])
    avg_spread = sum(spreads) / len(spreads) if spreads else None
    
    return {best_line: (u_norm, o_norm, avg_spread)}

def haal_polymarket_wedstrijden(overround_method="power", verbose=False, motd_file="data/motd.json"):
    """
    Haalt live WK-wedstrijden en bijbehorende kansen op via de Polymarket API.
    """
    motd_pairs = []
    if motd_file:
        try:
            if os.path.exists(motd_file):
                with open(motd_file, 'r', encoding='utf-8') as f:
                    motd_pairs = json.load(f)
            elif verbose:
                print(f"  [Debug] MOTD-bestand '{motd_file}' niet gevonden.")
        except Exception as e:
            print(f"{YELLOW}⚠️ Waarschuwing: Kon MOTD-bestand '{motd_file}' niet laden: {e}{RESET}")

    url = "https://gamma-api.polymarket.com/events"
    params = {
        "tag_id": 100350,
        "active": "true",
        "closed": "false",
        "limit": 100
    }
    
    r = None
    max_retries = 3
    delay = 1.0
    last_err = None
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                break
            else:
                last_err = f"statuscode {r.status_code}"
        except requests.exceptions.RequestException as err:
            last_err = str(err)
        
        if attempt < max_retries - 1:
            if verbose:
                print(f"  [Debug] API-poging {attempt+1} mislukt ({last_err}). Opnieuw proberen in {delay}s...")
            time.sleep(delay)
            delay *= 2
            
    if r is None or r.status_code != 200:
        return None, f"Fout bij ophalen Polymarket data na {max_retries} pogingen: {last_err}"
        
    try:
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
            ou_raw = {}
            team_ou_home_raw = {}
            team_ou_away_raw = {}
            non_draw_markets = []
            btts_markets = []
            cs_home_markets = []
            cs_away_markets = []
            
            for m in markets:
                q = m.get("question", "").lower()
                prices_str = m.get("outcomePrices")
                if not prices_str:
                    continue
                prices = json.loads(prices_str)
                if len(prices) < 1:
                    continue
                yes_price = float(prices[0])
                no_price = float(prices[1]) if len(prices) >= 2 else 1.0 - yes_price
                liq = float(m.get("liquidityNum") or m.get("liquidity") or 0.0)
                try:
                    spread = float(m.get("spread")) if m.get("spread") is not None else None
                except (ValueError, TypeError):
                    spread = None
                
                is_team_ou = False
                match_team_ou = re.search(
                    r'(?:will\s+)?(.+?)\s+(?:score\s+)?(over|under)\s+(\d+\.5)\s+goals'
                    r'|'
                    r'(over|under)\s+(\d+\.5)\s+goals?\s+(?:for\s+)?(.+)',
                    q
                )
                if match_team_ou:
                    if match_team_ou.group(1) is not None:
                        team_name = match_team_ou.group(1).strip()
                        type_ou = match_team_ou.group(2)
                        line = float(match_team_ou.group(3))
                    else:
                        team_name = match_team_ou.group(6).strip()
                        type_ou = match_team_ou.group(4)
                        line = float(match_team_ou.group(5))
                    
                    team_name = team_name.rstrip('?').strip()
                    matched = match_team(team_name, home_team, away_team)
                    
                    if matched == 'home':
                        if line not in team_ou_home_raw:
                            team_ou_home_raw[line] = {'under': None, 'over': None, 'under_liq': 0.0, 'over_liq': 0.0, 'under_spread': None, 'over_spread': None}
                        if type_ou == 'under':
                            team_ou_home_raw[line]['under'] = yes_price
                            team_ou_home_raw[line]['under_liq'] = liq
                            team_ou_home_raw[line]['under_spread'] = spread
                        else:
                            team_ou_home_raw[line]['over'] = yes_price
                            team_ou_home_raw[line]['over_liq'] = liq
                            team_ou_home_raw[line]['over_spread'] = spread
                        is_team_ou = True
                    elif matched == 'away':
                        if line not in team_ou_away_raw:
                            team_ou_away_raw[line] = {'under': None, 'over': None, 'under_liq': 0.0, 'over_liq': 0.0, 'under_spread': None, 'over_spread': None}
                        if type_ou == 'under':
                            team_ou_away_raw[line]['under'] = yes_price
                            team_ou_away_raw[line]['under_liq'] = liq
                            team_ou_away_raw[line]['under_spread'] = spread
                        else:
                            team_ou_away_raw[line]['over'] = yes_price
                            team_ou_away_raw[line]['over_liq'] = liq
                            team_ou_away_raw[line]['over_spread'] = spread
                        is_team_ou = True
                
                if is_team_ou:
                    pass
                else:
                    match_ou = re.search(r'(over|under) (\d+\.5) goals', q)
                    if match_ou:
                        type_ou = match_ou.group(1)
                        line = float(match_ou.group(2))
                        if line not in ou_raw:
                            ou_raw[line] = {'under': None, 'over': None, 'under_liq': 0.0, 'over_liq': 0.0, 'under_spread': None, 'over_spread': None}
                        if type_ou == 'under':
                            ou_raw[line]['under'] = yes_price
                            ou_raw[line]['under_liq'] = liq
                            ou_raw[line]['under_spread'] = spread
                        else:
                            ou_raw[line]['over'] = yes_price
                            ou_raw[line]['over_liq'] = liq
                            ou_raw[line]['over_spread'] = spread
                    elif "both teams to score" in q:
                        btts_markets.append((yes_price, no_price, liq, spread, m.get("question", "")))
                    elif any(term in q for term in ["clean sheet", "win to nil", "win to-nil", "shutout"]):
                        matched = None
                        q_norm = q.lower()
                        home_norm = normalize_team_name(home_team).lower()
                        away_norm = normalize_team_name(away_team).lower()
                        
                        if home_norm in q_norm:
                            matched = "home"
                        elif away_norm in q_norm:
                            matched = "away"
                        else:
                            words = re.findall(r'\w+', q)
                            matched_home_fuzzy = False
                            matched_away_fuzzy = False
                            for n_words in range(1, 4):
                                for i in range(len(words) - n_words + 1):
                                    sub_phrase = " ".join(words[i:i+n_words])
                                    sub_norm = normalize_team_name(sub_phrase).lower()
                                    if difflib.SequenceMatcher(None, sub_norm, home_norm).ratio() >= 0.8:
                                        matched_home_fuzzy = True
                                    if difflib.SequenceMatcher(None, sub_norm, away_norm).ratio() >= 0.8:
                                        matched_away_fuzzy = True
                            if matched_home_fuzzy and not matched_away_fuzzy:
                                matched = "home"
                            elif matched_away_fuzzy and not matched_home_fuzzy:
                                matched = "away"
                                
                        if matched == "home":
                            cs_home_markets.append((yes_price, no_price, liq, spread, m.get("question", "")))
                        elif matched == "away":
                            cs_away_markets.append((yes_price, no_price, liq, spread, m.get("question", "")))
                    elif "draw" in q:
                        draw_prob = yes_price
                    else:
                        non_draw_markets.append((q, yes_price))
                    
            if len(non_draw_markets) >= 2:
                best_m_home = best_m_away = None
                
                for mq, mp in non_draw_markets:
                    matched = None
                    mq_norm = mq.lower()
                    home_norm = normalize_team_name(home_team).lower()
                    away_norm = normalize_team_name(away_team).lower()
                    
                    if home_norm in mq_norm:
                        matched = "home"
                    elif away_norm in mq_norm:
                        matched = "away"
                    else:
                        words = re.findall(r'\w+', mq)
                        matched_home_fuzzy = False
                        matched_away_fuzzy = False
                        for n_words in range(1, 4):
                            for i in range(len(words) - n_words + 1):
                                sub_phrase = " ".join(words[i:i+n_words])
                                sub_norm = normalize_team_name(sub_phrase).lower()
                                if difflib.SequenceMatcher(None, sub_norm, home_norm).ratio() >= 0.8:
                                    matched_home_fuzzy = True
                                if difflib.SequenceMatcher(None, sub_norm, away_norm).ratio() >= 0.8:
                                    matched_away_fuzzy = True
                        if matched_home_fuzzy and not matched_away_fuzzy:
                            matched = "home"
                        elif matched_away_fuzzy and not matched_home_fuzzy:
                            matched = "away"
                            
                    if matched == "home":
                        best_m_home = mp
                    elif matched == "away":
                        best_m_away = mp
                        
                if best_m_home is not None and best_m_away is not None:
                    home_prob = best_m_home
                    away_prob = best_m_away
 
            if home_prob is not None and draw_prob is not None and away_prob is not None:
                p_h = home_prob
                p_d = draw_prob
                p_a = away_prob
                
                try:
                    norm_h, norm_d, norm_a = normaliseer_kansen(p_h * 100.0, p_d * 100.0, p_a * 100.0, method=overround_method)
                    sum_norm = norm_h + norm_d + norm_a
                    if abs(sum_norm - 1.0) > 1e-4:
                        print(f"{YELLOW}⚠️ Waarschuwing: Genormaliseerde 1X2-kansen sommeren niet tot 100% voor {home_team} vs. {away_team}: {sum_norm*100:.2f}%{RESET}")
                except Exception as val_err:
                    if verbose:
                        print(f"  [Debug Val] Fout bij kansen-normalisatie-check: {val_err}")
                
                est_lambda_h = -math.log(max(0.01, min(0.99, p_d + p_a)))
                est_lambda_a = -math.log(max(0.01, min(0.99, p_h + p_d)))
                est_total_match = est_lambda_h + est_lambda_a
                
                final_ou = selecteer_en_normaliseer_lijn(
                    ou_raw, est_total_match, overround_method,
                    "Wedstrijd O/U", home_team, away_team, verbose=verbose
                )
                
                final_team_ou_home = selecteer_en_normaliseer_lijn(
                    team_ou_home_raw, est_lambda_h, overround_method,
                    "Team O/U Thuis", home_team, away_team, verbose=verbose
                )
                
                final_team_ou_away = selecteer_en_normaliseer_lijn(
                    team_ou_away_raw, est_lambda_a, overround_method,
                    "Team O/U Uit", home_team, away_team, verbose=verbose
                )
                
                btts_prob = None
                if btts_markets:
                    btts_markets.sort(key=lambda x: -x[2])
                    best_btts = btts_markets[0]
                    yes_p, no_p = best_btts[0], best_btts[1]
                    if overround_method == "power":
                        norm = normaliseer_kansen_power([yes_p, no_p])
                        btts_prob = norm[0]
                    else:
                        tot = yes_p + no_p
                        btts_prob = yes_p / tot if tot > 0 else yes_p
                    if verbose:
                        print(f"  [Verbose Extra] BTTS geselecteerd voor {home_team} vs. {away_team}: '{best_btts[4]}' (kans: {btts_prob*100:.1f}%, liq: {best_btts[2]:.2f})")
 
                clean_sheet_home_prob = None
                if cs_home_markets:
                    cs_home_markets.sort(key=lambda x: -x[2])
                    best_cs_h = cs_home_markets[0]
                    yes_p, no_p = best_cs_h[0], best_cs_h[1]
                    if overround_method == "power":
                        norm = normaliseer_kansen_power([yes_p, no_p])
                        clean_sheet_home_prob = norm[0]
                    else:
                        tot = yes_p + no_p
                        clean_sheet_home_prob = yes_p / tot if tot > 0 else yes_p
                    if verbose:
                        print(f"  [Verbose Extra] CS Thuis geselecteerd voor {home_team} vs. {away_team}: '{best_cs_h[4]}' (kans: {clean_sheet_home_prob*100:.1f}%, liq: {best_cs_h[2]:.2f})")
 
                clean_sheet_away_prob = None
                if cs_away_markets:
                    cs_away_markets.sort(key=lambda x: -x[2])
                    best_cs_a = cs_away_markets[0]
                    yes_p, no_p = best_cs_a[0], best_cs_a[1]
                    if overround_method == "power":
                        norm = normaliseer_kansen_power([yes_p, no_p])
                        clean_sheet_away_prob = norm[0]
                    else:
                        tot = yes_p + no_p
                        clean_sheet_away_prob = yes_p / tot if tot > 0 else yes_p
                    if verbose:
                        print(f"  [Verbose Extra] CS Uit geselecteerd voor {home_team} vs. {away_team}: '{best_cs_a[4]}' (kans: {clean_sheet_away_prob*100:.1f}%, liq: {best_cs_a[2]:.2f})")
 
                parsed_matches.append({
                    "title": team_part,
                    "home": home_team,
                    "away": away_team,
                    "home_prob": home_prob * 100.0,
                    "draw_prob": draw_prob * 100.0,
                    "away_prob": away_prob * 100.0,
                    "ou_probs": final_ou,
                    "team_ou_home": final_team_ou_home,
                    "team_ou_away": final_team_ou_away,
                    "btts_prob": btts_prob,
                    "clean_sheet_home_prob": clean_sheet_home_prob,
                    "clean_sheet_away_prob": clean_sheet_away_prob,
                    "date": e.get("endDate", ""),
                    "is_motd": is_motd_match(home_team, away_team, motd_pairs)
                })
            else:
                print(f"{YELLOW}⚠️ Waarschuwing: Wedstrijd {home_team} vs. {away_team} mist 1X2-data na parsing!{RESET}")
                
        matched_indices = set()
        for m in parsed_matches:
            h_norm = normalize_team_name(m["home"])
            a_norm = normalize_team_name(m["away"])
            for idx, (ta, tb) in enumerate(motd_pairs):
                ta_norm = normalize_team_name(ta)
                tb_norm = normalize_team_name(tb)
                if (h_norm == ta_norm and a_norm == tb_norm) or (h_norm == tb_norm and a_norm == ta_norm):
                    matched_indices.add(idx)
                    
        for idx, (ta, tb) in enumerate(motd_pairs):
            if idx not in matched_indices:
                print(f"{YELLOW}⚠️ Waarschuwing: MOTD-pair '{ta}' vs. '{tb}' niet gevonden in de Polymarket-batch!{RESET}")
                
        parsed_matches.sort(key=lambda x: x["date"])
        return parsed_matches, None
    except Exception as err:
        return None, f"Fout bij verwerking van Polymarket data: {err}"
