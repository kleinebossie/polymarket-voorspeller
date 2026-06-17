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
import random
import datetime
from zoneinfo import ZoneInfo
import requests
from scipy.optimize import minimize

# Standaard optimalisatie-gewichten voor Poisson lambda-bepaling (fit)
# Gevonden via grid-search backtest op 41 historische wedstrijden (inclusief WK 2022 groepsfase).
# De combinatie Match O/U = 1.0 en Team O/U = 0.5 gaf de hoogste gemiddelde score (4.385 punten per wedstrijd).
WEIGHT_MATCH_OU = 1.0
WEIGHT_TEAM_OU = 0.5
WEIGHT_EXTRA_MARKETS = 0.6
SCORER_HIT_RATE = 0.35

# Lichte Tikhonov-regularisatie op de Dixon-Coles ρ-parameter.
# Met alléén 1X2-kansen (2 vrije targets) is het model met 3 parameters (λ_h, λ_a, ρ)
# onderbepaald: er bestaat een vlakke 'ridge' waarlangs ρ vrij kan schuiven zonder de
# 1X2-fit te veranderen. Verschillende optimizers (L-BFGS-B vs. Nelder-Mead) glijden dan
# naar verschillende uiteinden van die ridge. Een minieme straf op ρ² maakt het minimum
# uniek (ρ → 0 zodra de data ρ niet vastlegt) zonder goed-bepaalde fits (met O/U-data)
# merkbaar te verschuiven. Dit is essentieel voor de Python/JS-pariteit (zie test_js_parity.py).
RHO_REG = 1e-3

# ---------------------------------------------------------------------------
# Tie-breaker (toernooi-vragen) model-constanten
# ---------------------------------------------------------------------------
# Deze waarden zijn configureerbaar via de CLI (--yellow-base-min, --red-card-rate)
# en worden gebruikt door bereken_tie_breakers() om de tie-breaker-minuten af te
# leiden uit het Poisson-model i.p.v. uit hardcoded constanten.
#
# Eerste gele kaart: op grote toernooien (WK/EK) valt de eerste gele kaart gemiddeld
# rond de 30e minuut (bron: Opta/FIFA wedstrijdsamenvattingen WK 2018-2022). We schalen
# deze basis-mediaan licht met de wedstrijdintensiteit (λ_total = λ_h + λ_a): een opener,
# doelpuntrijkere wedstrijd kent gemiddeld iets eerder de eerste kaart.
FIRST_YELLOW_BASE_MIN = 30.0      # referentie-mediaan (minuut) bij een gemiddelde wedstrijd
FIRST_YELLOW_REF_LAMBDA = 2.6     # referentie λ_total waarbij de basis-minuut exact geldt

# Eerste rode kaart: zeer zeldzaam. Op het WK vallen er gemiddeld ~0.22 rode kaarten
# (direct of 2x geel) per wedstrijd. We modelleren P(rode kaart) = 1 - exp(-rate).
# Ligt die kans onder de drempel, dan adviseren we "geen rode kaart" (minuut > 90).
RED_CARD_RATE = 0.22
RED_CARD_THRESHOLD = 0.5


def bereken_tie_breakers(lam_h, lam_a, yellow_base_min=None, yellow_ref_lambda=None,
                         red_card_rate=None, red_card_threshold=None):
    """
    Leidt de toernooi-tie-breaker-voorspellingen af uit het Poisson-model i.p.v. uit
    vaste constanten. Hierdoor varieert de uitkomst per wedstrijd op basis van de λ's.

    Model:
    - Eerste doelpunt-minuut: doelpunten arriveren als een Poisson-proces over 90 minuten
      met totale rate λ_total = λ_h + λ_a. De mediaan-minuut van de exponentiële verdeling
      van het eerste doelpunt is 90 * (1 - ln(2)) / λ_total, geklemd tussen 1 en 90.
      Hoge-xG-wedstrijden geven zo een vroegere minuut, lage-xG-wedstrijden een latere.
    - Eerste gele kaart: een historisch WK-gemiddelde (~30e min) dat licht inverse met
      de wedstrijdintensiteit (λ_total) wordt geschaald.
    - Eerste rode kaart: zeldzaam event. Als P(rode kaart) onder de drempel ligt, luidt
      het advies "geen rode kaart" (minuut > 90); anders een mediaan-minuut-schatting.

    Parameters:
    lam_h (float): Verwachte doelpunten thuisploeg (λ_h).
    lam_a (float): Verwachte doelpunten uitploeg (λ_a).
    yellow_base_min (float, optioneel): Basis-mediaan eerste gele kaart. Standaard FIRST_YELLOW_BASE_MIN.
    yellow_ref_lambda (float, optioneel): Referentie λ_total. Standaard FIRST_YELLOW_REF_LAMBDA.
    red_card_rate (float, optioneel): Verwacht aantal rode kaarten per wedstrijd. Standaard RED_CARD_RATE.
    red_card_threshold (float, optioneel): Drempel waaronder "geen rode kaart" wordt geadviseerd.

    Returns:
    dict: De berekende minuten met korte uitleg per tie-breaker.
    """
    if yellow_base_min is None:
        yellow_base_min = FIRST_YELLOW_BASE_MIN
    if yellow_ref_lambda is None:
        yellow_ref_lambda = FIRST_YELLOW_REF_LAMBDA
    if red_card_rate is None:
        red_card_rate = RED_CARD_RATE
    if red_card_threshold is None:
        red_card_threshold = RED_CARD_THRESHOLD

    lam_total = max(lam_h + lam_a, 1e-9)

    # 1) Eerste doelpunt: exponentiële mediaan over 90 minuten.
    eerste_doelpunt = round(90.0 * (1.0 - math.log(2.0)) / lam_total)
    eerste_doelpunt = max(1, min(90, eerste_doelpunt))

    # 2) Eerste gele kaart: schaal de basis-mediaan inverse met de intensiteit.
    gele_kaart = round(yellow_base_min * (yellow_ref_lambda / lam_total))
    gele_kaart = max(1, min(90, gele_kaart))

    # 3) Eerste rode kaart: schaal de rate licht met de intensiteit en bepaal de kans.
    eff_red_rate = red_card_rate * (lam_total / yellow_ref_lambda)
    p_rode_kaart = 1.0 - math.exp(-eff_red_rate)
    if p_rode_kaart < red_card_threshold:
        rode_kaart_minuut = None  # geen rode kaart verwacht
        rode_kaart_uitleg = (f"P(rode kaart) ≈ {p_rode_kaart*100:.0f}% < {red_card_threshold*100:.0f}% "
                             f"→ geen rode kaart verwacht (> 90e min).")
    else:
        rode_kaart_minuut = max(1, min(90, round(90.0 * (1.0 - math.log(2.0)) / eff_red_rate)))
        rode_kaart_uitleg = f"P(rode kaart) ≈ {p_rode_kaart*100:.0f}%; verwachte mediaan-minuut."

    return {
        "eerste_doelpunt_minuut": eerste_doelpunt,
        "eerste_doelpunt_uitleg": f"Exponentiële mediaan over 90 min bij λ_total = {lam_total:.2f}.",
        "gele_kaart_minuut": gele_kaart,
        "gele_kaart_uitleg": f"Historisch WK-gemiddelde ({yellow_base_min:.0f}e min) geschaald met intensiteit.",
        "rode_kaart_minuut": rode_kaart_minuut,
        "rode_kaart_kans": p_rode_kaart,
        "rode_kaart_uitleg": rode_kaart_uitleg,
    }



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

def normaliseer_kansen_power(kansen, target_sum=1.0):
    """
    Normaliseert een lijst van kansen naar een doelsom (standaard 1.0)
    met behulp van de power-methode (pi_i^k).
    """
    # Filter of clip eventuele negatieve kansen of zeros
    kansen = [max(0.0001, min(0.9999, float(p))) for p in kansen]
    som = sum(kansen)
    if som == 0:
        return [1.0 / len(kansen)] * len(kansen)
    
    # Als er maar 1 kans is, stel deze direct in op de doelsom
    if len(kansen) == 1:
        return [target_sum]
        
    # Als de som al heel dicht bij target_sum ligt, return direct
    if abs(som - target_sum) < 1e-9:
        return [p * (target_sum / som) for p in kansen]
        
    # Root finding voor k met bisection method
    # We willen k vinden waarvoor sum(p_i^k) = target_sum
    if som > target_sum:
        # Als de som te groot is, hebben we k > 1.0 nodig
        k_low = 1.0
        k_high = 2.0
        # Vind een bovengrens voor k
        for _ in range(50):
            s = sum(p**k_high for p in kansen)
            if s < target_sum:
                break
            k_high *= 2.0
    else:
        # Als de som te klein is, hebben we k < 1.0 nodig
        k_low = 0.001
        k_high = 1.0
        # Vind een ondergrens voor k
        for _ in range(50):
            s = sum(p**k_low for p in kansen)
            if s > target_sum:
                break
            k_low /= 2.0

    # Bisection loop
    for _ in range(100):
        k_mid = (k_low + k_high) / 2.0
        s = sum(p**k_mid for p in kansen)
        if abs(s - target_sum) < 1e-12:
            k = k_mid
            break
        if s > target_sum:
            k_low = k_mid
        else:
            k_high = k_mid
    else:
        k = (k_low + k_high) / 2.0

    result = [p**k for p in kansen]
    # Breng eventuele zeer kleine afrondingsfoutjes in lijn met target_sum
    s_res = sum(result)
    if s_res > 0:
        result = [r * (target_sum / s_res) for r in result]
    return result

def normaliseer_kansen(home, draw, away, method="power"):
    """
    Zorgt ervoor dat de drie kansen (thuis, gelijk, uit) samen exact 1.0 worden.
    Ondersteunt zowel lineaire als power-normalisatie.
    
    Parameters:
    home (float): De ingevoerde kans op thuiswinst.
    draw (float): De ingevoerde kans op een gelijkspel.
    away (float): De ingevoerde kans op uitwinst.
    method (str, optioneel): De normalisatiemethode ('linear' of 'power'). Standaard 'power'.
    
    Returns:
    tuple: Een drietal met de genormaliseerde kansen (thuis, gelijk, uit) die optellen tot 1.0.
    """
    # Bepaal de schaal: als de som > 1.5 of een van de waardes > 1.0, schaal dan naar 0-1.
    is_percentage = (home > 1.0 or draw > 1.0 or away > 1.0 or (home + draw + away) > 1.5)
    
    h = home / 100.0 if is_percentage else home
    d = draw / 100.0 if is_percentage else draw
    a = away / 100.0 if is_percentage else away
    
    if method == "power":
        norm = normaliseer_kansen_power([h, d, a], target_sum=1.0)
        return norm[0], norm[1], norm[2]
    else:
        totaal = h + d + a
        if totaal == 0:
            return 0.0, 0.0, 0.0
        return h / totaal, d / totaal, a / totaal

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

def bepaal_poisson_lambdas(target_h, target_d, target_a, target_ou=None, target_team_ou_home=None, target_team_ou_away=None, target_btts=None, target_clean_sheet_home=None, target_clean_sheet_away=None, loss_type="logloss", verbose=False, weight_match_ou=None, weight_team_ou=None, weight_extra_markets=None):
    """
    Vindt de optimale Poisson-lambda's en de Dixon-Coles rho-waarde die het beste aansluiten
    bij de gewenste winst-, gelijkspel- en verlieskansen (en eventuele over/under kansen).
    Maakt gebruik van L-BFGS-B met expliciete bounds en multi-start om lokale minima te vermijden.
    
    Parameters:
    target_h (float): De gewenste (genormaliseerde) kans op thuiswinst.
    target_d (float): De gewenste (genormaliseerde) kans op gelijkspel.
    target_a (float): De gewenste (genormaliseerde) kans op uitwinst.
    target_ou (dict, optioneel): Kansen voor over/under doelpuntengrenzen.
    target_team_ou_home (dict, optioneel): Kansen voor team-specifieke over/under grenzen (thuis).
    target_team_ou_away (dict, optioneel): Kansen voor team-specifieke over/under grenzen (uit).
    target_btts (float, optioneel): Kansen voor beide teams scoren (BTTS).
    target_clean_sheet_home (float, optioneel): Kansen voor clean sheet van thuisploeg.
    target_clean_sheet_away (float, optioneel): Kansen voor clean sheet van uitploeg.
    loss_type (str, optioneel): De te gebruiken verliesfunctie ('mse' of 'logloss'). Standaard 'logloss'.
    verbose (bool, optioneel): Of er debug-logging getoond moet worden voor de fits. Standaard False.
    weight_match_ou (float, optioneel): Het gewicht voor de wedstrijd Over/Under fit-termen.
    weight_team_ou (float, optioneel): Het gewicht voor de team Over/Under fit-termen.
    weight_extra_markets (float, optioneel): Het gewicht voor de extra markten fit-termen.
    
    Returns:
    tuple: Een drietal met de berekende parameters (lambda_thuis, lambda_uit, rho).
    """
    if weight_match_ou is None:
        weight_match_ou = WEIGHT_MATCH_OU
    if weight_team_ou is None:
        weight_team_ou = WEIGHT_TEAM_OU
    if weight_extra_markets is None:
        weight_extra_markets = WEIGHT_EXTRA_MARKETS

    if verbose:
        extra_targets = []
        if target_btts is not None:
            extra_targets.append(f"BTTS: {target_btts:.4f}")
        if target_clean_sheet_home is not None:
            extra_targets.append(f"CS Thuis: {target_clean_sheet_home:.4f}")
        if target_clean_sheet_away is not None:
            extra_targets.append(f"CS Uit: {target_clean_sheet_away:.4f}")
        if extra_targets:
            print(f"  [Debug Fit] Extra targets gebruikt in fit: {', '.join(extra_targets)} (gewicht: {weight_extra_markets:.2f})")

    def objective(params):
        lh, la, r = params
        
        matrix = calc_matrix(lh, la, r)
        h, d, a = get_1x2_and_ou(matrix)
        
        eps = 1e-15
        if loss_type == "logloss":
            # Cross-entropy loss voor 1X2
            error = -(
                target_h * math.log(max(h, eps)) +
                target_d * math.log(max(d, eps)) +
                target_a * math.log(max(a, eps))
            )
        else:
            # MSE loss voor 1X2
            error = (h - target_h)**2 + (d - target_d)**2 + (a - target_a)**2
        
        if target_ou:
            use_relative_weights = len(target_ou) > 1
            for line, values in target_ou.items():
                t_u, t_o = values[0], values[1]
                u = sum(p for (sc_h,sc_a), p in matrix.items() if sc_h+sc_a < line)
                o = sum(p for (sc_h,sc_a), p in matrix.items() if sc_h+sc_a > line)
                
                weight_factor = 1.0
                if use_relative_weights:
                    if len(values) >= 3 and values[2] is not None:
                        weight_factor = 1.0 / max(values[2], 1e-4)
                    else:
                        tot = t_u + t_o
                        if tot > 0:
                            p_u = t_u / tot
                            p_o = t_o / tot
                            vol = math.sqrt(max(p_u * p_o, 1e-6))
                            weight_factor = 0.5 / vol
                
                if loss_type == "logloss":
                    error += -(t_u * math.log(max(u, eps)) + t_o * math.log(max(o, eps))) * weight_match_ou * weight_factor
                else:
                    error += ((u - t_u)**2 + (o - t_o)**2) * weight_match_ou * weight_factor
                
        # Team totals thuisploeg: vergelijk de marginale thuisdoelpuntverdeling
        if target_team_ou_home:
            use_relative_weights = len(target_team_ou_home) > 1
            for line, values in target_team_ou_home.items():
                t_u, t_o = values[0], values[1]
                u = sum(p for (sc_h, sc_a), p in matrix.items() if sc_h < line)
                o = sum(p for (sc_h, sc_a), p in matrix.items() if sc_h > line)
                
                weight_factor = 1.0
                if use_relative_weights:
                    if len(values) >= 3 and values[2] is not None:
                        weight_factor = 1.0 / max(values[2], 1e-4)
                    else:
                        tot = t_u + t_o
                        if tot > 0:
                            p_u = t_u / tot
                            p_o = t_o / tot
                            vol = math.sqrt(max(p_u * p_o, 1e-6))
                            weight_factor = 0.5 / vol
                
                if loss_type == "logloss":
                    error += -(t_u * math.log(max(u, eps)) + t_o * math.log(max(o, eps))) * weight_team_ou * weight_factor
                else:
                    error += ((u - t_u)**2 + (o - t_o)**2) * weight_team_ou * weight_factor
                
        # Team totals uitploeg: vergelijk de marginale uitdoelpuntverdeling
        if target_team_ou_away:
            use_relative_weights = len(target_team_ou_away) > 1
            for line, values in target_team_ou_away.items():
                t_u, t_o = values[0], values[1]
                u = sum(p for (sc_h, sc_a), p in matrix.items() if sc_a < line)
                o = sum(p for (sc_h, sc_a), p in matrix.items() if sc_a > line)
                
                weight_factor = 1.0
                if use_relative_weights:
                    if len(values) >= 3 and values[2] is not None:
                        weight_factor = 1.0 / max(values[2], 1e-4)
                    else:
                        tot = t_u + t_o
                        if tot > 0:
                            p_u = t_u / tot
                            p_o = t_o / tot
                            vol = math.sqrt(max(p_u * p_o, 1e-6))
                            weight_factor = 0.5 / vol
                
                if loss_type == "logloss":
                    error += -(t_u * math.log(max(u, eps)) + t_o * math.log(max(o, eps))) * weight_team_ou * weight_factor
                else:
                    error += ((u - t_u)**2 + (o - t_o)**2) * weight_team_ou * weight_factor
                    
        # Both Teams to Score (BTTS) fit
        if target_btts is not None:
            p_btts = sum(p for (sc_h, sc_a), p in matrix.items() if sc_h > 0 and sc_a > 0)
            if loss_type == "logloss":
                error += -(target_btts * math.log(max(p_btts, eps)) + (1.0 - target_btts) * math.log(max(1.0 - p_btts, eps))) * weight_extra_markets
            else:
                error += ((p_btts - target_btts)**2) * weight_extra_markets

        # Clean Sheet Thuis: P(a=0)
        if target_clean_sheet_home is not None:
            p_cs_h = sum(p for (sc_h, sc_a), p in matrix.items() if sc_a == 0)
            if loss_type == "logloss":
                error += -(target_clean_sheet_home * math.log(max(p_cs_h, eps)) + (1.0 - target_clean_sheet_home) * math.log(max(1.0 - p_cs_h, eps))) * weight_extra_markets
            else:
                error += ((p_cs_h - target_clean_sheet_home)**2) * weight_extra_markets

        # Clean Sheet Uit: P(h=0)
        if target_clean_sheet_away is not None:
            p_cs_a = sum(p for (sc_h, sc_a), p in matrix.items() if sc_h == 0)
            if loss_type == "logloss":
                error += -(target_clean_sheet_away * math.log(max(p_cs_a, eps)) + (1.0 - target_clean_sheet_away) * math.log(max(1.0 - p_cs_a, eps))) * weight_extra_markets
            else:
                error += ((p_cs_a - target_clean_sheet_away)**2) * weight_extra_markets

        # Lichte ρ-regularisatie: maakt het minimum uniek bij onderbepaalde (1X2-only) fits.
        error += RHO_REG * (r ** 2)

        return error

    # Optimalisatie bounds
    bounds = [
        (0.05, 5.0),    # lam_h
        (0.05, 5.0),    # lam_a
        (-0.25, 0.10)   # rho
    ]

    # Genereer 5 startpunten:
    # 1. Standaard startpunt
    start_points = [[1.3, 1.0, -0.05]]
    
    # 2. Maher-analytische schatting
    m_h = -math.log(max(0.01, min(0.99, target_d + target_a)))
    m_a = -math.log(max(0.01, min(0.99, target_h + target_d)))
    m_h = max(0.05, min(m_h, 5.0))
    m_a = max(0.05, min(m_a, 5.0))
    start_points.append([m_h, m_a, -0.05])
    
    # 3. Drie willekeurige starts (deterministisch gezaaid voor consistentie)
    rng = random.Random(42)
    for _ in range(3):
        r_h = rng.uniform(0.05, 5.0)
        r_a = rng.uniform(0.05, 5.0)
        r_rho = rng.uniform(-0.25, 0.10)
        start_points.append([r_h, r_a, r_rho])

    best_loss = float('inf')
    best_params = None

    # Optimaliseer vanaf elk startpunt en bewaar de beste
    for i, start_pt in enumerate(start_points):
        try:
            res = minimize(objective, start_pt, method='L-BFGS-B', bounds=bounds,
                           options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 2000})
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except Exception as e:
            if verbose:
                print(f"    [Warning Fit] Startpunt {i} faalde: {e}")

    # Fallback naar beste startpunt als optimalisatie volledig faalt
    if best_params is None:
        best_init_loss = float('inf')
        for start_pt in start_points:
            l = objective(start_pt)
            if l < best_init_loss:
                best_init_loss = l
                best_params = start_pt
        if verbose:
            print("    [Warning Fit] Alle L-BFGS-B runs faalden. Fallback naar beste startpunt.")

    lam_h_opt, lam_a_opt, rho_opt = best_params
    
    if verbose:
        print(f"  [Debug Fit] Optimal parameters: lam_h={lam_h_opt:.4f}, lam_a={lam_a_opt:.4f}, rho={rho_opt:.4f} | Fit-residual ({loss_type}): {best_loss:.6f}")

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

def calc_ev_motd(pred_h, pred_a, matrix, scorer_rate=None, pred_scorer_h=None, pred_scorer_a=None):
    """
    Berekent de verwachte waarde (Expected Value, EV) in punten voor de Wedstrijd van de Dag (MOTD),
    waarbij extra punten voor doelpuntenmakers (spitsen) worden meegerekend.
    
    Parameters:
    pred_h (int): Het voorspelde aantal doelpunten van het thuisteam.
    pred_a (int): Het voorspelde aantal doelpunten van het uitteam.
    matrix (dict): De berekende kansenmatrix voor alle uitslagen.
    scorer_rate (float, optioneel): De scoringskans van de spits. Standaard SCORER_HIT_RATE.
    pred_scorer_h (bool, optioneel): Voorspelling doelpuntenmaker thuis.
    pred_scorer_a (bool, optioneel): Voorspelling doelpuntenmaker uit.
    
    Returns:
    tuple: (totale_ev, (thuis_scorer_tip, uit_scorer_tip), score_ev, scorer_ev).
    """
    if scorer_rate is None:
        scorer_rate = SCORER_HIT_RATE
        
    if pred_scorer_h is None:
        pred_scorer_h = (pred_h > 0)
    if pred_scorer_a is None:
        pred_scorer_a = (pred_a > 0)
        
    score_ev = 0.0
    scorer_ev = 0.0
    
    for (act_h, act_a), prob in matrix.items():
        # Wedstrijdresultaat punten
        score_pts = 0
        if pred_h == act_h and pred_a == act_a:
            score_pts += 12
        else:
            pred_toto = 1 if pred_h > pred_a else (-1 if pred_h < pred_a else 0)
            act_toto = 1 if act_h > act_a else (-1 if act_h < act_a else 0)
            if pred_toto == act_toto:
                if pred_toto == 0:
                    score_pts += 8
                else:
                    score_pts += 6
            
            if pred_h == act_h: score_pts += 2
            if pred_a == act_a: score_pts += 2
            
        # Doelpuntenmaker punten
        scorer_pts = 0
        if not pred_scorer_h:
            if act_h == 0: scorer_pts += 4
        else:
            if act_h > 0: scorer_pts += 4 * scorer_rate
            
        if not pred_scorer_a:
            if act_a == 0: scorer_pts += 4
        else:
            if act_a > 0: scorer_pts += 4 * scorer_rate
            
        score_ev += prob * score_pts
        scorer_ev += prob * scorer_pts
        
    total_ev = score_ev + scorer_ev
    return total_ev, (pred_scorer_h, pred_scorer_a), score_ev, scorer_ev


def calculate_actual_points(pred_h, pred_a, act_h, act_a, is_motd, scorer_rate=None, pred_scorer_h=None, pred_scorer_a=None):
    """
    Bereken de behaalde punten voor een voorspelling tegen de werkelijke uitslag.
    """
    matrix = {(act_h, act_a): 1.0}
    if is_motd:
        pts, _, _, _ = calc_ev_motd(pred_h, pred_a, matrix, scorer_rate=scorer_rate, pred_scorer_h=pred_scorer_h, pred_scorer_a=pred_scorer_a)
    else:
        pts = calc_ev_regular(pred_h, pred_a, matrix)
    return pts

def voorspel(home_pct, draw_pct, away_pct, is_motd, ou_probs=None, team_ou_home=None, team_ou_away=None, btts_prob=None, clean_sheet_home_prob=None, clean_sheet_away_prob=None, loss_type="logloss", overround_method="power", verbose=False, weight_match_ou=None, weight_team_ou=None, weight_extra_markets=None, tiebreak="probability", scorer_rate=None):
    """
    Berekent de optimale voorspelling door de uitslag te zoeken die de verwachte waarde (EV) maximaliseert.
    
    Parameters:
    home_pct (float): De kans op thuiswinst (percentage).
    draw_pct (float): De kans op gelijkspel (percentage).
    away_pct (float): De kans op uitwinst (percentage).
    is_motd (bool): Geeft aan of dit de Wedstrijd van de Dag (MOTD) is.
    ou_probs (dict, optioneel): Kansen voor over/under grenzen.
    team_ou_home (dict, optioneel): Kansen voor team-specifieke over/under grenzen (thuis).
    team_ou_away (dict, optioneel): Kansen voor team-specifieke over/under grenzen (uit).
    btts_prob (float, optioneel): De kans op beide teams scoren.
    clean_sheet_home_prob (float, optioneel): De kans op clean sheet thuis.
    clean_sheet_away_prob (float, optioneel): De kans op clean sheet uit.
    loss_type (str, optioneel): De te gebruiken verliesfunctie ('mse' of 'logloss'). Standaard 'logloss'.
    overround_method (str, optioneel): De te gebruiken normalisatiemethode ('linear' of 'power'). Standaard 'power'.
    verbose (bool, optioneel): Of er debug-logging getoond moet worden voor de fits. Standaard False.
    weight_match_ou (float, optioneel): Het gewicht voor de wedstrijd Over/Under fit-termen.
    weight_team_ou (float, optioneel): Het gewicht voor de team Over/Under fit-termen.
    weight_extra_markets (float, optioneel): Het gewicht voor de extra markten fit-termen.
    tiebreak (str, optioneel): De te gebruiken tie-breaker strategie ('probability' of 'conservative'). Standaard 'probability'.
    scorer_rate (float, optioneel): De scoringskans van de spits. Standaard SCORER_HIT_RATE.
    
    Returns:
    dict: Een woordenboek met alle resultaten, zoals genormaliseerde kansen, lambda's, rho,
          de geadviseerde uitslag, tips voor doelpuntenmakers en de maximale verwachte punten.
    """
    if scorer_rate is None:
        scorer_rate = SCORER_HIT_RATE
        
    p_h, p_d, p_a = normaliseer_kansen(home_pct, draw_pct, away_pct, method=overround_method)
    lam_h, lam_a, rho = bepaal_poisson_lambdas(p_h, p_d, p_a, ou_probs, target_team_ou_home=team_ou_home, target_team_ou_away=team_ou_away, target_btts=btts_prob, target_clean_sheet_home=clean_sheet_home_prob, target_clean_sheet_away=clean_sheet_away_prob, loss_type=loss_type, verbose=verbose, weight_match_ou=weight_match_ou, weight_team_ou=weight_team_ou, weight_extra_markets=weight_extra_markets)
    matrix = calc_matrix(lam_h, lam_a, rho)
    
    # Bepaal de dynamische EV-zoekruimte op basis van de berekende lambda's
    max_score = min(9, math.ceil(max(lam_h, lam_a) + 2))
    
    # Bereken de cumulatieve matrix-massa buiten het raster
    buiten_raster_massa = sum(prob for (act_h, act_a), prob in matrix.items() if act_h > max_score or act_a > max_score)
    
    # Als de cumulatieve massa buiten het raster groter is dan 1%, verhoog max_score met 1 (tot max 9)
    if buiten_raster_massa > 0.01 and max_score < 9:
        max_score += 1
        buiten_raster_massa = sum(prob for (act_h, act_a), prob in matrix.items() if act_h > max_score or act_a > max_score)
        
    if verbose:
        print(f"  [Verbose Raster] EV-zoekraster: 0-{max_score} voor beide teams | Buiten-raster-massa: {buiten_raster_massa*100:.2f}%")
        print(f"  [Verbose Tie-Break] Gekozen via tie-break: {tiebreak}")
        
    # Bepaal optimale scorer tips voor thuis en uit (alleen relevant voor MOTD)
    opt_sh = False
    opt_sa = False
    scorer_ev_h = 0.0
    scorer_ev_a = 0.0
    
    if is_motd:
        p_act_h_gt_0 = sum(prob for (act_h, act_a), prob in matrix.items() if act_h > 0)
        p_act_a_gt_0 = sum(prob for (act_h, act_a), prob in matrix.items() if act_a > 0)
        
        # Thuis
        ev_spits_h = 4 * scorer_rate * p_act_h_gt_0
        ev_geen_h = 4 * (1.0 - p_act_h_gt_0)
        if ev_spits_h >= ev_geen_h:
            opt_sh = True
            scorer_ev_h = ev_spits_h
        else:
            opt_sh = False
            scorer_ev_h = ev_geen_h
            
        # Uit
        ev_spits_a = 4 * scorer_rate * p_act_a_gt_0
        ev_geen_a = 4 * (1.0 - p_act_a_gt_0)
        if ev_spits_a >= ev_geen_a:
            opt_sa = True
            scorer_ev_a = ev_spits_a
        else:
            opt_sa = False
            scorer_ev_a = ev_geen_a
            
    alle_voorspellingen = []
    for h in range(max_score + 1):
        for a in range(max_score + 1):
            if is_motd:
                ev, scorers, score_ev, scorer_ev = calc_ev_motd(h, a, matrix, scorer_rate=scorer_rate, pred_scorer_h=opt_sh, pred_scorer_a=opt_sa)
            else:
                ev = calc_ev_regular(h, a, matrix)
                scorers = (False, False)
                score_ev = ev
                scorer_ev = 0.0
                
            prob = matrix.get((h, a), 0.0)
            
            # Bereken P(>=1 pt) en P(>=5 pt) cumulatief over alle matrix-uitkomsten
            p_1pt = 0.0
            p_5pt = 0.0
            for (act_h, act_a), act_prob in matrix.items():
                pts = calculate_actual_points(h, a, act_h, act_a, is_motd, scorer_rate=scorer_rate, pred_scorer_h=opt_sh, pred_scorer_a=opt_sa)
                if pts >= 1.0:
                    p_1pt += act_prob
                if pts >= 5.0:
                    p_5pt += act_prob
            
            alle_voorspellingen.append({
                "uitslag": f"{h}-{a}",
                "h": h,
                "a": a,
                "ev": ev,
                "score_ev": score_ev,
                "scorer_ev": scorer_ev,
                "kans": prob * 100.0,
                "p_exact": prob,
                "p_1pt": p_1pt,
                "p_5pt": p_5pt,
                "scorers": scorers
            })

            
    # Deterministische tie-break sortering (gebaseerd op poule-scoring):
    # a) Hoogste EV (Expected Value)
    # b) Hoogste P(exacte score) — kies de waarschijnlijkere score (exacte uitslag geeft meeste punten in de poule)
    # c) Hoogste P(≥5 punten) — meer kans op substantiële punten (TOTO-win/gelijk en correcte doelsaldi)
    # d) Laagste som van doelpunten (conservatiever, minder risico op doelpuntenspreiding)
    # e) Laagste aantal thuisdoelpunten (voor een definitieve, deterministische tie-break)
    if tiebreak == "conservative":
        alle_voorspellingen.sort(key=lambda x: (
            x["ev"],
            -(x["h"] + x["a"]),
            -x["h"],
            x["p_exact"],
            x["p_5pt"]
        ), reverse=True)
    else:  # probability
        alle_voorspellingen.sort(key=lambda x: (
            x["ev"],
            x["p_exact"],
            x["p_5pt"],
            -(x["h"] + x["a"]),
            -x["h"]
        ), reverse=True)
                
    # Bereken delta_ev ten opzichte van de 2e beste EV
    second_best_ev = alle_voorspellingen[1]["ev"] if len(alle_voorspellingen) > 1 else 0.0
    for item in alle_voorspellingen:
        item["delta_ev"] = item["ev"] - second_best_ev
        
    top_5 = alle_voorspellingen[:5]
    
    best_item = alle_voorspellingen[0]
    best_ev = best_item["ev"]
    best_pred = (best_item["h"], best_item["a"])
    best_scorers = best_item["scorers"]
    
    uitslag = f"{best_pred[0]}-{best_pred[1]}"
    
    if is_motd:
        scorer_thuis = "Spits (of penaltynemer)" if best_scorers[0] else "Geen score"
        scorer_uit = "Spits (of penaltynemer)" if best_scorers[1] else "Geen score"
        best_score_ev = best_item["score_ev"]
        best_scorer_ev = best_item["scorer_ev"]
        uitleg = (
            f"Maximale EV: {best_ev:.2f} verwachte punten (Score EV: {best_score_ev:.2f} pt, "
            f"Scorer EV: {best_scorer_ev:.2f} pt | Factor: {scorer_rate:.2f})."
        )
        # Bereken of de tip afwijkt van de simpele regel (h > 0 / a > 0)
        scorer_tip_onafhankelijk = (best_scorers[0] != (best_pred[0] > 0)) or (best_scorers[1] != (best_pred[1] > 0))
    else:
        scorer_thuis = ""
        scorer_uit = ""
        best_score_ev = best_ev
        best_scorer_ev = 0.0
        uitleg = f"Maximale EV: {best_ev:.2f} verwachte punten."
        scorer_tip_onafhankelijk = False

    # Toernooi tie-breaker-voorspellingen afgeleid uit het model (Prompt 13)
    tie_breakers = bereken_tie_breakers(lam_h, lam_a)

    return {
        "genormaliseerd": (p_h, p_d, p_a),
        "lambda": (lam_h, lam_a),
        "rho": rho,
        "uitslag": uitslag,
        "tie_breakers": tie_breakers,
        "scorer_thuis": scorer_thuis,
        "scorer_uit": scorer_uit,
        "uitleg": uitleg,
        "xpts": best_ev,
        "score_ev": best_score_ev,
        "scorer_ev": best_scorer_ev,
        "scorer_rate": scorer_rate,
        "team_ou_home": team_ou_home,
        "team_ou_away": team_ou_away,
        "top_5": top_5,
        "scorer_ev_thuis": scorer_ev_h if is_motd else 0.0,
        "scorer_ev_uit": scorer_ev_a if is_motd else 0.0,
        "scorer_tip_onafhankelijk": scorer_tip_onafhankelijk,
        "scorer_thuis_bool": best_scorers[0] if is_motd else False,
        "scorer_uit_bool": best_scorers[1] if is_motd else False
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
    
    t_home = res.get("team_ou_home")
    t_away = res.get("team_ou_away")
    if t_home or t_away:
        parts = []
        if t_home:
            for line, (u, o) in sorted(t_home.items()):
                parts.append(f"Thuis O{line}: {o*100:.0f}%")
        if t_away:
            for line, (u, o) in sorted(t_away.items()):
                parts.append(f"Uit O{line}: {o*100:.0f}%")
        if parts:
            print(f"  • Ploeg Totals: {' | '.join(parts)}")
    
    print(f"\n{GREEN}{BOLD}🏆  MAXIMALE EXPECTED VALUE (EV) ADVIES:{RESET}")
    print(f"  • {BOLD}Voorspelde uitslag:{RESET} {GREEN}{BOLD}{res['uitslag']}{RESET}")
    
    if is_motd:
        print(f"  • {BOLD}Doelpuntenmaker Thuis:{RESET} {YELLOW}{res['scorer_thuis']}{RESET}")
        print(f"  • {BOLD}Doelpuntenmaker Uit:{RESET} {YELLOW}{res['scorer_uit']}{RESET}")
        print(f"  • {BOLD}Verwachting apart:{RESET} Score EV: {YELLOW}{res.get('score_ev', 0.0):.2f}{RESET} pt | Scorer EV: {YELLOW}{res.get('scorer_ev', 0.0):.2f}{RESET} pt (Factor: {res.get('scorer_rate', SCORER_HIT_RATE):.2f})")
        if res.get("scorer_tip_onafhankelijk"):
            print(f"  • {CYAN}{BOLD}⚠️  Opmerking: Doelpuntenmaker-tip wijkt af van de uitslag-voorspelling!{RESET}")
            print(f"    {CYAN}Uitleg: Zelfs bij de voorspelde uitslag ({res['uitslag']}) heeft de gekozen tip wiskundig een hogere EV door de algemene doelkansen.{RESET}")

    
    print(f"\n{BOLD}💡  BEREKENING:{RESET}")
    print(f"  {res['uitleg']}")
    
    print(f"\n{BOLD}📊  TOP 5 VOORSPELLINGEN (RISICO-INZICHT):{RESET}")
    print(f"  {BOLD}{'Uitslag':<7} | {'EV':<5} | {'Kans%':<5} | {'P(>=5pt)':<8} | {'ΔEV':<6}{RESET}")
    print("  " + "-" * 43)
    for pred in res.get("top_5", []):
        u_val = pred["uitslag"]
        ev_val = f"{pred['ev']:.2f}"
        kans_val = f"{pred['kans']:.1f}%"
        p5_val = f"{pred['p_5pt']*100.0:.1f}%"
        
        delta_val = pred["delta_ev"]
        if delta_val == 0.0:
            delta_val_str = "0.00"
        else:
            delta_val_str = f"{delta_val:+.2f}"
            
        # Kleur delta_ev zonder de alignment te verstoren
        if delta_val > 0.0:
            delta_colored = f"{GREEN}{delta_val_str:<6}{RESET}"
        elif delta_val < 0.0:
            delta_colored = f"{RED}{delta_val_str:<6}{RESET}"
        else:
            delta_colored = f"{delta_val_str:<6}"
            
        print(f"  {u_val:<7} | {ev_val:<5} | {kans_val:<5} | {p5_val:<8} | {delta_colored}")
        
    if toon_extra:
        # Tie-breakers worden afgeleid uit het model (Prompt 13): ze variëren per
        # wedstrijd op basis van de berekende λ's i.p.v. vaste constanten.
        tb = res.get("tie_breakers") or bereken_tie_breakers(lam_h, lam_a)
        rode_minuut = tb.get("rode_kaart_minuut")
        rode_str = f"{rode_minuut}e minuut" if rode_minuut is not None else "Geen rode kaart (> 90e min)"
        print(f"\n{BOLD}⏱️  TIE-BREAKER EXTRA VRAGEN (modelgebaseerd):{RESET}")
        print(f"  • {BOLD}Minuut van het 1e toernooidoelpunt:{RESET} {YELLOW}{tb['eerste_doelpunt_minuut']}e minuut{RESET}")
        print(f"    {CYAN}{tb['eerste_doelpunt_uitleg']}{RESET}")
        print(f"  • {BOLD}Minuut van de 1e gele kaart:{RESET} {YELLOW}{tb['gele_kaart_minuut']}e minuut{RESET}")
        print(f"    {CYAN}{tb['gele_kaart_uitleg']}{RESET}")
        print(f"  • {BOLD}Minuut van de 1e rode kaart:{RESET} {YELLOW}{rode_str}{RESET}")
        print(f"    {CYAN}{tb['rode_kaart_uitleg']}{RESET}\n")

def interactieve_modus(toon_extra=False, loss_type="logloss", overround_method="power", verbose=False, weight_match_ou=None, weight_team_ou=None, tiebreak="probability", scorer_rate=None):
    """
    Start een interactief vraag-en-antwoordscherm in de terminal om een voorspelling voor één wedstrijd te berekenen.
    
    Parameters:
    toon_extra (bool, optioneel): Of er extra tie-breaker statistieken getoond moeten worden. Standaard False.
    loss_type (str, optioneel): De te gebruiken verliesfunctie ('mse' of 'logloss'). Standaard 'logloss'.
    overround_method (str, optioneel): De te gebruiken normalisatiemethode ('linear' of 'power'). Standaard 'power'.
    verbose (bool, optioneel): Of er debug-logging getoond moet worden voor de fits. Standaard False.
    weight_match_ou (float, optioneel): Het gewicht voor de wedstrijd Over/Under fit-termen.
    weight_team_ou (float, optioneel): Het gewicht voor de team Over/Under fit-termen.
    tiebreak (str, optioneel): De te gebruiken tie-breaker strategie. Standaard 'probability'.
    scorer_rate (float, optioneel): De scoringskans van de spits. Standaard SCORER_HIT_RATE.
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
            
    if is_motd:
        default_sr = scorer_rate if scorer_rate is not None else SCORER_HIT_RATE
        while True:
            sr_in = input(f"  Kans dat spits scoort bij teamdoelpunt [standaard: {default_sr}]: ").strip()
            if not sr_in:
                scorer_rate = default_sr
                break
            try:
                scorer_rate = float(sr_in)
                if not (0.0 <= scorer_rate <= 1.0):
                    raise ValueError("Kans moet tussen 0.0 en 1.0 liggen")
                break
            except ValueError as e:
                print(f"  {RED}❌ {e}. Probeer het opnieuw.{RESET}")

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
            
    res = voorspel(home, draw, away, is_motd, loss_type=loss_type, overround_method=overround_method, verbose=verbose, weight_match_ou=weight_match_ou, weight_team_ou=weight_team_ou, tiebreak=tiebreak, scorer_rate=scorer_rate)
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

def selecteer_en_normaliseer_lijn(raw_data, est_total, overround_method, type_label, home_team, away_team, verbose=False):
    """
    Selecteert maximaal 1 lijn uit raw_data op basis van:
      a) Beide kanten (under + over) beschikbaar
      b) Hoogste gecombineerde liquiditeit
      c) Dichtst bij verwacht aantal doelpunten (est_total)
    Normaliseert de geselecteerde lijn en geeft een dict terug: {lijn: (u_norm, o_norm, avg_spread)}
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
        
    # Sorteer op:
    # 1. has_both (True eerst -> -1, False -> 0)
    # 2. combined_liq (dalend -> -combined_liq)
    # 3. diff (stijgend -> diff)
    candidates.sort(key=lambda x: (-int(x[3]), -x[4], x[5]))
    
    best = candidates[0]
    best_line = best[0]
    best_u = best[1]
    best_o = best[2]
    best_has_both = best[3]
    best_liq = best[4]
    best_diff = best[5]
    best_info = best[6]
    
    # Log geselecteerde en genegeerde lijnen
    if verbose:
        print(f"  [Verbose O/U] Selectie voor '{type_label}' ({home_team} vs. {away_team}):")
        print(f"    -> Geselecteerd: Lijn {best_line} (beide kanten: {best_has_both}, liquiditeit: {best_liq:.2f}, diff: {best_diff:.4f})")
        for c in candidates[1:]:
            print(f"       Genegeerd: Lijn {c[0]} (beide kanten: {c[3]}, liquiditeit: {c[4]:.2f}, diff: {c[5]:.4f})")
            
    # Normaliseren
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
        
    # Bereken gemiddelde spread
    spreads = []
    if best_info['under_spread'] is not None:
        spreads.append(best_info['under_spread'])
    if best_info['over_spread'] is not None:
        spreads.append(best_info['over_spread'])
    avg_spread = sum(spreads) / len(spreads) if spreads else None
    
    return {best_line: (u_norm, o_norm, avg_spread)}

def haal_polymarket_wedstrijden(overround_method="power", verbose=False):
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
                
                # Probeer eerst team-specifiek O/U te herkennen
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
                    team_words = set(re.findall(r'\w+', team_name.lower()))
                    home_words = set(re.findall(r'\w+', home_team.lower()))
                    away_words = set(re.findall(r'\w+', away_team.lower()))
                    
                    sc_h = len(team_words.intersection(home_words))
                    sc_a = len(team_words.intersection(away_words))
                    
                    if (sc_h > 0) != (sc_a > 0):
                        if sc_h > 0:
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
                        else:
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
                        home_words = set(re.findall(r'\w+', home_team.lower()))
                        away_words = set(re.findall(r'\w+', away_team.lower()))
                        q_words = set(re.findall(r'\w+', q))
                        sc_h = len(q_words.intersection(home_words))
                        sc_a = len(q_words.intersection(away_words))
                        if sc_h > sc_a:
                            cs_home_markets.append((yes_price, no_price, liq, spread, m.get("question", "")))
                        elif sc_a > sc_h:
                            cs_away_markets.append((yes_price, no_price, liq, spread, m.get("question", "")))
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
                p_h = home_prob
                p_d = draw_prob
                p_a = away_prob
                
                # Maher-like lambda approximation
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
                
                # Selecteer en normaliseer extra markten
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
                    "is_motd": is_motd_match(home_team, away_team)
                })
                
        parsed_matches.sort(key=lambda x: x["date"])
        return parsed_matches, None
    except Exception as err:
        return None, f"Fout bij verbinding met Polymarket: {err}"

def exporteer_naar_bestand(alle_res, bestandsnaam, inclusief_top5=False):
    """
    Exporteert alle berekende uitslagen chronologisch naar een tekstbestand met xG, rho en MOTD scorer tips.
    
    Parameters:
    alle_res (list): Een lijst met paren van (wedstrijd_data, voorspelling_resultaat).
    bestandsnaam (str): Het pad naar het uit te voeren tekstbestand.
    inclusief_top5 (bool, optioneel): Of de top-5 risico-analyse per wedstrijd getoond moet worden.
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
                
                has_team_ou = bool(res.get("team_ou_home") or res.get("team_ou_away"))
                xg_str = f"{lam_h:.2f} - {lam_a:.2f}"
                if has_team_ou:
                    xg_str += " ✓"
                
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
            
            if inclusief_top5:
                f.write("\n" + "=" * 154 + "\n")
                f.write("                                            RISICO-INZICHT: TOP 5 ALTERNATIEVE UITSLAGEN PER WEDSTRIJD\n")
                f.write("=" * 154 + "\n\n")
                for m, res in alle_res:
                    f.write(f"Wedstrijd: {m['home']} vs. {m['away']} (MOTD: {'Ja' if m['is_motd'] else 'Nee'})\n")
                    f.write(f"  {'Uitslag':<8} | {'EV (pts)':<8} | {'Kans%':<6} | {'P(>=1pt)':<8} | {'P(>=5pt)':<8} | {'ΔEV':<6}\n")
                    f.write("  " + "-" * 57 + "\n")
                    for pred in res.get("top_5", []):
                        u_val = pred["uitslag"]
                        ev_val = f"{pred['ev']:.2f}"
                        kans_val = f"{pred['kans']:.1f}%"
                        p1_val = f"{pred['p_1pt']*100:.1f}%"
                        p5_val = f"{pred['p_5pt']*100:.1f}%"
                        delta_val = pred["delta_ev"]
                        delta_val_str = f"{delta_val:+.2f}" if delta_val != 0.0 else "0.00"
                        f.write(f"  {u_val:<8} | {ev_val:<8} | {kans_val:<6} | {p1_val:<8} | {p5_val:<8} | {delta_val_str:<6}\n")
                    f.write("\n")
            
            f.write("\n=================================================================================================================================================\n")
            f.write("Gegenereerd door de Voetbalpoules Polymarket Voorspeller CLI.\n")
            
        print(f"{GREEN}✓ Voorspellingen succesvol opgeslagen in {BOLD}{bestandsnaam}{RESET}!\n")
    except Exception as e:
        print(f"{RED}❌ Fout bij opslaan van bestand: {e}{RESET}\n")

def _calculator_core_js():
    """
    Genereert de gedeelde JavaScript 'calculator core' die de Python-functie voorspel()
    1-op-1 spiegelt (log-loss objective, power-overround, multi-start optimalisatie,
    dynamisch EV-raster, tie-break logica, MOTD-scorer-optimalisatie en model-tie-breakers).

    Dit is de single source of truth voor de browser-wiskunde: dezelfde string wordt zowel
    in index.html ingebed als door test_js_parity.py via Node uitgevoerd. Zo blijft de
    JS-rekenmachine identiek aan de CLI (zie acceptatiecriteria Prompt 14).

    De module-constanten worden hier ingevuld zodat de gewichten nooit kunnen driften.
    """
    js = r'''
        // =====================================================================
        // CALCULATOR CORE — automatisch gegenereerd vanuit polymarket_voorspeller.py
        // (functie _calculator_core_js). NIET handmatig aanpassen: bewerk de Python-bron.
        // Spiegelt voorspel() zodat de browser identieke adviezen geeft als de CLI.
        // =====================================================================
        const WEIGHT_MATCH_OU = __WEIGHT_MATCH_OU__;
        const WEIGHT_TEAM_OU = __WEIGHT_TEAM_OU__;
        const WEIGHT_EXTRA_MARKETS = __WEIGHT_EXTRA_MARKETS__;
        const SCORER_HIT_RATE = __SCORER_HIT_RATE__;
        const FIRST_YELLOW_BASE_MIN = __FIRST_YELLOW_BASE_MIN__;
        const FIRST_YELLOW_REF_LAMBDA = __FIRST_YELLOW_REF_LAMBDA__;
        const RED_CARD_RATE = __RED_CARD_RATE__;
        const RED_CARD_THRESHOLD = __RED_CARD_THRESHOLD__;
        const RHO_REG = __RHO_REG__;

        function factorial(n) { let r = 1; for (let i = 2; i <= n; i++) r *= i; return r; }
        function poisson(lam, k) { return Math.exp(-lam) * Math.pow(lam, k) / factorial(k); }
        function dixonColesTau(h, a, lh, la, rho) {
            if (h === 0 && a === 0) return 1.0 - lh * la * rho;
            if (h === 1 && a === 0) return 1.0 + la * rho;
            if (h === 0 && a === 1) return 1.0 + lh * rho;
            if (h === 1 && a === 1) return 1.0 - rho;
            return 1.0;
        }
        function calcMatrix(lh, la, rho) {
            if (rho === undefined) rho = 0.0;
            let m = [], tot = 0.0;
            for (let h = 0; h < 10; h++) {
                m[h] = [];
                for (let a = 0; a < 10; a++) {
                    let p = poisson(lh, h) * poisson(la, a) * dixonColesTau(h, a, lh, la, rho);
                    p = Math.max(0.0, p); m[h][a] = p; tot += p;
                }
            }
            if (tot > 0.0) { for (let h = 0; h < 10; h++) for (let a = 0; a < 10; a++) m[h][a] /= tot; }
            return m;
        }
        function get1X2(m) {
            let hw = 0.0, d = 0.0, aw = 0.0;
            for (let h = 0; h < 10; h++) for (let a = 0; a < 10; a++) {
                let p = m[h][a];
                if (h > a) hw += p; else if (h === a) d += p; else aw += p;
            }
            return [hw, d, aw];
        }
        // Power-method overround (spiegelt normaliseer_kansen_power)
        function normPower(kansen, targetSum) {
            if (targetSum === undefined) targetSum = 1.0;
            kansen = kansen.map(p => Math.max(0.0001, Math.min(0.9999, p)));
            let som = kansen.reduce((s, p) => s + p, 0);
            if (som === 0) return kansen.map(() => 1.0 / kansen.length);
            if (kansen.length === 1) return [targetSum];
            if (Math.abs(som - targetSum) < 1e-9) return kansen.map(p => p * (targetSum / som));
            let kLow, kHigh;
            if (som > targetSum) {
                kLow = 1.0; kHigh = 2.0;
                for (let i = 0; i < 50; i++) { let s = kansen.reduce((acc, p) => acc + Math.pow(p, kHigh), 0); if (s < targetSum) break; kHigh *= 2.0; }
            } else {
                kLow = 0.001; kHigh = 1.0;
                for (let i = 0; i < 50; i++) { let s = kansen.reduce((acc, p) => acc + Math.pow(p, kLow), 0); if (s > targetSum) break; kLow /= 2.0; }
            }
            let k = (kLow + kHigh) / 2.0, converged = false;
            for (let i = 0; i < 100; i++) {
                let kMid = (kLow + kHigh) / 2.0;
                let s = kansen.reduce((acc, p) => acc + Math.pow(p, kMid), 0);
                if (Math.abs(s - targetSum) < 1e-12) { k = kMid; converged = true; break; }
                if (s > targetSum) kLow = kMid; else kHigh = kMid;
            }
            if (!converged) k = (kLow + kHigh) / 2.0;
            let result = kansen.map(p => Math.pow(p, k));
            let sRes = result.reduce((s, r) => s + r, 0);
            if (sRes > 0) result = result.map(r => r * (targetSum / sRes));
            return result;
        }
        function normaliseer1X2(home, draw, away, method) {
            let isPct = (home > 1.0 || draw > 1.0 || away > 1.0 || (home + draw + away) > 1.5);
            let h = isPct ? home / 100.0 : home, d = isPct ? draw / 100.0 : draw, a = isPct ? away / 100.0 : away;
            if (method === 'power') { let n = normPower([h, d, a], 1.0); return [n[0], n[1], n[2]]; }
            let tot = h + d + a; if (tot === 0) return [0, 0, 0]; return [h / tot, d / tot, a / tot];
        }
        // Over/Under fit-term (spiegelt de O/U-lussen in bepaal_poisson_lambdas)
        function ouError(matrix, lines, kind, lossType, weight, eps) {
            if (!lines) return 0.0;
            let keys = Object.keys(lines); if (keys.length === 0) return 0.0;
            let useRel = keys.length > 1, err = 0.0;
            for (let key of keys) {
                let line = parseFloat(key); let vals = lines[key]; let tU = vals[0], tO = vals[1];
                let u = 0.0, o = 0.0;
                for (let sh = 0; sh < 10; sh++) for (let sa = 0; sa < 10; sa++) {
                    let p = matrix[sh][sa];
                    let metric = (kind === 'match') ? (sh + sa) : (kind === 'home' ? sh : sa);
                    if (metric < line) u += p;
                    if (metric > line) o += p;
                }
                let wf = 1.0;
                if (useRel) {
                    if (vals.length >= 3 && vals[2] != null) wf = 1.0 / Math.max(vals[2], 1e-4);
                    else { let tot = tU + tO; if (tot > 0) { let pu = tU / tot, po = tO / tot; let vol = Math.sqrt(Math.max(pu * po, 1e-6)); wf = 0.5 / vol; } }
                }
                if (lossType === 'logloss') err += -(tU * Math.log(Math.max(u, eps)) + tO * Math.log(Math.max(o, eps))) * weight * wf;
                else err += (Math.pow(u - tU, 2) + Math.pow(o - tO, 2)) * weight * wf;
            }
            return err;
        }
        function objective(params, t) {
            let lh = Math.max(0.05, Math.min(params[0], 5.0));
            let la = Math.max(0.05, Math.min(params[1], 5.0));
            let r = Math.max(-0.25, Math.min(params[2], 0.10));
            let matrix = calcMatrix(lh, la, r);
            let g = get1X2(matrix); let h = g[0], d = g[1], a = g[2];
            let eps = 1e-15, error;
            if (t.loss_type === 'logloss') error = -(t.target_h * Math.log(Math.max(h, eps)) + t.target_d * Math.log(Math.max(d, eps)) + t.target_a * Math.log(Math.max(a, eps)));
            else error = Math.pow(h - t.target_h, 2) + Math.pow(d - t.target_d, 2) + Math.pow(a - t.target_a, 2);
            error += ouError(matrix, t.ou, 'match', t.loss_type, t.weight_match_ou, eps);
            error += ouError(matrix, t.team_ou_home, 'home', t.loss_type, t.weight_team_ou, eps);
            error += ouError(matrix, t.team_ou_away, 'away', t.loss_type, t.weight_team_ou, eps);
            if (t.btts != null) {
                let p = 0.0; for (let sh = 0; sh < 10; sh++) for (let sa = 0; sa < 10; sa++) { if (sh > 0 && sa > 0) p += matrix[sh][sa]; }
                if (t.loss_type === 'logloss') error += -(t.btts * Math.log(Math.max(p, eps)) + (1.0 - t.btts) * Math.log(Math.max(1.0 - p, eps))) * t.weight_extra_markets;
                else error += Math.pow(p - t.btts, 2) * t.weight_extra_markets;
            }
            if (t.cs_home != null) {
                let p = 0.0; for (let sh = 0; sh < 10; sh++) for (let sa = 0; sa < 10; sa++) { if (sa === 0) p += matrix[sh][sa]; }
                if (t.loss_type === 'logloss') error += -(t.cs_home * Math.log(Math.max(p, eps)) + (1.0 - t.cs_home) * Math.log(Math.max(1.0 - p, eps))) * t.weight_extra_markets;
                else error += Math.pow(p - t.cs_home, 2) * t.weight_extra_markets;
            }
            if (t.cs_away != null) {
                let p = 0.0; for (let sh = 0; sh < 10; sh++) for (let sa = 0; sa < 10; sa++) { if (sh === 0) p += matrix[sh][sa]; }
                if (t.loss_type === 'logloss') error += -(t.cs_away * Math.log(Math.max(p, eps)) + (1.0 - t.cs_away) * Math.log(Math.max(1.0 - p, eps))) * t.weight_extra_markets;
                else error += Math.pow(p - t.cs_away, 2) * t.weight_extra_markets;
            }
            // Lichte ρ-regularisatie: maakt het minimum uniek bij onderbepaalde (1X2-only) fits.
            error += RHO_REG * (r * r);
            return error;
        }
        function nelderMead(start, evalFunc) {
            const clip = (p) => [Math.max(0.05, Math.min(p[0], 5.0)), Math.max(0.05, Math.min(p[1], 5.0)), Math.max(-0.25, Math.min(p[2], 0.10))];
            let simplex = [
                [start[0], start[1], start[2]],
                [start[0] + 0.05 * Math.abs(start[0]) + 1e-4, start[1], start[2]],
                [start[0], start[1] + 0.05 * Math.abs(start[1]) + 1e-4, start[2]],
                [start[0], start[1], start[2] + 0.05]
            ];
            for (let i = 0; i < 4; i++) simplex[i] = clip(simplex[i]);
            let values = simplex.map(p => evalFunc(p));
            const maxIter = 2000, tol = 1e-16, alpha = 1.0, gamma = 2.0, beta = 0.5, sigma = 0.5;
            for (let iter = 0; iter < maxIter; iter++) {
                let idx = [0, 1, 2, 3]; idx.sort((x, y) => values[x] - values[y]);
                simplex = idx.map(i => simplex[i]); values = idx.map(i => values[i]);
                if (Math.abs(values[3] - values[0]) < tol) break;
                let c = [0, 0, 0]; for (let j = 0; j < 3; j++) { c[0] += simplex[j][0]; c[1] += simplex[j][1]; c[2] += simplex[j][2]; } c = c.map(v => v / 3);
                let refl = clip([c[0] + alpha * (c[0] - simplex[3][0]), c[1] + alpha * (c[1] - simplex[3][1]), c[2] + alpha * (c[2] - simplex[3][2])]);
                let fR = evalFunc(refl);
                if (fR < values[1] && fR >= values[0]) { simplex[3] = refl; values[3] = fR; continue; }
                if (fR < values[0]) {
                    let ex = clip([c[0] + gamma * (refl[0] - c[0]), c[1] + gamma * (refl[1] - c[1]), c[2] + gamma * (refl[2] - c[2])]);
                    let fE = evalFunc(ex);
                    if (fE < fR) { simplex[3] = ex; values[3] = fE; } else { simplex[3] = refl; values[3] = fR; }
                    continue;
                }
                let contracted;
                if (fR < values[3]) {
                    contracted = clip([c[0] + beta * (refl[0] - c[0]), c[1] + beta * (refl[1] - c[1]), c[2] + beta * (refl[2] - c[2])]);
                    let fC = evalFunc(contracted);
                    if (fC <= fR) { simplex[3] = contracted; values[3] = fC; continue; }
                } else {
                    contracted = clip([c[0] + beta * (simplex[3][0] - c[0]), c[1] + beta * (simplex[3][1] - c[1]), c[2] + beta * (simplex[3][2] - c[2])]);
                    let fC = evalFunc(contracted);
                    if (fC < values[3]) { simplex[3] = contracted; values[3] = fC; continue; }
                }
                for (let i = 1; i < 4; i++) {
                    simplex[i] = clip([simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0]), simplex[0][1] + sigma * (simplex[i][1] - simplex[0][1]), simplex[0][2] + sigma * (simplex[i][2] - simplex[0][2])]);
                    values[i] = evalFunc(simplex[i]);
                }
            }
            let idx = [0, 1, 2, 3]; idx.sort((x, y) => values[x] - values[y]);
            return clip(simplex[idx[0]]);
        }
        // Multi-start optimalisatie. De starts (standaard + Maher + deterministische spreiding)
        // convergeren alle naar het globale optimum, dus de keuze met laagste loss is
        // identiek aan het L-BFGS-B-resultaat van de Python-implementatie.
        function optimize(t) {
            let starts = [[1.3, 1.0, -0.05]];
            let mh = -Math.log(Math.max(0.01, Math.min(0.99, t.target_d + t.target_a)));
            let ma = -Math.log(Math.max(0.01, Math.min(0.99, t.target_h + t.target_d)));
            mh = Math.max(0.05, Math.min(mh, 5.0)); ma = Math.max(0.05, Math.min(ma, 5.0));
            starts.push([mh, ma, -0.05]);
            starts.push([0.5, 0.5, -0.05]); starts.push([2.5, 1.5, -0.10]); starts.push([1.0, 2.5, 0.00]);
            starts.push([3.5, 0.8, -0.15]); starts.push([0.3, 3.0, 0.05]);
            let evalFunc = (p) => objective(p, t);
            let best = null, bestLoss = Infinity;
            for (let s of starts) { let res = nelderMead(s, evalFunc); let l = evalFunc(res); if (l < bestLoss) { bestLoss = l; best = res; } }
            return best;
        }
        function calcEvRegular(ph, pa, m) {
            let ev = 0.0;
            for (let ah = 0; ah < 10; ah++) for (let aa = 0; aa < 10; aa++) {
                let prob = m[ah][aa], pts = 0;
                if (ph === ah && pa === aa) pts += 10;
                else { let pt = ph > pa ? 1 : (ph < pa ? -1 : 0); let at = ah > aa ? 1 : (ah < aa ? -1 : 0); if (pt === at) pts += (pt === 0) ? 7 : 5; if (ph === ah) pts += 2; if (pa === aa) pts += 2; }
                ev += prob * pts;
            }
            return ev;
        }
        function calcEvMotd(ph, pa, m, scorerRate, psh, psa) {
            if (scorerRate === undefined) scorerRate = SCORER_HIT_RATE;
            if (psh === undefined || psh === null) psh = (ph > 0);
            if (psa === undefined || psa === null) psa = (pa > 0);
            let scoreEv = 0.0, scorerEv = 0.0;
            for (let ah = 0; ah < 10; ah++) for (let aa = 0; aa < 10; aa++) {
                let prob = m[ah][aa], sp = 0;
                if (ph === ah && pa === aa) sp += 12;
                else { let pt = ph > pa ? 1 : (ph < pa ? -1 : 0); let at = ah > aa ? 1 : (ah < aa ? -1 : 0); if (pt === at) sp += (pt === 0) ? 8 : 6; if (ph === ah) sp += 2; if (pa === aa) sp += 2; }
                let scp = 0;
                if (!psh) { if (ah === 0) scp += 4; } else { if (ah > 0) scp += 4 * scorerRate; }
                if (!psa) { if (aa === 0) scp += 4; } else { if (aa > 0) scp += 4 * scorerRate; }
                scoreEv += prob * sp; scorerEv += prob * scp;
            }
            return [scoreEv + scorerEv, [psh, psa], scoreEv, scorerEv];
        }
        function calcActualPoints(ph, pa, ah, aa, isMotd, scorerRate, psh, psa) {
            let pts = 0;
            if (ph === ah && pa === aa) pts += isMotd ? 12 : 10;
            else {
                let pt = ph > pa ? 1 : (ph < pa ? -1 : 0); let at = ah > aa ? 1 : (ah < aa ? -1 : 0);
                if (pt === at) pts += isMotd ? ((pt === 0) ? 8 : 6) : ((pt === 0) ? 7 : 5);
                if (ph === ah) pts += 2; if (pa === aa) pts += 2;
            }
            if (isMotd) {
                if (!psh) { if (ah === 0) pts += 4; } else { if (ah > 0) pts += 4 * scorerRate; }
                if (!psa) { if (aa === 0) pts += 4; } else { if (aa > 0) pts += 4 * scorerRate; }
            }
            return pts;
        }
        // Modelgebaseerde tie-breakers (spiegelt bereken_tie_breakers)
        function berekenTieBreakers(lh, la) {
            let lamTotal = Math.max(lh + la, 1e-9);
            let eersteDoelpunt = Math.round(90.0 * (1.0 - Math.log(2.0)) / lamTotal);
            eersteDoelpunt = Math.max(1, Math.min(90, eersteDoelpunt));
            let geleKaart = Math.round(FIRST_YELLOW_BASE_MIN * (FIRST_YELLOW_REF_LAMBDA / lamTotal));
            geleKaart = Math.max(1, Math.min(90, geleKaart));
            let effRed = RED_CARD_RATE * (lamTotal / FIRST_YELLOW_REF_LAMBDA);
            let pRed = 1.0 - Math.exp(-effRed);
            let rodeKaart = null;
            if (pRed >= RED_CARD_THRESHOLD) rodeKaart = Math.max(1, Math.min(90, Math.round(90.0 * (1.0 - Math.log(2.0)) / effRed)));
            return { eerste_doelpunt_minuut: eersteDoelpunt, gele_kaart_minuut: geleKaart, rode_kaart_minuut: rodeKaart, rode_kaart_kans: pRed };
        }
        function massaBuiten(m, maxScore) {
            let s = 0.0; for (let h = 0; h < 10; h++) for (let a = 0; a < 10; a++) { if (h > maxScore || a > maxScore) s += m[h][a]; } return s;
        }
        function cmpDesc(a, b, keyFn) { let ka = keyFn(a), kb = keyFn(b); for (let i = 0; i < ka.length; i++) { if (ka[i] !== kb[i]) return kb[i] - ka[i]; } return 0; }
        // Hoofdfunctie — spiegelt voorspel(). Input/Output-keys komen overeen met de Python-dict.
        function predictMatch(inp) {
            let lossType = inp.loss_type || 'logloss';
            let overround = inp.overround_method || 'power';
            let scorerRate = (inp.scorer_rate != null) ? inp.scorer_rate : SCORER_HIT_RATE;
            let wMatch = (inp.weight_match_ou != null) ? inp.weight_match_ou : WEIGHT_MATCH_OU;
            let wTeam = (inp.weight_team_ou != null) ? inp.weight_team_ou : WEIGHT_TEAM_OU;
            let wExtra = (inp.weight_extra_markets != null) ? inp.weight_extra_markets : WEIGHT_EXTRA_MARKETS;
            let tiebreak = inp.tiebreak || 'probability';
            let isMotd = !!inp.is_motd;
            let g = normaliseer1X2(inp.home_pct, inp.draw_pct, inp.away_pct, overround);
            let pH = g[0], pD = g[1], pA = g[2];
            let t = {
                target_h: pH, target_d: pD, target_a: pA,
                ou: inp.ou_probs || null, team_ou_home: inp.team_ou_home || null, team_ou_away: inp.team_ou_away || null,
                btts: (inp.btts_prob != null ? inp.btts_prob : null),
                cs_home: (inp.clean_sheet_home_prob != null ? inp.clean_sheet_home_prob : null),
                cs_away: (inp.clean_sheet_away_prob != null ? inp.clean_sheet_away_prob : null),
                loss_type: lossType, weight_match_ou: wMatch, weight_team_ou: wTeam, weight_extra_markets: wExtra
            };
            let opt = optimize(t); let lamH = opt[0], lamA = opt[1], rho = opt[2];
            let matrix = calcMatrix(lamH, lamA, rho);
            let maxScore = Math.min(9, Math.ceil(Math.max(lamH, lamA) + 2));
            let buiten = massaBuiten(matrix, maxScore);
            if (buiten > 0.01 && maxScore < 9) { maxScore += 1; buiten = massaBuiten(matrix, maxScore); }
            let optSh = false, optSa = false, scorerEvH = 0.0, scorerEvA = 0.0;
            if (isMotd) {
                let pHgt = 0.0, pAgt = 0.0;
                for (let ah = 0; ah < 10; ah++) for (let aa = 0; aa < 10; aa++) { let p = matrix[ah][aa]; if (ah > 0) pHgt += p; if (aa > 0) pAgt += p; }
                let evSpitsH = 4 * scorerRate * pHgt, evGeenH = 4 * (1.0 - pHgt);
                if (evSpitsH >= evGeenH) { optSh = true; scorerEvH = evSpitsH; } else { optSh = false; scorerEvH = evGeenH; }
                let evSpitsA = 4 * scorerRate * pAgt, evGeenA = 4 * (1.0 - pAgt);
                if (evSpitsA >= evGeenA) { optSa = true; scorerEvA = evSpitsA; } else { optSa = false; scorerEvA = evGeenA; }
            }
            let preds = [];
            for (let h = 0; h <= maxScore; h++) for (let a = 0; a <= maxScore; a++) {
                let ev, scoreEv, scorerEv, scorers;
                if (isMotd) { let r = calcEvMotd(h, a, matrix, scorerRate, optSh, optSa); ev = r[0]; scorers = r[1]; scoreEv = r[2]; scorerEv = r[3]; }
                else { ev = calcEvRegular(h, a, matrix); scorers = [false, false]; scoreEv = ev; scorerEv = 0.0; }
                let prob = matrix[h][a] || 0.0;
                let p1 = 0.0, p5 = 0.0;
                for (let ah = 0; ah < 10; ah++) for (let aa = 0; aa < 10; aa++) {
                    let ap = matrix[ah][aa];
                    let pts = calcActualPoints(h, a, ah, aa, isMotd, scorerRate, optSh, optSa);
                    if (pts >= 1.0) p1 += ap; if (pts >= 5.0) p5 += ap;
                }
                preds.push({ uitslag: h + '-' + a, h: h, a: a, ev: ev, score_ev: scoreEv, scorer_ev: scorerEv, kans: prob * 100.0, p_exact: prob, p_1pt: p1, p_5pt: p5, scorers: scorers });
            }
            let keyFn;
            if (tiebreak === 'conservative') keyFn = (x) => [x.ev, -(x.h + x.a), -x.h, x.p_exact, x.p_5pt];
            else keyFn = (x) => [x.ev, x.p_exact, x.p_5pt, -(x.h + x.a), -x.h];
            preds.sort((a, b) => cmpDesc(a, b, keyFn));
            let secondBest = preds.length > 1 ? preds[1].ev : 0.0;
            preds.forEach(p => { p.delta_ev = p.ev - secondBest; });
            let top5 = preds.slice(0, 5);
            let best = preds[0];
            let result = {
                genormaliseerd: [pH, pD, pA], lambda: [lamH, lamA], rho: rho, uitslag: best.h + '-' + best.a,
                xpts: best.ev, score_ev: best.score_ev, scorer_ev: best.scorer_ev, scorer_rate: scorerRate,
                top_5: top5, tie_breakers: berekenTieBreakers(lamH, lamA), is_motd: isMotd,
                scorer_ev_thuis: isMotd ? scorerEvH : 0.0, scorer_ev_uit: isMotd ? scorerEvA : 0.0,
                scorer_thuis_bool: isMotd ? best.scorers[0] : false, scorer_uit_bool: isMotd ? best.scorers[1] : false
            };
            result.scorer_tip_onafhankelijk = isMotd ? ((best.scorers[0] !== (best.h > 0)) || (best.scorers[1] !== (best.a > 0))) : false;
            return result;
        }
        if (typeof module !== 'undefined' && module.exports) {
            module.exports = { predictMatch, calcMatrix, normPower, normaliseer1X2, berekenTieBreakers, optimize, get1X2 };
        }
'''
    replacements = {
        "__WEIGHT_MATCH_OU__": repr(float(WEIGHT_MATCH_OU)),
        "__WEIGHT_TEAM_OU__": repr(float(WEIGHT_TEAM_OU)),
        "__WEIGHT_EXTRA_MARKETS__": repr(float(WEIGHT_EXTRA_MARKETS)),
        "__SCORER_HIT_RATE__": repr(float(SCORER_HIT_RATE)),
        "__FIRST_YELLOW_BASE_MIN__": repr(float(FIRST_YELLOW_BASE_MIN)),
        "__FIRST_YELLOW_REF_LAMBDA__": repr(float(FIRST_YELLOW_REF_LAMBDA)),
        "__RED_CARD_RATE__": repr(float(RED_CARD_RATE)),
        "__RED_CARD_THRESHOLD__": repr(float(RED_CARD_THRESHOLD)),
        "__RHO_REG__": repr(float(RHO_REG)),
    }
    for token, value in replacements.items():
        js = js.replace(token, value)
    return js


def exporteer_naar_html(alle_res, bestandsnaam, scorer_rate=None):
    """
    Genereert een prachtige, mobielvriendelijke HTML-pagina (index.html) met de voorspellingen.
    
    Parameters:
    alle_res (list): Een lijst met paren van (wedstrijd_data, voorspelling_resultaat).
    bestandsnaam (str): Het pad naar het te genereren HTML-bestand.
    scorer_rate (float, optioneel): De scoringskans van de spits. Standaard SCORER_HIT_RATE.
    """
    if scorer_rate is None:
        scorer_rate = SCORER_HIT_RATE
        
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

        /* Rekenmodule Calculator Styles */
        .calculator-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(6, 182, 212, 0.2);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 8px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(6, 182, 212, 0.05);
        }}
        
        .calculator-card:hover {{
            border-color: rgba(6, 182, 212, 0.4);
            box-shadow: 0 8px 30px rgba(6, 182, 212, 0.12);
        }}
        
        .calculator-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
        }}
        
        .calc-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
            padding-bottom: 4px;
        }}
        
        .calc-header h2 {{
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #06b6d4, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .calc-toggle-icon {{
            font-size: 1.1rem;
            color: var(--accent-blue);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .calculator-card.collapsed .calc-toggle-icon {{
            transform: rotate(-90deg);
        }}
        
        .calc-body {{
            margin-top: 18px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            max-height: 1200px;
            transition: max-height 0.4s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s, margin-top 0.4s;
            opacity: 1;
        }}
        
        .calculator-card.collapsed .calc-body {{
            max-height: 0;
            margin-top: 0;
            opacity: 0;
            overflow: hidden;
            pointer-events: none;
        }}
        
        .calc-section-title {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .form-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }}
        
        .form-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        
        .form-group label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        
        .input-wrapper {{
            position: relative;
            display: flex;
            align-items: center;
        }}
        
        .input-wrapper input, .input-wrapper select {{
            width: 100%;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 10px 12px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        
        .input-wrapper input:focus, .input-wrapper select:focus {{
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.15);
            background: rgba(15, 23, 42, 0.8);
        }}
        
        .input-wrapper .input-suffix {{
            position: absolute;
            right: 12px;
            color: var(--text-secondary);
            font-size: 0.8rem;
            pointer-events: none;
            font-weight: 600;
        }}
        
        .input-wrapper.has-suffix input {{
            padding-right: 28px;
        }}
        
        /* Switch Toggle */
        .toggle-group {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(15, 23, 42, 0.3);
            padding: 12px 14px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            transition: all 0.2s ease;
        }}
        
        .toggle-group:hover {{
            border-color: rgba(255, 255, 255, 0.12);
            background: rgba(15, 23, 42, 0.4);
        }}
        
        .toggle-label-container {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        
        .toggle-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .toggle-desc {{
            font-size: 0.72rem;
            color: var(--text-secondary);
        }}
        
        .switch {{
            position: relative;
            display: inline-block;
            width: 46px;
            height: 24px;
            flex-shrink: 0;
        }}
        
        .switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}
        
        .slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(15, 23, 42, 0.8);
            transition: .3s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 34px;
            border: 1px solid var(--border-color);
        }}
        
        .slider:before {{
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 3px;
            bottom: 3px;
            background-color: var(--text-secondary);
            transition: .3s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 50%;
        }}
        
        input:checked + .slider {{
            background-color: rgba(6, 182, 212, 0.2);
            border-color: var(--accent-blue);
        }}
        
        input:checked + .slider:before {{
            transform: translateX(22px);
            background-color: var(--accent-blue);
        }}
        
        /* Optional Over/Under section */
        .ou-toggle-btn {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 10px 14px;
            font-size: 0.8rem;
            color: var(--accent-blue);
            cursor: pointer;
            font-weight: 600;
            user-select: none;
            transition: all 0.2s ease;
        }}
        
        .ou-toggle-btn:hover {{
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(6, 182, 212, 0.3);
        }}
        
        .ou-toggle-btn .caret {{
            font-size: 0.8rem;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .ou-toggle-btn.expanded .caret {{
            transform: rotate(180deg);
        }}
        
        .ou-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            background: rgba(15, 23, 42, 0.25);
            padding: 14px;
            border-radius: 12px;
            border: 1px dashed var(--border-color);
            max-height: 0;
            opacity: 0;
            overflow: hidden;
            transition: max-height 0.4s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s, padding 0.4s;
            padding-top: 0;
            padding-bottom: 0;
            border-width: 0;
        }}
        
        .ou-container.expanded {{
            max-height: 400px;
            opacity: 1;
            padding-top: 14px;
            padding-bottom: 14px;
            border-width: 1px;
            margin-top: 4px;
        }}
        
        .ou-team-column {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        .ou-team-title {{
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        /* Calculate Button */
        .calc-btn {{
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-green));
            border: none;
            border-radius: 12px;
            color: white;
            font-family: inherit;
            font-weight: 800;
            font-size: 0.95rem;
            padding: 12px 20px;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25);
            margin-top: 6px;
        }}
        
        .calc-btn:hover {{
            filter: brightness(1.08);
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4);
            transform: translateY(-1px);
        }}
        
        .calc-btn:active {{
            transform: translateY(1px);
            box-shadow: 0 2px 10px rgba(16, 185, 129, 0.2);
        }}
        
        /* Results Card style */
        .calc-results-wrapper {{
            display: none;
            flex-direction: column;
            gap: 12px;
            border-top: 1px solid var(--border-color);
            padding-top: 18px;
            margin-top: 4px;
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.4s ease-out;
        }}
        
        .calc-results-wrapper.visible {{
            display: flex;
            opacity: 1;
            transform: translateY(0);
        }}
        
        .calc-results-card {{
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: relative;
        }}
        
        .calculator-card.is-motd-active .calc-results-card {{
            background: rgba(245, 158, 11, 0.06);
            border-color: rgba(245, 158, 11, 0.18);
        }}
        
        .calc-results-details {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        
        .calc-results-score {{
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--accent-green);
            line-height: 1;
        }}
        
        .calculator-card.is-motd-active .calc-results-score {{
            color: var(--accent-yellow);
        }}
        
        .calc-error-message {{
            display: none;
            color: var(--accent-red);
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            padding: 8px 12px;
            border-radius: 8px;
            margin-top: 4px;
        }}
        
        /* Spinner */
        .spinner {{
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.6s linear infinite;
            display: none;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        /* Top 5 predictions table */
        .top-predictions-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 6px;
            font-size: 0.82rem;
        }}
        
        .top-predictions-table th, .top-predictions-table td {{
            padding: 6px 8px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }}
        
        .top-predictions-table th {{
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.05em;
        }}
        
        .top-predictions-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .top-predictions-table tr.rank-1 td {{
            color: var(--accent-green);
        }}
        
        .top-predictions-table tr.rank-1 {{
            background: rgba(16, 185, 129, 0.08);
            font-weight: 600;
        }}
        
        .is-motd .top-predictions-table tr.rank-1 td,
        .is-motd-active .top-predictions-table tr.rank-1 td {{
            color: var(--accent-yellow);
        }}
        
        .is-motd .top-predictions-table tr.rank-1,
        .is-motd-active .top-predictions-table tr.rank-1 {{
            background: rgba(245, 158, 11, 0.08);
        }}
        
        .badge-rank {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 800;
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-secondary);
        }}
        
        .rank-1 .badge-rank {{
            background: var(--accent-green);
            color: var(--bg-color);
        }}
        
        .is-motd .rank-1 .badge-rank,
        .is-motd-active .rank-1 .badge-rank {{
            background: var(--accent-yellow);
            color: var(--bg-color);
        }}
        
        details.top-predictions-details {{
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.01);
            transition: background 0.2s ease, border-color 0.2s ease;
        }}
        details.top-predictions-details[open] {{
            background: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.12);
        }}
        details.top-predictions-details summary {{
            list-style: none;
            outline: none;
        }}
        details.top-predictions-details summary::-webkit-details-marker {{
            display: none;
        }}
        details.top-predictions-details summary .toggle-icon {{
            transition: transform 0.2s ease;
            font-size: 0.7rem;
            color: var(--accent-blue);
        }}
        details.top-predictions-details[open] summary .toggle-icon {{
            transform: rotate(180deg);
        }}
        
        @media (max-width: 480px) {{
            .ou-container {{
                grid-template-columns: 1fr;
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

        <!-- Rekenmodule Calculator -->
        <div class="match-card calculator-card collapsed" id="prediction-calculator">
            <div class="calc-header" onclick="toggleCalculator()">
                <h2><span>🧮 Interactieve Rekenmodule</span></h2>
                <span class="calc-toggle-icon">▼</span>
            </div>
            
            <div class="calc-body">
                <div class="calc-section">
                    <span class="calc-section-title">📊 1X2 Kansen (Implied Odds)</span>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="input-win-home">Thuiswinst</label>
                            <div class="input-wrapper has-suffix">
                                <input type="number" id="input-win-home" value="66" min="0" max="100" step="1" oninput="validateSum1X2()">
                                <span class="input-suffix">%</span>
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="input-win-draw">Gelijkspel</label>
                            <div class="input-wrapper has-suffix">
                                <input type="number" id="input-win-draw" value="22" min="0" max="100" step="1" oninput="validateSum1X2()">
                                <span class="input-suffix">%</span>
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="input-win-away">Uitwinst</label>
                            <div class="input-wrapper has-suffix">
                                <input type="number" id="input-win-away" value="12" min="0" max="100" step="1" oninput="validateSum1X2()">
                                <span class="input-suffix">%</span>
                            </div>
                        </div>
                    </div>
                    <div class="calc-error-message" id="calc-error-1x2">De som van de kansen mag niet 0 zijn. De ingevoerde kansen worden automatisch genormaliseerd naar 100%.</div>
                </div>
                
                <div class="toggle-group">
                    <div class="toggle-label-container">
                        <span class="toggle-title">Wedstrijd van de Dag (MOTD)</span>
                        <span class="toggle-desc">Activeert extra punten voor doelpuntenmakers in de EV-berekening.</span>
                    </div>
                    <label class="switch">
                        <input type="checkbox" id="input-is-motd" onchange="toggleMotdStyle()">
                        <span class="slider"></span>
                    </label>
                </div>
                
                <div class="toggle-group" id="scorer-rate-wrapper" style="display: none; border-top: 1px solid var(--border-color); padding-top: 15px; margin-top: 15px;">
                    <div class="toggle-label-container">
                        <span class="toggle-title">Scorer Hit-Rate</span>
                        <span class="toggle-desc">Verwachte kans dat de spits scoort als het team scoort.</span>
                    </div>
                    <div class="input-wrapper" style="width: 80px;">
                        <input type="number" id="input-scorer-rate" value="{scorer_rate:.2f}" min="0.0" max="1.0" step="0.05" style="text-align: right; padding-right: 10px;">
                    </div>
                </div>


                
                <div class="calc-section">
                    <div class="ou-toggle-btn" onclick="toggleOuSection()" id="ou-btn">
                        <span>🛡️ Geavanceerde Team Over/Under Odds (Optioneel)</span>
                        <span class="caret">▼</span>
                    </div>
                    
                    <div class="ou-container" id="ou-fields-container">
                        <!-- Thuisploeg Over/Under -->
                        <div class="ou-team-column">
                            <div class="ou-team-title">🏠 Thuisploeg</div>
                            <div class="form-group">
                                <label for="ou-home-line">Doelpuntenlijn</label>
                                <div class="input-wrapper">
                                    <select id="ou-home-line">
                                        <option value="0.5">0.5</option>
                                        <option value="1.5" selected>1.5</option>
                                        <option value="2.5">2.5</option>
                                        <option value="3.5">3.5</option>
                                    </select>
                                </div>
                            </div>
                            <div class="form-row" style="grid-template-columns: 1fr 1fr;">
                                <div class="form-group">
                                    <label for="ou-home-under">Kans Under</label>
                                    <div class="input-wrapper has-suffix">
                                        <input type="number" id="ou-home-under" placeholder="Optioneel" min="0" max="100" step="1">
                                        <span class="input-suffix">%</span>
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label for="ou-home-over">Kans Over</label>
                                    <div class="input-wrapper has-suffix">
                                        <input type="number" id="ou-home-over" placeholder="Optioneel" min="0" max="100" step="1">
                                        <span class="input-suffix">%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Uitploeg Over/Under -->
                        <div class="ou-team-column">
                            <div class="ou-team-title">✈️ Uitploeg</div>
                            <div class="form-group">
                                <label for="ou-away-line">Doelpuntenlijn</label>
                                <div class="input-wrapper">
                                    <select id="ou-away-line">
                                        <option value="0.5">0.5</option>
                                        <option value="1.5" selected>1.5</option>
                                        <option value="2.5">2.5</option>
                                        <option value="3.5">3.5</option>
                                    </select>
                                </div>
                            </div>
                            <div class="form-row" style="grid-template-columns: 1fr 1fr;">
                                <div class="form-group">
                                    <label for="ou-away-under">Kans Under</label>
                                    <div class="input-wrapper has-suffix">
                                        <input type="number" id="ou-away-under" placeholder="Optioneel" min="0" max="100" step="1">
                                        <span class="input-suffix">%</span>
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label for="ou-away-over">Kans Over</label>
                                    <div class="input-wrapper has-suffix">
                                        <input type="number" id="ou-away-over" placeholder="Optioneel" min="0" max="100" step="1">
                                        <span class="input-suffix">%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="calc-section">
                    <div class="ou-toggle-btn" onclick="toggleExtraSection()" id="extra-btn">
                        <span>⚽ Wedstrijd O/U &amp; Extra Markten (Optioneel)</span>
                        <span class="caret">▼</span>
                    </div>

                    <div class="ou-container" id="extra-fields-container" style="grid-template-columns: 1fr 1fr;">
                        <!-- Wedstrijd Over/Under -->
                        <div class="ou-team-column">
                            <div class="ou-team-title">🥅 Wedstrijd Totaal</div>
                            <div class="form-group">
                                <label for="ou-match-line">Doelpuntenlijn</label>
                                <div class="input-wrapper">
                                    <select id="ou-match-line">
                                        <option value="1.5">1.5</option>
                                        <option value="2.5" selected>2.5</option>
                                        <option value="3.5">3.5</option>
                                        <option value="4.5">4.5</option>
                                    </select>
                                </div>
                            </div>
                            <div class="form-row" style="grid-template-columns: 1fr 1fr;">
                                <div class="form-group">
                                    <label for="ou-match-under">Kans Under</label>
                                    <div class="input-wrapper has-suffix">
                                        <input type="number" id="ou-match-under" placeholder="Optioneel" min="0" max="100" step="1">
                                        <span class="input-suffix">%</span>
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label for="ou-match-over">Kans Over</label>
                                    <div class="input-wrapper has-suffix">
                                        <input type="number" id="ou-match-over" placeholder="Optioneel" min="0" max="100" step="1">
                                        <span class="input-suffix">%</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Extra markten -->
                        <div class="ou-team-column">
                            <div class="ou-team-title">🎯 Extra Markten</div>
                            <div class="form-group">
                                <label for="input-btts">Beide teams scoren (Ja)</label>
                                <div class="input-wrapper has-suffix">
                                    <input type="number" id="input-btts" placeholder="Optioneel" min="0" max="100" step="1">
                                    <span class="input-suffix">%</span>
                                </div>
                            </div>
                            <div class="form-row" style="grid-template-columns: 1fr 1fr;">
                                <div class="form-group">
                                    <label for="input-cs-home">Clean sheet thuis</label>
                                    <div class="input-wrapper has-suffix">
                                        <input type="number" id="input-cs-home" placeholder="Opt." min="0" max="100" step="1">
                                        <span class="input-suffix">%</span>
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label for="input-cs-away">Clean sheet uit</label>
                                    <div class="input-wrapper has-suffix">
                                        <input type="number" id="input-cs-away" placeholder="Opt." min="0" max="100" step="1">
                                        <span class="input-suffix">%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <button class="calc-btn" onclick="calculatePrediction()">
                    <span class="spinner" id="calc-spinner"></span>
                    <span id="calc-btn-text">Bereken Optimale Voorspelling</span>
                </button>
                
                <!-- Resultaten -->
                <div class="calc-results-wrapper" id="results-wrapper">
                    <span class="calc-results-title">🏆 Wiskundig Optimaal Advies</span>
                    <div class="calc-results-card">
                        <div class="calc-results-details">
                            <span class="detail-label" style="color: var(--text-primary); font-size: 0.8rem;">Aanbevolen Uitslag</span>
                            <span class="subtitle" style="font-size: 0.75rem; margin-bottom: 2px;">
                                Verwachte Punten (EV): <strong style="color: var(--accent-blue);" id="result-ev">0.00 pts</strong>
                            </span>
                            <span class="subtitle" style="font-size: 0.75rem;" id="result-xg">
                                xG Thuis: 0.00 | xG Uit: 0.00 (ρ: 0.00)
                            </span>
                        </div>
                        <span class="calc-results-score" id="result-score">0-0</span>
                    </div>
                    
                    <div id="calc-top-5-container" style="margin-top: 8px;">
                        <!-- Hier komt de top 5 tabel dynamically -->
                    </div>

                    <div class="scorer-tips" id="calc-tiebreakers" style="margin-top: 8px;">
                        <!-- Hier komen de tie-breakers dynamically -->
                    </div>

                    <div class="scorer-tips" id="result-scorer-tips" style="display: none;">
                        <span class="detail-label" style="color: var(--accent-yellow);">Doelpuntenmaker Tips (MOTD)</span>
                        <div class="scorer-row">
                            <span class="scorer-team">Thuisploeg:</span>
                            <span class="scorer-name" id="result-scorer-home">Geen score</span>
                        </div>
                        <div class="scorer-row">
                            <span class="scorer-team">Uitploeg:</span>
                            <span class="scorer-name" id="result-scorer-away">Geen score</span>
                        </div>
                        <div class="scorer-factor-info" id="result-scorer-split-ev" style="margin-top: 8px; font-size: 0.75rem; color: var(--text-secondary); border-top: 1px solid var(--border-color); padding-top: 6px; display: flex; justify-content: space-between; width: 100%;">
                            <span>Score EV: <strong id="result-score-ev-val">0.00 pts</strong></span>
                            <span>Scorer EV: <strong id="result-scorer-ev-val">0.00 pts</strong></span>
                            <span>Factor: <strong id="result-scorer-factor-val">0.35</strong></span>
                        </div>
                    </div>

                </div>
            </div>
        </div>
"""
    
    for m, res in alle_res:
        # Genereer top 5 tabel HTML met risico-informatie
        top_5_html = '<table class="top-predictions-table"><thead><tr><th>Rank</th><th>Uitslag</th><th>EV</th><th>Kans</th><th>P(&ge;1pt)</th><th>P(&ge;5pt)</th><th style="text-align: right;">&Delta;EV</th></tr></thead><tbody>'
        for rank_idx, pred in enumerate(res.get("top_5", [])):
            rank = rank_idx + 1
            is_rank_1 = (rank == 1)
            rank_class = f"rank-{rank}"
            badge_text = "Hoofdadvies" if is_rank_1 else f"#{rank}"
            
            uitslag_val = pred["uitslag"]
            kans_val = f"{pred['kans']:.1f}%"
            ev_val_str = f"{pred['ev']:.2f} pts"
            p1_val_str = f"{pred.get('p_1pt', 0.0)*100:.1f}%"
            p5_val_str = f"{pred.get('p_5pt', 0.0)*100:.1f}%"
            
            delta_val = pred.get("delta_ev", 0.0)
            if delta_val == 0.0:
                delta_val_str = "0.00"
                delta_color = "var(--text-secondary)"
            elif delta_val > 0.0:
                delta_val_str = f"{delta_val:+.2f}"
                delta_color = "var(--accent-green)"
            else:
                delta_val_str = f"{delta_val:+.2f}"
                delta_color = "var(--accent-red)"
            
            top_5_html += f"""
            <tr class="{rank_class}">
                <td><span class="badge-rank">{badge_text}</span></td>
                <td><strong>{uitslag_val}</strong></td>
                <td>{ev_val_str}</td>
                <td>{kans_val}</td>
                <td>{p1_val_str}</td>
                <td>{p5_val_str}</td>
                <td style="text-align: right; font-weight: 600; color: {delta_color};">{delta_val_str}</td>
            </tr>
            """
        top_5_html += "</tbody></table>"

        motd_badge = '<span class="motd-badge">Wedstrijd van de Dag</span>' if m["is_motd"] else ''
        card_class = 'is-motd' if m["is_motd"] else ''
        datum_str = converteer_utc_naar_nl(m['date'])
        kansen_str = f"{m['home_prob']:.0f}% / {m['draw_prob']:.0f}% / {m['away_prob']:.0f}%"
        lam_h, lam_a = res["lambda"]
        rho = res.get("rho", 0.0)
        ev_val = res.get("xpts", 0.0)
        
        # Check of team totals beschikbaar zijn
        team_ou_home = m.get('team_ou_home', {})
        team_ou_away = m.get('team_ou_away', {})
        
        team_indicator_html = ""
        team_ou_html = ""
        if team_ou_home or team_ou_away:
            team_indicator_html = " ✓ team O/U"
            
            team_ou_html = '<div class="detail-item" style="grid-column: span 2;">'
            team_ou_html += '<span class="detail-label">Ploeg Totals (Polymarket)</span>'
            team_ou_html += '<div style="display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.85rem;">'
            
            for line in sorted(set(list(team_ou_home.keys()) + list(team_ou_away.keys()))):
                if line in team_ou_home:
                    u, o = team_ou_home[line]
                    team_ou_html += f'<span style="color: var(--accent-blue);">{m["home"]}: O{line} {o*100:.0f}%</span>'
                if line in team_ou_away:
                    u, o = team_ou_away[line]
                    team_ou_html += f'<span style="color: var(--accent-blue);">{m["away"]}: O{line} {o*100:.0f}%</span>'
            
            team_ou_html += '</div></div>'
            
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
                    <span class="detail-value">{lam_h:.2f} - {lam_a:.2f} (ρ: {rho:.2f}){team_indicator_html}</span>
                </div>
                {team_ou_html}
                
                <div class="prediction-box">
                    <div class="detail-item">
                        <span class="detail-label" style="color: var(--text-primary);">Aanbevolen Uitslag</span>
                        <span class="subtitle" style="font-size: 0.75rem;">Verwachte Punten (EV): <strong style="color: var(--accent-blue);">{res.get('xpts', 0.0):.2f} pts</strong></span>
                    </div>
                    <span class="pred-score">{res['uitslag']}</span>
                </div>
                
                <details class="top-predictions-details" style="grid-column: span 2; margin-top: 4px;">
                    <summary class="detail-label" style="cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: 600; color: var(--text-secondary); font-size: 0.75rem; user-select: none;">
                        <span>📊 Top 5 Verwachte Uitslagen (Risico-inzicht)</span>
                        <span class="toggle-icon">▼</span>
                    </summary>
                    <div style="margin-top: 8px;">
                        {top_5_html}
                    </div>
                </details>
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
                    <div class="scorer-factor-info" style="margin-top: 8px; font-size: 0.75rem; color: var(--text-secondary); border-top: 1px solid var(--border-color); padding-top: 6px; display: flex; justify-content: space-between;">
                        <span>Score EV: <strong>{res.get('score_ev', 0.0):.2f} pts</strong></span>
                        <span>Scorer EV: <strong>{res.get('scorer_ev', 0.0):.2f} pts</strong></span>
                        <span>Factor: <strong>{res.get('scorer_rate', SCORER_HIT_RATE):.2f}</strong></span>
                    </div>
                </div>
            """

        # Modelgebaseerde tie-breaker-voorspellingen (Prompt 13)
        tb = res.get("tie_breakers") or bereken_tie_breakers(lam_h, lam_a)
        rode_minuut = tb.get("rode_kaart_minuut")
        rode_str = f"{rode_minuut}e min" if rode_minuut is not None else "Geen (&gt; 90e)"
        html_content += f"""
                <details class="top-predictions-details" style="grid-column: span 2; margin-top: 4px;">
                    <summary class="detail-label" style="cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-weight: 600; color: var(--text-secondary); font-size: 0.75rem; user-select: none;">
                        <span>⏱️ Toernooi Tie-breakers (modelgebaseerd)</span>
                        <span class="toggle-icon">▼</span>
                    </summary>
                    <div style="margin-top: 8px; display: flex; flex-direction: column; gap: 4px; font-size: 0.8rem;">
                        <div class="scorer-row"><span class="scorer-team">1e doelpunt:</span><span class="scorer-name">{tb['eerste_doelpunt_minuut']}e min</span></div>
                        <div class="scorer-row"><span class="scorer-team">1e gele kaart:</span><span class="scorer-name">{tb['gele_kaart_minuut']}e min</span></div>
                        <div class="scorer-row"><span class="scorer-team">1e rode kaart:</span><span class="scorer-name">{rode_str}</span></div>
                    </div>
                </details>
        """

        html_content += """
            </div>
        </div>
        """

    html_content += """
    </div>

    <script>
        // --- CALCULATOR INTERFACE BINDINGS ---
        function toggleCalculator() {
            const calc = document.getElementById('prediction-calculator');
            calc.classList.toggle('collapsed');
        }

        function toggleOuSection() {
            const btn = document.getElementById('ou-btn');
            const container = document.getElementById('ou-fields-container');
            container.classList.toggle('expanded');
            if (container.classList.contains('expanded')) {
                btn.classList.add('expanded');
            } else {
                btn.classList.remove('expanded');
            }
        }

        function toggleExtraSection() {
            const btn = document.getElementById('extra-btn');
            const container = document.getElementById('extra-fields-container');
            container.classList.toggle('expanded');
            if (container.classList.contains('expanded')) {
                btn.classList.add('expanded');
            } else {
                btn.classList.remove('expanded');
            }
        }

        function toggleMotdStyle() {
            const calc = document.getElementById('prediction-calculator');
            const isMotd = document.getElementById('input-is-motd').checked;
            const scorerWrapper = document.getElementById('scorer-rate-wrapper');
            if (isMotd) {
                calc.classList.add('is-motd-active');
                if (scorerWrapper) scorerWrapper.style.display = 'flex';
            } else {
                calc.classList.remove('is-motd-active');
                if (scorerWrapper) scorerWrapper.style.display = 'none';
            }
        }

        function validateSum1X2() {
            const homeVal = parseFloat(document.getElementById('input-win-home').value) || 0;
            const drawVal = parseFloat(document.getElementById('input-win-draw').value) || 0;
            const awayVal = parseFloat(document.getElementById('input-win-away').value) || 0;
            const err = document.getElementById('calc-error-1x2');
            if (homeVal + drawVal + awayVal === 0) {
                err.style.display = 'block';
                return false;
            } else {
                err.style.display = 'none';
                return true;
            }
        }
"""

    # Gedeelde rekenkern: single source of truth, identiek aan de Python-functie voorspel().
    html_content += _calculator_core_js()

    html_content += """
        // --- UI GLUE: lees inputs, normaliseer markten (power) en render predictMatch() ---
        // Spiegelt de Polymarket-parsing: under/over en ja/nee worden met de power-methode
        // genormaliseerd voordat ze als kansen aan predictMatch worden doorgegeven.
        function readNormalizedOu(underId, overId, lineId) {
            let uRaw = document.getElementById(underId).value;
            let oRaw = document.getElementById(overId).value;
            if (uRaw === '' || oRaw === '') return null;
            let u = parseFloat(uRaw) || 0, o = parseFloat(oRaw) || 0;
            if (u + o <= 0) return null;
            let line = parseFloat(document.getElementById(lineId).value);
            let norm = normPower([u / 100.0, o / 100.0], 1.0);
            let d = {}; d[line] = [norm[0], norm[1]]; return d;
        }
        function readNormalizedYes(yesId) {
            let v = document.getElementById(yesId).value;
            if (v === '') return null;
            let y = parseFloat(v) || 0; if (y <= 0) return null;
            let norm = normPower([y / 100.0, 1.0 - y / 100.0], 1.0);
            return norm[0];
        }

        function calculatePrediction() {
            if (!validateSum1X2()) return;
            const btnText = document.getElementById('calc-btn-text');
            const spinner = document.getElementById('calc-spinner');
            const resultsWrapper = document.getElementById('results-wrapper');
            btnText.style.display = 'none';
            spinner.style.display = 'inline-block';
            setTimeout(() => {
                try {
                    let homePct = parseFloat(document.getElementById('input-win-home').value) || 0;
                    let drawPct = parseFloat(document.getElementById('input-win-draw').value) || 0;
                    let awayPct = parseFloat(document.getElementById('input-win-away').value) || 0;
                    let isMotd = document.getElementById('input-is-motd').checked;
                    let scorerRateInput = document.getElementById('input-scorer-rate');
                    let scorerRate = scorerRateInput ? parseFloat(scorerRateInput.value) : SCORER_HIT_RATE;
                    if (isNaN(scorerRate)) scorerRate = SCORER_HIT_RATE;

                    let inp = {
                        home_pct: homePct, draw_pct: drawPct, away_pct: awayPct, is_motd: isMotd,
                        team_ou_home: readNormalizedOu('ou-home-under', 'ou-home-over', 'ou-home-line'),
                        team_ou_away: readNormalizedOu('ou-away-under', 'ou-away-over', 'ou-away-line'),
                        ou_probs: readNormalizedOu('ou-match-under', 'ou-match-over', 'ou-match-line'),
                        btts_prob: readNormalizedYes('input-btts'),
                        clean_sheet_home_prob: readNormalizedYes('input-cs-home'),
                        clean_sheet_away_prob: readNormalizedYes('input-cs-away'),
                        scorer_rate: scorerRate
                    };

                    let res = predictMatch(inp);
                    let lamH = res.lambda[0], lamA = res.lambda[1], rho = res.rho;
                    let best = res.top_5[0];

                    document.getElementById('result-score').innerText = res.uitslag;
                    document.getElementById('result-ev').innerText = res.xpts.toFixed(2) + ' pts';
                    document.getElementById('result-xg').innerText = 'xG Thuis: ' + lamH.toFixed(2) + ' | xG Uit: ' + lamA.toFixed(2) + ' (\u03c1: ' + rho.toFixed(2) + ')';

                    let html = '<table class="top-predictions-table"><thead><tr><th>Rank</th><th>Uitslag</th><th>EV</th><th>Kans</th><th>P(&ge;1pt)</th><th>P(&ge;5pt)</th><th style="text-align: right;">&Delta;EV</th></tr></thead><tbody>';
                    res.top_5.forEach((pred, index) => {
                        let rank = index + 1;
                        let rankClass = 'rank-' + rank;
                        let badge = rank === 1 ? 'Hoofdadvies' : '#' + rank;
                        let dv = pred.delta_ev;
                        let dvStr = dv === 0 ? '0.00' : (dv > 0 ? '+' + dv.toFixed(2) : dv.toFixed(2));
                        let dColor = dv > 0 ? 'var(--accent-green)' : (dv === 0 ? 'var(--text-secondary)' : 'var(--accent-red)');
                        html += '<tr class="' + rankClass + '"><td><span class="badge-rank">' + badge + '</span></td>' +
                            '<td><strong>' + pred.uitslag + '</strong></td>' +
                            '<td>' + pred.ev.toFixed(2) + ' pts</td>' +
                            '<td>' + pred.kans.toFixed(1) + '%</td>' +
                            '<td>' + (pred.p_1pt * 100).toFixed(1) + '%</td>' +
                            '<td>' + (pred.p_5pt * 100).toFixed(1) + '%</td>' +
                            '<td style="text-align: right; font-weight: 600; color: ' + dColor + ';">' + dvStr + '</td></tr>';
                    });
                    html += '</tbody></table>';
                    document.getElementById('calc-top-5-container').innerHTML = html;

                    let tb = res.tie_breakers;
                    let rodeStr = tb.rode_kaart_minuut != null ? (tb.rode_kaart_minuut + 'e min') : 'Geen (&gt; 90e)';
                    document.getElementById('calc-tiebreakers').innerHTML =
                        '<span class="detail-label" style="color: var(--accent-blue);">\u23f1\ufe0f Toernooi Tie-breakers (modelgebaseerd)</span>' +
                        '<div class="scorer-row"><span class="scorer-team">1e doelpunt:</span><span class="scorer-name" style="color: var(--accent-blue);">' + tb.eerste_doelpunt_minuut + 'e min</span></div>' +
                        '<div class="scorer-row"><span class="scorer-team">1e gele kaart:</span><span class="scorer-name" style="color: var(--accent-blue);">' + tb.gele_kaart_minuut + 'e min</span></div>' +
                        '<div class="scorer-row"><span class="scorer-team">1e rode kaart:</span><span class="scorer-name" style="color: var(--accent-blue);">' + rodeStr + '</span></div>';

                    const scorerDiv = document.getElementById('result-scorer-tips');
                    if (isMotd) {
                        document.getElementById('result-scorer-home').innerText = best.scorers[0] ? 'Spits (of penaltynemer)' : 'Geen score';
                        document.getElementById('result-scorer-away').innerText = best.scorers[1] ? 'Spits (of penaltynemer)' : 'Geen score';
                        document.getElementById('result-score-ev-val').innerText = best.score_ev.toFixed(2) + ' pts';
                        document.getElementById('result-scorer-ev-val').innerText = best.scorer_ev.toFixed(2) + ' pts';
                        document.getElementById('result-scorer-factor-val').innerText = scorerRate.toFixed(2);
                        scorerDiv.style.display = 'flex';
                    } else {
                        scorerDiv.style.display = 'none';
                    }

                    spinner.style.display = 'none';
                    btnText.style.display = 'inline-block';
                    resultsWrapper.style.display = 'flex';
                    resultsWrapper.offsetHeight;
                    resultsWrapper.classList.add('visible');
                } catch (e) {
                    spinner.style.display = 'none';
                    btnText.style.display = 'inline-block';
                    console.error('Berekening mislukt:', e);
                }
            }, 400);
        }
    </script>
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

def polymarket_modus(toon_extra=False, output_file=None, loss_type="logloss", overround_method="power", verbose=False, weight_match_ou=None, weight_team_ou=None, weight_extra_markets=None, inclusief_top5=False, tiebreak="probability", scorer_rate=None):
    """
    Start de Polymarket-modus waarin de gebruiker live wedstrijden kan bekijken en voorspellen via de terminal.
    
    Parameters:
    toon_extra (bool, optioneel): Of er extra tie-breaker statistieken getoond moeten worden. Standaard False.
    output_file (str, optioneel): Bestand om voorspellingen naar te exporteren.
    loss_type (str, optioneel): De te gebruiken verliesfunctie ('mse' of 'logloss'). Standaard 'logloss'.
    overround_method (str, optioneel): De te gebruiken normalisatiemethode ('linear' of 'power'). Standaard 'power'.
    verbose (bool, optioneel): Of er debug-logging getoond moet worden voor de fits. Standaard False.
    weight_match_ou (float, optioneel): Het gewicht voor de wedstrijd Over/Under fit-termen.
    weight_team_ou (float, optioneel): Het gewicht voor de team Over/Under fit-termen.
    weight_extra_markets (float, optioneel): Het gewicht voor de extra markten fit-termen.
    inclusief_top5 (bool, optioneel): Of top-5 risico-analyse getoond/geëxporteerd moet worden.
    tiebreak (str, optioneel): De te gebruiken tie-breaker strategie. Standaard 'probability'.
    scorer_rate (float, optioneel): De scoringskans van de spits. Standaard SCORER_HIT_RATE.
    """
    print_header()
    print(f"{CYAN}{BOLD}Bezig met ophalen van actieve WK-wedstrijden en odds van Polymarket...{RESET}")
    matches, error = haal_polymarket_wedstrijden(overround_method=overround_method, verbose=verbose)
    if error:
        print(f"{RED}❌ {error}{RESET}")
        return
        
    if not matches:
        print(f"{YELLOW}Geen actieve WK-wedstrijden gevonden op Polymarket.{RESET}")
        return
        
    while True:
        print(f"{BOLD}GEVONDEN WEDSTRIJDEN:{RESET}")
        print("-" * 65)
        for idx, m in enumerate(matches, start=1):
            motd_lbl = f" {YELLOW}[MOTD]{RESET}" if m['is_motd'] else ""
            print(f"{idx:2d}. {m['home']:<20} vs. {m['away']:<20} ({converteer_utc_naar_nl(m['date'])}CET){motd_lbl}")
        print("-" * 65)
        print(f"{len(matches) + 1:2d}. {BOLD}Voorspel ALLE wedstrijden en toon samenvatting{RESET}")
        print(f"{len(matches) + 2:2d}. Stoppen en terug naar hoofdmenu")
        print("-" * 65)
        
        try:
            invoer = input(f"{BOLD}Kies een wedstrijd (1-{len(matches) + 2}): {RESET}").strip()
            if not invoer:
                continue
            val = int(invoer)
        except ValueError:
            print(f"{RED}Ongeldige keuze. Voer een getal in.{RESET}\n")
            continue
            
        if val == len(matches) + 2:
            break
            
        if 1 <= val <= len(matches):
            m = matches[val - 1]
            print(f"\n{BOLD}VERWERKT WEDSTRIJD: {m['home']} vs. {m['away']}{RESET}")
            print(f"  • Odds (1/X/2): {m['home_prob']:.1f}% / {m['draw_prob']:.1f}% / {m['away_prob']:.1f}%")
            if m.get('btts_prob') is not None:
                print(f"  • BTTS (Ja): {m['btts_prob']*100:.1f}%")
            if m.get('clean_sheet_home_prob') is not None:
                print(f"  • Clean Sheet {m['home']}: {m['clean_sheet_home_prob']*100:.1f}%")
            if m.get('clean_sheet_away_prob') is not None:
                print(f"  • Clean Sheet {m['away']}: {m['clean_sheet_away_prob']*100:.1f}%")
            
            # Vraag of dit MOTD is
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
            
            res = voorspel(
                m['home_prob'], m['draw_prob'], m['away_prob'],
                is_motd,
                ou_probs=m.get('ou_probs'),
                team_ou_home=m.get('team_ou_home'),
                team_ou_away=m.get('team_ou_away'),
                btts_prob=m.get('btts_prob'),
                clean_sheet_home_prob=m.get('clean_sheet_home_prob'),
                clean_sheet_away_prob=m.get('clean_sheet_away_prob'),
                loss_type=loss_type,
                overround_method=overround_method,
                verbose=verbose,
                weight_match_ou=weight_match_ou,
                weight_team_ou=weight_team_ou,
                weight_extra_markets=weight_extra_markets,
                tiebreak=tiebreak,
                scorer_rate=scorer_rate
            )
            print_resultaat(res, is_motd, toon_extra=toon_extra)
            break
            
        elif val == len(matches) + 1:
            # Voorspel ALLE wedstrijden
            print(f"\n{BOLD}Berekent voorspellingen voor alle {len(matches)} wedstrijden...{RESET}\n")
            
            alle_res = []
            for m in matches:
                res = voorspel(
                    m['home_prob'], m['draw_prob'], m['away_prob'],
                    is_motd=m['is_motd'],
                    ou_probs=m['ou_probs'],
                    team_ou_home=m.get('team_ou_home'),
                    team_ou_away=m.get('team_ou_away'),
                    btts_prob=m.get('btts_prob'),
                    clean_sheet_home_prob=m.get('clean_sheet_home_prob'),
                    clean_sheet_away_prob=m.get('clean_sheet_away_prob'),
                    loss_type=loss_type,
                    overround_method=overround_method,
                    verbose=verbose,
                    weight_match_ou=weight_match_ou,
                    weight_team_ou=weight_team_ou,
                    weight_extra_markets=weight_extra_markets,
                    tiebreak=tiebreak,
                    scorer_rate=scorer_rate
                )
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
                exporteer_naar_bestand(alle_res, output_file, inclusief_top5=inclusief_top5)
            break
        else:
            print(f"{RED}Ongeldig nummer. Kies een getal tussen 1 en {len(matches)+1}.{RESET}\n")

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
    parser.add_argument("--loss", choices=["mse", "logloss"], default="logloss", help="De te gebruiken verliesfunctie (standaard: logloss)")
    parser.add_argument("--overround", choices=["linear", "power"], default="power", help="De te gebruiken overround correctiemethode (standaard: power)")
    parser.add_argument("--weight-match-ou", type=float, default=None, help="Gewicht voor wedstrijd Over/Under fit (standaard: 0.5)")
    parser.add_argument("--weight-team-ou", type=float, default=None, help="Gewicht voor team Over/Under fit (standaard: 0.8)")
    parser.add_argument("--weight-extra-markets", type=float, default=None, dest="weight_extra_markets", help="Gewicht voor BTTS/Clean Sheet fit-termen (standaard: 0.6)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Toon extra debug-informatie, zoals optimalisatie-residuals")
    parser.add_argument("--top5", action="store_true", help="Inclusief top-5 risico-analyse in exports en CLI")
    parser.add_argument("--tiebreak", choices=["probability", "conservative"], default="probability", help="De te gebruiken tie-breaker strategie bij gelijke EV (standaard: probability)")
    parser.add_argument("--scorer-rate", type=float, default=None, help="De scoringskans van de spits bij MOTD (standaard: 0.35)")
    parser.add_argument("--yellow-base-min", type=float, default=None, dest="yellow_base_min", help="Basis-mediaan (minuut) eerste gele kaart voor tie-breakers (standaard: 30)")
    parser.add_argument("--red-card-rate", type=float, default=None, dest="red_card_rate", help="Verwacht aantal rode kaarten per wedstrijd voor tie-breakers (standaard: 0.22)")

    args = parser.parse_args()

    # Configureerbare tie-breaker-parameters: override de module-constanten indien opgegeven.
    global FIRST_YELLOW_BASE_MIN, RED_CARD_RATE
    if args.yellow_base_min is not None:
        FIRST_YELLOW_BASE_MIN = args.yellow_base_min
    if args.red_card_rate is not None:
        RED_CARD_RATE = args.red_card_rate
    
    # Als polymarket-modus is gekozen:
    if args.polymarket:
        if args.output or args.web:
            print_header()
            print(f"{CYAN}{BOLD}Bezig met batch-verwerking van alle WK-kansen van Polymarket...{RESET}")
            matches, error = haal_polymarket_wedstrijden(overround_method=args.overround, verbose=args.verbose)
            if error:
                print(f"{RED}❌ {error}{RESET}")
                sys.exit(1)
            if not matches:
                print(f"{YELLOW}Geen actieve WK-wedstrijden gevonden op Polymarket.{RESET}")
                sys.exit(0)
                
            alle_res = []
            for m in matches:
                res = voorspel(
                    m['home_prob'], m['draw_prob'], m['away_prob'],
                    is_motd=m['is_motd'],
                    ou_probs=m['ou_probs'],
                    team_ou_home=m.get('team_ou_home'),
                    team_ou_away=m.get('team_ou_away'),
                    btts_prob=m.get('btts_prob'),
                    clean_sheet_home_prob=m.get('clean_sheet_home_prob'),
                    clean_sheet_away_prob=m.get('clean_sheet_away_prob'),
                    loss_type=args.loss,
                    overround_method=args.overround,
                    verbose=args.verbose,
                    weight_match_ou=args.weight_match_ou,
                    weight_team_ou=args.weight_team_ou,
                    weight_extra_markets=args.weight_extra_markets,
                    tiebreak=args.tiebreak,
                    scorer_rate=args.scorer_rate
                )
                alle_res.append((m, res))
                
            if args.output:
                exporteer_naar_bestand(alle_res, args.output, inclusief_top5=args.top5)
            if args.web:
                exporteer_naar_html(alle_res, args.web, scorer_rate=args.scorer_rate)
            sys.exit(0)
        else:
            try:
                polymarket_modus(
                    toon_extra=args.extra,
                    loss_type=args.loss,
                    overround_method=args.overround,
                    verbose=args.verbose,
                    weight_match_ou=args.weight_match_ou,
                    weight_team_ou=args.weight_team_ou,
                    weight_extra_markets=args.weight_extra_markets,
                    inclusief_top5=args.top5,
                    tiebreak=args.tiebreak,
                    scorer_rate=args.scorer_rate
                )
            except (KeyboardInterrupt, SystemExit):
                print(f"\n\n{YELLOW}Programma afgebroken. Tot ziens! 👋{RESET}\n")
            sys.exit(0)
            
    # Als er geen argumenten zijn opgegeven of specifiek --interactive is meegegeven:
    if args.interactive or (args.home is None and args.draw is None and args.away is None):
        try:
            interactieve_modus(
                toon_extra=args.extra,
                loss_type=args.loss,
                overround_method=args.overround,
                verbose=args.verbose,
                weight_match_ou=args.weight_match_ou,
                weight_team_ou=args.weight_team_ou,
                tiebreak=args.tiebreak,
                scorer_rate=args.scorer_rate
            )
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
            
        res = voorspel(
            home, draw, away, args.motd,
            loss_type=args.loss,
            overround_method=args.overround,
            verbose=args.verbose,
            weight_match_ou=args.weight_match_ou,
            weight_team_ou=args.weight_team_ou,
            tiebreak=args.tiebreak,
            scorer_rate=args.scorer_rate
        )
        print_header()
        print_resultaat(res, args.motd, toon_extra=args.extra)


if __name__ == "__main__":
    main()
