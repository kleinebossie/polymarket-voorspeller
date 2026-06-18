#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Poisson en Negatieve Binomiaal Modelfitting Module
--------------------------------------------------
Bevat alle statistische verdelingen, parameter-schattingen en Dixon-Coles aanpassingen.
"""

import math
import random
from scipy.optimize import minimize

# ANSI Kleurcodes voor debug-logging
RESET = "\033[0m"
YELLOW = "\033[93m"

# Standaard optimalisatie-gewichten voor Poisson lambda-bepaling (fit)
WEIGHT_MATCH_OU = 1.0
WEIGHT_TEAM_OU = 0.5
WEIGHT_EXTRA_MARKETS = 0.6
SCORER_HIT_RATE = 0.35

# Lichte Tikhonov-regularisatie op de Dixon-Coles ρ-parameter.
RHO_REG = 1e-3

# Verzameling van parsing/fitting waarschuwingen (gebruikt voor data/parse_log.json)
PARSING_WARNINGS = []

# Tie-breaker (toernooi-vragen) model-constanten
FIRST_YELLOW_BASE_MIN = 30.0
FIRST_YELLOW_REF_LAMBDA = 2.6
RED_CARD_RATE = 0.22
RED_CARD_THRESHOLD = 0.5

def poisson(lam, k):
    """
    Berekent de kans op exact k doelpunten met een bepaald gemiddelde (lambda).
    """
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def neg_binom(lam, y, r):
    """
    Berekent de kans op exact y doelpunten met een bepaald gemiddelde (lambda)
    en een overdispersieparameter r volgens de Negatieve Binomiaal-verdeling.
    """
    if lam <= 0:
        return 1.0 if y == 0 else 0.0
    if r > 1000.0:
        return poisson(lam, y)
    
    if y == 0:
        coeff = 1.0
    else:
        num = 1.0
        den = 1.0
        for i in range(y):
            num *= (r + i)
            den *= (i + 1)
        coeff = num / den
        
    p = r / (r + lam)
    q = lam / (r + lam)
    
    try:
        log_prob = math.log(coeff) + r * math.log(p) + y * math.log(q)
        return math.exp(log_prob)
    except (ValueError, OverflowError):
        return coeff * (p ** r) * (q ** y)

def dixon_coles_tau(h, a, lam_h, lam_a, rho):
    """
    Berekent de Dixon-Coles correctiefactor voor uitslagen met lage scores (0 en 1 doelpunten).
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

def calc_matrix_nb(lam_h, lam_a, rho=0.0, r=4.0):
    """
    Berekent de kansenmatrix voor uitslagen van 0-0 tot 9-9 op basis van de Negatieve Binomiaal-verdeling
    en de Dixon-Coles correctieparameter.
    """
    matrix = {}
    totaal = 0.0
    for h in range(10):
        for a in range(10):
            prob = neg_binom(lam_h, h, r) * neg_binom(lam_a, a, r) * dixon_coles_tau(h, a, lam_h, lam_a, rho)
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
    """
    h_win = sum(p for (h,a), p in matrix.items() if h > a)
    d = sum(p for (h,a), p in matrix.items() if h == a)
    a_win = sum(p for (h,a), p in matrix.items() if h < a)
    return h_win, d, a_win

def bepaal_poisson_lambdas(target_h, target_d, target_a, target_ou=None, target_team_ou_home=None, target_team_ou_away=None, target_btts=None, target_clean_sheet_home=None, target_clean_sheet_away=None, loss_type="logloss", verbose=False, weight_match_ou=None, weight_team_ou=None, weight_extra_markets=None, model="poisson"):
    """
    Vindt de optimale parameters die het beste aansluiten bij de gewenste winst-, gelijkspel- en verlieskansen
    (en eventuele over/under kansen). Ondersteunt zowel Poisson als Negatieve Binomiaal.
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
        if model == "negbinom":
            lh, la, rho_val, r_val = params
            matrix = calc_matrix_nb(lh, la, rho_val, r_val)
        else:
            lh, la, rho_val = params
            matrix = calc_matrix(lh, la, rho_val)
            
        h, d, a = get_1x2_and_ou(matrix)
        
        eps = 1e-15
        if loss_type == "logloss":
            error = -(
                target_h * math.log(max(h, eps)) +
                target_d * math.log(max(d, eps)) +
                target_a * math.log(max(a, eps))
            )
        else:
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

        # Lichte ρ-regularisatie
        error += RHO_REG * (rho_val ** 2)

        return error

    # Optimalisatie bounds
    if model == "negbinom":
        bounds = [
            (0.05, 5.0),    # lam_h
            (0.05, 5.0),    # lam_a
            (-0.25, 0.10),  # rho
            (0.1, 50.0)     # r
        ]
    else:
        bounds = [
            (0.05, 5.0),    # lam_h
            (0.05, 5.0),    # lam_a
            (-0.25, 0.10)   # rho
        ]

    # Genereer startpunten
    if model == "negbinom":
        start_points = [[1.3, 1.0, -0.05, 4.0]]
        m_h = -math.log(max(0.01, min(0.99, target_d + target_a)))
        m_a = -math.log(max(0.01, min(0.99, target_h + target_d)))
        m_h = max(0.05, min(m_h, 5.0))
        m_a = max(0.05, min(m_a, 5.0))
        start_points.append([m_h, m_a, -0.05, 4.0])
        start_points.append([0.5, 0.5, -0.05, 2.0])
        start_points.append([2.5, 1.5, -0.10, 8.0])
        start_points.append([1.0, 2.5, 0.00, 5.0])
        start_points.append([3.5, 0.8, -0.15, 12.0])
        start_points.append([0.3, 3.0, 0.05, 6.0])
    else:
        start_points = [[1.3, 1.0, -0.05]]
        m_h = -math.log(max(0.01, min(0.99, target_d + target_a)))
        m_a = -math.log(max(0.01, min(0.99, target_h + target_d)))
        m_h = max(0.05, min(m_h, 5.0))
        m_a = max(0.05, min(m_a, 5.0))
        start_points.append([m_h, m_a, -0.05])
        
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

    if model == "negbinom":
        lam_h_opt, lam_a_opt, rho_opt, r_opt = best_params
    else:
        lam_h_opt, lam_a_opt, rho_opt = best_params
        r_opt = None
    
    # Valideer fit-residual en lambda ranges
    try:
        if model == "negbinom":
            fit_matrix = calc_matrix_nb(lam_h_opt, lam_a_opt, rho_opt, r_opt)
        else:
            fit_matrix = calc_matrix(lam_h_opt, lam_a_opt, rho_opt)
            
        fit_h, fit_d, fit_a = get_1x2_and_ou(fit_matrix)
        residual = max(abs(fit_h - target_h), abs(fit_d - target_d), abs(fit_a - target_a))
        
        global PARSING_WARNINGS
        if residual > 0.03:
            msg = f"Fit-residual ({residual:.4f}) overschrijdt drempel (0.03) voor wedstrijd!"
            print(f"{YELLOW}⚠️ Waarschuwing: {msg}{RESET}")
            PARSING_WARNINGS.append(msg)
            
        if not (0.05 <= lam_h_opt <= 5.0) or not (0.05 <= lam_a_opt <= 5.0):
            msg = f"Gekozen lambda's ({lam_h_opt:.4f}, {lam_a_opt:.4f}) vallen buiten het bereik [0.05, 5.0]!"
            print(f"{YELLOW}⚠️ Waarschuwing: {msg}{RESET}")
            PARSING_WARNINGS.append(msg)
    except Exception:
        pass
    
    if verbose:
        if model == "negbinom":
            print(f"  [Debug Fit] Optimal parameters (negbinom): lam_h={lam_h_opt:.4f}, lam_a={lam_a_opt:.4f}, rho={rho_opt:.4f}, r={r_opt:.4f} | Fit-residual ({loss_type}): {best_loss:.6f}")
        else:
            print(f"  [Debug Fit] Optimal parameters (poisson): lam_h={lam_h_opt:.4f}, lam_a={lam_a_opt:.4f}, rho={rho_opt:.4f} | Fit-residual ({loss_type}): {best_loss:.6f}")

    if model == "negbinom":
        return lam_h_opt, lam_a_opt, rho_opt, r_opt
    else:
        return lam_h_opt, lam_a_opt, rho_opt

def bereken_tie_breakers(lam_h, lam_a, yellow_base_min=None, yellow_ref_lambda=None,
                          red_card_rate=None, red_card_threshold=None):
    """
    Leidt de toernooi-tie-breaker-voorspellingen af uit het Poisson-model i.p.v. uit
    vaste constanten. Hierdoor varieert de uitkomst per wedstrijd op basis van de λ's.
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
        rode_kaart_minuut = None
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
