#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Expected Value (EV) en Voorspel Module
-------------------------------------
Berekent de verwachte waarde in punten en zoekt de optimale uitslag die de EV maximaliseert.
"""

import math
from model.overround import normaliseer_kansen
from model.poisson import (
    bepaal_poisson_lambdas,
    calc_matrix,
    calc_matrix_nb,
    bereken_tie_breakers,
    SCORER_HIT_RATE
)

def calc_ev_regular(pred_h, pred_a, matrix):
    """
    Berekent de verwachte waarde (Expected Value, EV) in punten voor een voorspelde uitslag in een normale poule.
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

def voorspel(home_pct, draw_pct, away_pct, is_motd, ou_probs=None, team_ou_home=None, team_ou_away=None, btts_prob=None, clean_sheet_home_prob=None, clean_sheet_away_prob=None, loss_type="logloss", overround_method="power", verbose=False, weight_match_ou=None, weight_team_ou=None, weight_extra_markets=None, tiebreak="probability", scorer_rate=None, model="poisson"):
    """
    Berekent de optimale voorspelling door de uitslag te zoeken die de verwachte waarde (EV) maximaliseert.
    """
    if scorer_rate is None:
        scorer_rate = SCORER_HIT_RATE
        
    p_h, p_d, p_a = normaliseer_kansen(home_pct, draw_pct, away_pct, method=overround_method)
    if model == "negbinom":
        lam_h, lam_a, rho, r_val = bepaal_poisson_lambdas(p_h, p_d, p_a, ou_probs, target_team_ou_home=team_ou_home, target_team_ou_away=team_ou_away, target_btts=btts_prob, target_clean_sheet_home=clean_sheet_home_prob, target_clean_sheet_away=clean_sheet_away_prob, loss_type=loss_type, verbose=verbose, weight_match_ou=weight_match_ou, weight_team_ou=weight_team_ou, weight_extra_markets=weight_extra_markets, model=model)
        matrix = calc_matrix_nb(lam_h, lam_a, rho, r_val)
    else:
        lam_h, lam_a, rho = bepaal_poisson_lambdas(p_h, p_d, p_a, ou_probs, target_team_ou_home=team_ou_home, target_team_ou_away=team_ou_away, target_btts=btts_prob, target_clean_sheet_home=clean_sheet_home_prob, target_clean_sheet_away=clean_sheet_away_prob, loss_type=loss_type, verbose=verbose, weight_match_ou=weight_match_ou, weight_team_ou=weight_team_ou, weight_extra_markets=weight_extra_markets, model=model)
        r_val = None
        matrix = calc_matrix(lam_h, lam_a, rho)
    
    max_score = min(9, math.ceil(max(lam_h, lam_a) + 2))
    buiten_raster_massa = sum(prob for (act_h, act_a), prob in matrix.items() if act_h > max_score or act_a > max_score)

    if buiten_raster_massa > 0.01 and max_score < 9:
        max_score += 1
        buiten_raster_massa = sum(prob for (act_h, act_a), prob in matrix.items() if act_h > max_score or act_a > max_score)
        
    if verbose:
        print(f"  [Verbose Raster] EV-zoekraster: 0-{max_score} voor beide teams | Buiten-raster-massa: {buiten_raster_massa*100:.2f}%")
        print(f"  [Verbose Tie-Break] Gekozen via tie-break: {tiebreak}")
        
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
        scorer_tip_onafhankelijk = (best_scorers[0] != (best_pred[0] > 0)) or (best_scorers[1] != (best_pred[1] > 0))
    else:
        scorer_thuis = ""
        scorer_uit = ""
        best_score_ev = best_ev
        best_scorer_ev = 0.0
        uitleg = f"Maximale EV: {best_ev:.2f} verwachte punten."
        scorer_tip_onafhankelijk = False

    tie_breakers = bereken_tie_breakers(lam_h, lam_a)

    return {
        "genormaliseerd": (p_h, p_d, p_a),
        "lambda": (lam_h, lam_a),
        "rho": rho,
        "r": r_val,
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
