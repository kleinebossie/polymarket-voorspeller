#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overround Normalisatie Module
-----------------------------
Functies voor het normaliseren van marktkansen door de winstmarge (overround) te
verwijderen. Ondersteunt lineaire schaling en de betrouwbaardere power-methode.
"""

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
