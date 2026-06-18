#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voetbalpoules Polymarket Voorspeller (CLI Wrapper)
--------------------------------------------------
Deze module fungeert als dunne startplek (entrypoint) voor de CLI.
Het importeert de berekeningsregels, modelbepalingen en dataladers uit de model/, data/ en export/ modules.
Exporteert ook de basisfuncties ter compatibiliteit met backtest.py en test_js_parity.py.
"""

import sys
import argparse
import json
import os
import math
import datetime
from zoneinfo import ZoneInfo

# Exposeer de functies voor test_js_parity.py en backtest.py
from model.overround import normaliseer_kansen
from model.poisson import (
    calc_matrix,
    calc_matrix_nb,
    bepaal_poisson_lambdas,
    bereken_tie_breakers,
    PARSING_WARNINGS,
    SCORER_HIT_RATE
)
from model.ev import voorspel, calc_ev_regular, calc_ev_motd
from data.polymarket import haal_polymarket_wedstrijden, normalize_team_name
from export.text import exporteer_naar_bestand, converteer_utc_naar_nl
from export.html import exporteer_naar_html, _calculator_core_js

# ANSI Kleurcodes voor mooie terminal-output
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"

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
    """
    val_str = val_str.strip().replace('%', '')
    try:
        val = float(val_str)
        if 0.0 <= val <= 1.0:
            val = val * 100.0
        return val
    except ValueError:
        raise ValueError(f"Ongeldig getal: '{val_str}'")

def print_resultaat(res, is_motd, toon_extra=False):
    """
    Toont de geanalyseerde gegevens en het voorspellingsadvies op een overzichtelijke manier in de terminal.
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
            
        if delta_val > 0.0:
            delta_colored = f"{GREEN}{delta_val_str:<6}{RESET}"
        elif delta_val < 0.0:
            delta_colored = f"{RED}{delta_val_str:<6}{RESET}"
        else:
            delta_colored = f"{delta_val_str:<6}"
            
        print(f"  {u_val:<7} | {ev_val:<5} | {kans_val:<5} | {p5_val:<8} | {delta_colored}")
        
    if toon_extra:
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

def interactieve_modus(toon_extra=False, loss_type="logloss", overround_method="power", verbose=False, weight_match_ou=None, weight_team_ou=None, tiebreak="probability", scorer_rate=None, model="negbinom"):
    """
    Start een interactief vraag-en-antwoordscherm in de terminal om een voorspelling voor één wedstrijd te berekenen.
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
            
    res = voorspel(home, draw, away, is_motd, loss_type=loss_type, overround_method=overround_method, verbose=verbose, weight_match_ou=weight_match_ou, weight_team_ou=weight_team_ou, tiebreak=tiebreak, scorer_rate=scorer_rate, model=model)
    print_resultaat(res, is_motd, toon_extra=toon_extra)

def polymarket_modus(toon_extra=False, output_file=None, loss_type="logloss", overround_method="power", verbose=False, weight_match_ou=None, weight_team_ou=None, weight_extra_markets=None, inclusief_top5=False, tiebreak="probability", scorer_rate=None, motd_file="data/motd.json", model="negbinom"):
    """
    Start de Polymarket-modus waarin de gebruiker live wedstrijden kan bekijken en voorspellen via de terminal.
    """
    print_header()
    print(f"{CYAN}{BOLD}Bezig met ophalen van actieve WK-wedstrijden en odds van Polymarket...{RESET}")
    matches, error = haal_polymarket_wedstrijden(overround_method=overround_method, verbose=verbose, motd_file=motd_file)
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
                scorer_rate=scorer_rate,
                model=model
            )
            print_resultaat(res, is_motd, toon_extra=toon_extra)
            break
            
        elif val == len(matches) + 1:
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
                    scorer_rate=scorer_rate,
                    model=model
                )
                alle_res.append((m, res))
            
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
    parser.add_argument("--weight-match-ou", type=float, default=None, help="Gewicht voor wedstrijd Over/Under fit (standaard: 1.0)")
    parser.add_argument("--weight-team-ou", type=float, default=None, help="Gewicht voor team Over/Under fit (standaard: 0.5)")
    parser.add_argument("--weight-extra-markets", type=float, default=None, dest="weight_extra_markets", help="Gewicht voor BTTS/Clean Sheet fit-termen (standaard: 0.6)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Toon extra debug-informatie, zoals optimalisatie-residuals")
    parser.add_argument("--top5", action="store_true", help="Inclusief top-5 risico-analyse in exports en CLI")
    parser.add_argument("--tiebreak", choices=["probability", "conservative"], default="probability", help="De te gebruiken tie-breaker strategie bij gelijke EV (standaard: probability)")
    parser.add_argument("--scorer-rate", type=float, default=None, help="De scoringskans van de spits bij MOTD (standaard: 0.35)")
    parser.add_argument("--yellow-base-min", type=float, default=None, dest="yellow_base_min", help="Basis-mediaan (minuut) eerste gele kaart voor tie-breakers (standaard: 30)")
    parser.add_argument("--red-card-rate", type=float, default=None, dest="red_card_rate", help="Verwacht aantal rode kaarten per wedstrijd voor tie-breakers (standaard: 0.22)")
    parser.add_argument("--motd-file", type=str, default="data/motd.json", help="Pad naar het JSON-bestand met MOTD team-paren (standaard: data/motd.json)")
    parser.add_argument("--model", choices=["poisson", "negbinom"], default="negbinom", help="Het te gebruiken statistische model (standaard: negbinom)")

    args = parser.parse_args()

    # Configureerbare tie-breaker-parameters: override de module-constanten indien opgegeven.
    import model.poisson
    if args.yellow_base_min is not None:
        model.poisson.FIRST_YELLOW_BASE_MIN = args.yellow_base_min
    if args.red_card_rate is not None:
        model.poisson.RED_CARD_RATE = args.red_card_rate
    
    # Als polymarket-modus is gekozen:
    if args.polymarket:
        if args.output or args.web:
            print_header()
            print(f"{CYAN}{BOLD}Bezig met batch-verwerking van alle WK-kansen van Polymarket...{RESET}")
            matches, error = haal_polymarket_wedstrijden(overround_method=args.overround, verbose=args.verbose, motd_file=args.motd_file)
            if error:
                print(f"{RED}❌ {error}{RESET}")
                sys.exit(1)
            if not matches:
                print(f"{YELLOW}Geen actieve WK-wedstrijden gevonden op Polymarket.{RESET}")
                sys.exit(0)
                
            alle_res = []
            for m in matches:
                is_motd = True if args.motd else m['is_motd']
                res = voorspel(
                    m['home_prob'], m['draw_prob'], m['away_prob'],
                    is_motd=is_motd,
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
                    scorer_rate=args.scorer_rate,
                    model=args.model
                )
                alle_res.append((m, res))
                
            # Schrijf parse log indien verbose of in CI
            if args.verbose or os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
                log_data = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "num_matches": len(matches),
                    "warnings": PARSING_WARNINGS,
                    "matches": [
                        {
                            "home": m["home"],
                            "away": m["away"],
                            "is_motd": m["is_motd"],
                            "odds_1x2": [m["home_prob"], m["draw_prob"], m["away_prob"]],
                        } for m in matches
                    ]
                }
                os.makedirs("data", exist_ok=True)
                try:
                    with open("data/parse_log.json", "w", encoding="utf-8") as pf:
                        json.dump(log_data, pf, indent=2)
                    if args.verbose:
                        print(f"{GREEN}✓ Parse log succesvol opgeslagen in data/parse_log.json{RESET}")
                except Exception as log_err:
                    print(f"{YELLOW}⚠️ Waarschuwing: Kon parse log niet opslaan: {log_err}{RESET}")
                
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
                    scorer_rate=args.scorer_rate,
                    motd_file=args.motd_file,
                    model=args.model
                )
            except (KeyboardInterrupt, SystemExit):
                print(f"\n\n{YELLOW}Programma afgebroken. Tot ziens! 👋{RESET}\n")
            sys.exit(0)
            
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
                scorer_rate=args.scorer_rate,
                model=args.model
            )
        except (KeyboardInterrupt, SystemExit):
            print(f"\n\n{YELLOW}Programma afgebroken. Tot ziens! 👋{RESET}\n")
    else:
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
            scorer_rate=args.scorer_rate,
            model=args.model
        )
        print_header()
        print_resultaat(res, args.motd, toon_extra=args.extra)

if __name__ == "__main__":
    main()
