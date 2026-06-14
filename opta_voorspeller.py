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
    """Verwerkt invoer als percentage (bijv. '45', '45%', '0.45') en geeft een float terug."""
    val_str = val_str.strip().replace('%', '')
    try:
        val = float(val_str)
        # Als de invoer tussen 0 en 1 ligt (bijv. 0.45), reken om naar percentage (45)
        if 0.0 <= val <= 1.0:
            val = val * 100.0
        return val
    except ValueError:
        raise ValueError(f"Ongeldig getal: '{val_str}'")

def normaliseer_kansen(home, draw, away):
    """Zorgt dat de som van de kansen precies 1.0 (100%) is."""
    totaal = home + draw + away
    if totaal == 0:
        return 0.0, 0.0, 0.0
    return home / totaal, draw / totaal, away / totaal

def bereken_lambda(p_h_prime, p_a_prime):
    """Berekent de verwachte doelpunten (lambda/xG) per team."""
    lambda_h = 1.05 + 1.85 * (p_h_prime - p_a_prime) + 0.5 * p_h_prime
    lambda_a = 1.05 - 1.85 * (p_h_prime - p_a_prime) + 0.5 * p_a_prime
    
    # Grenscontroles volgens het rapport (tussen 0.3 en 3.5)
    lambda_h = max(0.3, min(3.5, lambda_h))
    lambda_a = max(0.3, min(3.5, lambda_a))
    return lambda_h, lambda_a

def geef_uitleg_normaal(p_h_prime, p_a_prime, uitslag):
    """Geeft een eenvoudige wiskundige verklaring voor de gekozen uitslag."""
    if uitslag == "1-1":
        return (
            "De kansen liggen dicht bij elkaar (geen van beide teams heeft meer dan 55% winstkans).\n"
            "Omdat een goed voorspeld gelijkspel 7 punten oplevert (tegenover 5 voor winst),\n"
            "is 1-1 wiskundig veruit de veiligste keuze met de hoogste verwachte waarde."
        )
    else:
        fav_team = "thuisploeg" if p_h_prime > p_a_prime else "uitploeg"
        fav_pct = max(p_h_prime, p_a_prime) * 100
        return (
            f"De {fav_team} is een zware favoriet met maar liefst {fav_pct:.1f}% winstkans (meer dan 55%).\n"
            "Hierdoor kantelt de wiskunde in het voordeel van een overwinning.\n"
            f"De uitslag {uitslag} balanceert de kans op een clean sheet versus een doelpunt van de tegenstander."
        )

def geef_uitleg_motd(p_h_prime, p_a_prime, uitslag):
    """Geeft uitleg voor de Wedstrijd van de Dag uitslag."""
    if uitslag == "0-0":
        return (
            "Er is geen extreme favoriet in deze wedstrijd (winstkansen onder 48%).\n"
            "Hierdoor is de wiskundige 'Geen Score' (0-0) truc van kracht!\n"
            "Spitsen scoren in minder dan 35% van de gevallen de eerste goal, maar 'Geen score' is\n"
            "altijd 100% gegarandeerd correct als een team 0 goals maakt. Dit levert 8 extra bonuspunten op!"
        )
    else:
        fav_team = "thuisploeg" if p_h_prime > p_a_prime else "uitploeg"
        return (
            f"De {fav_team} is een overduidelijke favoriet (meer dan 48% winstkans) en de tegenstander is erg zwak.\n"
            f"Daarom voorspellen we een {uitslag} overwinning.\n"
            "Voor de favoriet vullen we de topspits in omdat de kans op goals erg groot is.\n"
            "Voor de underdog vullen we nog steeds 'Geen score' in om daar de veilige bonus te pakken."
        )

def voorspel(home_pct, draw_pct, away_pct, is_motd):
    """Berekent de optimale uitslag op basis van de beslisboom in het rapport."""
    # Stap 1: Normaliseren
    p_h_prime, p_d_prime, p_a_prime = normaliseer_kansen(home_pct, draw_pct, away_pct)
    
    # Stap 2: Heuristische verwachte doelpunten (lambda/xG)
    lambda_h, lambda_a = bereken_lambda(p_h_prime, p_a_prime)
    
    uitslag = ""
    scorer_thuis = ""
    scorer_uit = ""
    uitleg = ""
    
    # Stap 3: Algoritmische bepaling
    if not is_motd:
        # A. Normale Wedstrijd
        if p_h_prime > 0.55:
            if (p_d_prime + p_a_prime) < 0.35:
                uitslag = "2-0"
            else:
                uitslag = "2-1"
        elif p_a_prime > 0.55:
            if (p_h_prime + p_d_prime) < 0.35:
                uitslag = "0-2"
            else:
                uitslag = "1-2"
        else:
            uitslag = "1-1"
        uitleg = geef_uitleg_normaal(p_h_prime, p_a_prime, uitslag)
    else:
        # B. Wedstrijd van de Dag (MOTD)
        if p_h_prime > 0.48 and p_a_prime < 0.25:
            uitslag = "2-0"
            scorer_thuis = "Primaire startende spits (en vaste penaltynemer)"
            scorer_uit = "Geen score"
        elif p_a_prime > 0.48 and p_h_prime < 0.25:
            uitslag = "0-2"
            scorer_thuis = "Geen score"
            scorer_uit = "Primaire startende spits (en vaste penaltynemer)"
        else:
            uitslag = "0-0"
            scorer_thuis = "Geen score"
            scorer_uit = "Geen score"
        uitleg = geef_uitleg_motd(p_h_prime, p_a_prime, uitslag)
        
    return {
        "genormaliseerd": (p_h_prime, p_d_prime, p_a_prime),
        "lambda": (lambda_h, lambda_a),
        "uitslag": uitslag,
        "scorer_thuis": scorer_thuis,
        "scorer_uit": scorer_uit,
        "uitleg": uitleg
    }

def print_resultaat(res, is_motd, toon_extra=False):
    p_h, p_d, p_a = res["genormaliseerd"]
    lam_h, lam_a = res["lambda"]
    
    print(f"\n{BOLD}📊  GEANALYSEERDE GEGEVENS:{RESET}")
    print(f"  • Genormaliseerde kansen: Thuis: {p_h*100:.1f}% | Gelijk: {p_d*100:.1f}% | Uit: {p_a*100:.1f}%")
    print(f"  • Verwachte doelpunten (xG): Thuis: {lam_h:.2f} goals | Uit: {lam_a:.2f} goals")
    
    print(f"\n{GREEN}{BOLD}🏆  GEVISEERD ADVIES VOOR JOUW POULE:{RESET}")
    print(f"  • {BOLD}Voorspelde uitslag:{RESET} {GREEN}{BOLD}{res['uitslag']}{RESET}")
    
    if is_motd:
        print(f"  • {BOLD}Eerste doelpuntenmaker Thuis:{RESET} {YELLOW}{res['scorer_thuis']}{RESET}")
        print(f"  • {BOLD}Eerste doelpuntenmaker Uit:{RESET} {YELLOW}{res['scorer_uit']}{RESET}")
    
    print(f"\n{BOLD}💡  WAAROM DIT DE BESTE KEUZE IS:{RESET}")
    print(f"  {res['uitleg']}")
    
    if toon_extra:
        print(f"\n{BOLD}⏱️  TIE-BREAKER EXTRA VRAGEN (Wiskundige Mediaan):{RESET}")
        print(f"  • {BOLD}Minuut van het 1e toernooidoelpunt:{RESET} {YELLOW}31e minuut{RESET} (Mediaan EK/WK)")
        print(f"  • {BOLD}Minuut van de 1e gele kaart:{RESET} {YELLOW}36e minuut{RESET} (Asymmetrische scheidsrechter-bias)")
        print(f"  • {BOLD}Minuut van de 1e rode kaart:{RESET} {YELLOW}411e minuut{RESET} (Lange termijn toernooiklok)")
        print()
    else:
        print(f"\n{YELLOW}💡 Tip: Start het programma met de optie '--extra' of '-e' om de toernooi-strafvragen te zien.{RESET}\n")

def interactieve_modus(toon_extra=False):
    print_header()
    
    # Vraag type wedstrijd
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
            
    # Vraag kansen
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
    """Haalt actieve WK voetbalwedstrijden en hun kansen op van Polymarket."""
    url = "https://gamma-api.polymarket.com/events"
    params = {
        "tag_id": 100350,  # Voetbal tag
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
            
            # Filter specifiek op WK voetbal
            if not (slug.startswith("fifwc-") or "world-cup" in slug.lower() or "world cup" in title.lower()):
                continue
                
            markets = e.get("markets", [])
            
            # Splits teamnamen (bijv. "Ghana vs. Panama" of "EPL: Ghana vs. Panama")
            team_part = title
            if ":" in title:
                team_part = title.split(":")[-1].strip()
                
            teams = re.split(r'\s+vs\.?\s+', team_part, flags=re.IGNORECASE)
            if len(teams) != 2:
                continue
                
            home_team = teams[0].strip()
            away_team = teams[1].strip()
            
            home_prob = None
            draw_prob = None
            away_prob = None
            
            non_draw_markets = []
            for m in markets:
                q = m.get("question", "")
                prices_str = m.get("outcomePrices")
                if not prices_str:
                    continue
                    
                prices = json.loads(prices_str)
                if len(prices) < 1:
                    continue
                    
                yes_price = float(prices[0])
                
                if "draw" in q.lower():
                    draw_prob = yes_price
                else:
                    non_draw_markets.append((q, yes_price))
                    
            if len(non_draw_markets) == 2:
                home_words = set(re.findall(r'\w+', home_team.lower()))
                away_words = set(re.findall(r'\w+', away_team.lower()))
                
                m1_q, m1_p = non_draw_markets[0]
                m2_q, m2_p = non_draw_markets[1]
                
                m1_words = set(re.findall(r'\w+', m1_q.lower()))
                m2_words = set(re.findall(r'\w+', m2_q.lower()))
                
                m1_home_score = len(m1_words.intersection(home_words))
                m1_away_score = len(m1_words.intersection(away_words))
                
                m2_home_score = len(m2_words.intersection(home_words))
                m2_away_score = len(m2_words.intersection(away_words))
                
                score_A = m1_home_score + m2_away_score
                score_B = m2_home_score + m1_away_score
                
                if score_A >= score_B:
                    home_prob = m1_p
                    away_prob = m2_p
                else:
                    home_prob = m2_p
                    away_prob = m1_p
                    
            if home_prob is not None and draw_prob is not None and away_prob is not None:
                is_motd = is_motd_match(home_team, away_team)
                parsed_matches.append({
                    "title": team_part,
                    "home": home_team,
                    "away": away_team,
                    "home_prob": home_prob * 100.0,
                    "draw_prob": draw_prob * 100.0,
                    "away_prob": away_prob * 100.0,
                    "date": e.get("endDate", ""),
                    "is_motd": is_motd
                })
                
        # Sorteer chronologisch op datum/tijd voor een overzichtelijke lijst
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
            f.write(f"{'Datum/Tijd':<17} | {'Thuisploeg':<20} vs. {'Uitploeg':<20} | {'Odds (1/X/2)':<18} | {'Verw. Goals (xG)':<18} | {'Advies':<12} | {'Doelpuntenmaker Tips (MOTD)':<30}\n")
            f.write("-" * 134 + "\n")
            for m, res in alle_res:
                kansen_str = f"{m['home_prob']:.1f}% / {m['draw_prob']:.1f}% / {m['away_prob']:.1f}%"
                lam_h, lam_a = res["lambda"]
                xg_str = f"{lam_h:.2f} - {lam_a:.2f}"
                datum_str = m['date'].replace('T', ' ')[:16]
                
                advies_str = res["uitslag"]
                if m["is_motd"]:
                    advies_str += " [MOTD]"
                
                scorer_str = ""
                if m["is_motd"]:
                    thuis_tip = "Spits" if "spits" in res["scorer_thuis"].lower() else "Geen"
                    uit_tip = "Spits" if "spits" in res["scorer_uit"].lower() else "Geen"
                    scorer_str = f"Thuis: {thuis_tip} | Uit: {uit_tip}"
                    
                f.write(f"{datum_str:<17} | {m['home']:<20} vs. {m['away']:<20} | {kansen_str:<18} | {xg_str:<18} | {advies_str:<12} | {scorer_str:<30}\n")
            f.write("\n======================================================================================================================================\n")
            f.write("Gegenereerd door de Voetbalpoules Opta Voorspeller CLI.\n")
            
        print(f"{GREEN}✓ Voorspellingen succesvol opgeslagen in {BOLD}{bestandsnaam}{RESET}!\n")
    except Exception as e:
        print(f"{RED}❌ Fout bij opslaan van bestand: {e}{RESET}\n")

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
            datum_str = m['date'].replace('T', ' ')[:16]
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
                
                res = voorspel(m['home_prob'], m['draw_prob'], m['away_prob'], is_motd)
                print_resultaat(res, is_motd, toon_extra=toon_extra)
                break
                
            elif val == len(matches) + 1:
                # Voorspel ALLE wedstrijden
                print(f"\n{BOLD}Berekent voorspellingen voor alle {len(matches)} wedstrijden...{RESET}\n")
                
                alle_res = []
                for m in matches:
                    res = voorspel(m['home_prob'], m['draw_prob'], m['away_prob'], is_motd=m['is_motd'])
                    alle_res.append((m, res))
                    
                # Toon tabel
                print(f"{CYAN}{BOLD}======================================================================================================================================{RESET}")
                print(f"                                                    OVERZICHT ALLE VOORSPELDE WEDSTRIJDEN")
                print(f"{CYAN}{BOLD}======================================================================================================================================{RESET}")
                print(f"{BOLD}{'Datum/Tijd':<17} | {'Thuisploeg':<20} vs. {'Uitploeg':<20} | {'Odds (1/X/2)':<18} | {'xG (Thuis-Uit)':<15} | {'Uitslag':<12} | {'Doelpuntenmaker Tips (MOTD)':<30}{RESET}")
                print("-" * 134)
                for m, res in alle_res:
                    kansen_str = f"{m['home_prob']:.0f}% / {m['draw_prob']:.0f}% / {m['away_prob']:.0f}%"
                    lam_h, lam_a = res["lambda"]
                    xg_str = f"{lam_h:.2f} - {lam_a:.2f}"
                    datum_str = m['date'].replace('T', ' ')[:16]
                    
                    advies_str = res["uitslag"]
                    if m["is_motd"]:
                        advies_str += " [MOTD]"
                    
                    scorer_str = ""
                    if m["is_motd"]:
                        thuis_tip = "Spits" if "spits" in res["scorer_thuis"].lower() else "Geen"
                        uit_tip = "Spits" if "spits" in res["scorer_uit"].lower() else "Geen"
                        scorer_str = f"Thuis: {thuis_tip} | Uit: {uit_tip}"
                        
                    print(f"{datum_str:<17} | {m['home']:<20} vs. {m['away']:<20} | {kansen_str:<18} | {xg_str:<15} | {GREEN}{BOLD}{advies_str:<12}{RESET} | {YELLOW}{scorer_str:<30}{RESET}")
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
    
    args = parser.parse_args()
    
    # Als polymarket-modus is gekozen:
    if args.polymarket:
        if args.output:
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
                res = voorspel(m['home_prob'], m['draw_prob'], m['away_prob'], is_motd=m['is_motd'])
                alle_res.append((m, res))
                
            exporteer_naar_bestand(alle_res, args.output)
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
