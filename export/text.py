#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tekst Export Module
-------------------
Functies om voorspellingen weg te schrijven naar platte tekstbestanden.
"""

import datetime
from zoneinfo import ZoneInfo

# ANSI Kleurcodes voor terminal-output
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"

def converteer_utc_naar_nl(utc_str):
    """
    Zet een UTC-tijdstip om naar de Nederlandse tijdzone en formatteert dit als leesbare tekst.
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

def exporteer_naar_bestand(alle_res, bestandsnaam, inclusief_top5=False):
    """
    Exporteert alle berekende uitslagen chronologisch naar een tekstbestand met xG, rho en MOTD scorer tips.
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
