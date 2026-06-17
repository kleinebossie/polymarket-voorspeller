# AI Agent Prompts — sequentieel uitvoeren

Elke prompt is zelfstandig en gaat ervan uit dat **alle eerdere prompts al zijn afgerond**. Voer ze in volgorde uit.

---

## Prompt 1 — Backtest-framework

```
## Context
Repo: opta-voorspeller — een voetbalpoule-tool die Polymarket 1X2-kansen omzet naar
Poisson/Dixon-Coles λ's en de score met hoogste Expected Value (EV) adviseert.
Alle logica zit in `polymarket_voorspeller.py`. Er is nog geen validatie tegen echte uitslagen.

## Taak
Bouw een backtest/evaluatielaag die meet of de EV-strategie daadwerkelijk goede poulepunten oplevert.

## Vereisten
1. Maak een nieuw bestand `backtest.py` (of `tests/backtest.py`) dat:
   - Historische wedstrijden kan inlezen (CSV/JSON) met minimaal: thuis, uit, uitslag (h-a), optioneel MOTD-flag
   - Per wedstrijd ook de 1X2-kansen nodig heeft (uit CSV of via gearchiveerde Polymarket-data)
   - `voorspel()` aanroept en de geadviseerde score vergelijkt met de werkelijke uitslag
   - Punten berekent volgens de bestaande regels in `calc_ev_regular()` en `calc_ev_motd()` (niet dupliceren — hergebruik die functies)

2. Rapporteer per strategie:
   - Gemiddelde behaalde punten per wedstrijd
   - Hit rate: exact / toto / thuisdoelpunten / uitdoelpunten
   - Vergelijking met baselines:
     a) Altijd 1-0 thuisfavoriet (hoogste 1X2-kans)
     b) Modus van de score-matrix (meest waarschijnlijke score)
     c) Huidige EV-optimalisatie

3. Voeg een voorbeeld-dataset toe (`data/backtest_voorbeeld.csv`) met 5–10 wedstrijden
   (WK 2022/2024 of fictief) zodat het script direct draait.

4. CLI: `python backtest.py --data data/backtest_voorbeeld.csv`

## Acceptatiecriteria
- Script draait zonder fouten op de voorbeelddata
- Output toont duidelijk EV-strategie vs. baselines
- Geen wijzigingen aan de kernlogica van `voorspel()` behalve wat nodig is voor import/hergebruik
- Korte README-sectie in docstring of comment bovenaan `backtest.py`

## Niet doen
- Geen wijzigingen aan het fit-algoritme of optimizer (dat komt in latere prompts)
- Geen commits tenzij expliciet gevraagd
```

---

## Prompt 2 — Log-loss i.p.v. MSE voor lambda-fitting

```
## Context
Prompt 1 (backtest) is af. Het backtest-framework in `backtest.py` werkt.
In `polymarket_voorspeller.py` gebruikt `bepaal_poisson_lambdas()` kwadratische fout
op 1X2-kansen. Dat is suboptimaal voor probabiliteiten.

## Taak
Vervang de MSE-loss in `bepaal_poisson_lambdas()` door cross-entropy (log-loss) voor 1X2,
en pas dezelfde loss toe op O/U-termen.

## Vereisten
1. Nieuwe loss voor 1X2:
   `-sum(target_i * log(model_i))` met kleine epsilon tegen log(0)
2. O/U-termen: idem log-loss op under/over, behoud de bestaande gewichten (0.5 / 0.8) voorlopig
3. Voeg een CLI-flag `--loss mse|logloss` toe (default: `logloss`) zodat backtest beide kan vergelijken
4. Draai `backtest.py` op de voorbeelddata en documenteer in een kort comment of log-loss beter/slechter/gelijk presteert

## Acceptatiecriteria
- `python polymarket_voorspeller.py -p` produceert nog steeds geldige voorspellingen
- Geen regressie in Polymarket-batchmodus
- Backtest kan MSE vs log-loss vergelijken via de flag
- Embedded JS in `exporteer_naar_html()` wordt NIET aangepast (dat is prompt 14)

## Bestanden
- `polymarket_voorspeller.py` (objective in `bepaal_poisson_lambdas`, argparse)
```

---

## Prompt 3 — Overround-verwijdering (Shin of power method)

```
## Context
Prompts 1–2 zijn af. Backtest werkt, log-loss is de default loss-functie.
`normaliseer_kansen()` deelt nu simpelweg door de som — dat negeert bookmaker-marge.

## Taak
Implementeer een betere methode om implied probabilities uit Polymarket-prijzen te halen,
vóór ze naar `bepaal_poisson_lambdas()` gaan.

## Vereisten
1. Implementeer minimaal de **power method** (of Shin's method) als alternatief voor lineaire normalisatie
2. Pas toe op:
   - 1X2-kansen (thuis/gelijk/uit)
   - O/U-kansen (under/over per lijn)
3. CLI-flag: `--overround linear|power` (default: `power`)
4. Backtest: vergelijk linear vs power op de voorbeelddata; rapporteer verschil in gemiddelde punten

## Acceptatiecriteria
- Power-method probabilities sommeren tot 1.0
- Geen negatieve of >1 kansen na normalisatie
- Bestaande Polymarket-parsing in `haal_polymarket_wedstrijden()` gebruikt de nieuwe normalisatie
- Handmatige invoer (`-i`, `-t/-g/-u`) gebruikt dezelfde normalisatie

## Referentie
Zoek naar `normaliseer_kansen` en alle plekken waar `u/tot, o/tot` wordt berekend in `haal_polymarket_wedstrijden()`.
```

---

## Prompt 4 — Optimizer upgraden (multi-start + bounded optimizer)

```
## Context
Prompts 1–3 zijn af. Log-loss + power-method overround zijn actief.
`bepaal_poisson_lambdas()` gebruikt Nelder-Mead met vaste start `[1.3, 1.0, -0.05]`
en clipped parameters binnen de objective — dat is fragiel.

## Taak
Vervang de optimizer door een robuustere aanpak met expliciete bounds en multi-start.

## Vereisten
1. Gebruik `scipy.optimize.minimize` met `method='L-BFGS-B'` en bounds:
   - λ_h, λ_a: [0.05, 5.0]
   - ρ: [-0.25, 0.10]
2. Multi-start met minimaal 5 startpunten:
   - Huidige default [1.3, 1.0, -0.05]
   - Maher-achtige schatting uit 1X2-targets (implementeer een eenvoudige analytische start)
   - 3 random starts binnen bounds
3. Kies de oplossing met laagste loss
4. Verwijder clipping binnen `objective()` — bounds worden door de optimizer afgedwongen
5. Log optioneel de fit-residual (1X2 + O/U) per wedstrijd bij `-v` / `--verbose`

## Acceptatiecriteria
- Fit-residual daalt gemiddeld t.o.v. oude Nelder-Mead (meet via backtest of log residual)
- Geen wedstrijden waar optimizer faalt zonder fallback (fallback: beste multi-start resultaat)
- `voorspellingen.txt`-output blijft consistent (λ's mogen verschuiven, dat is verwacht)

## Niet doen
- JS-calculator nog niet aanpassen
```

---

## Prompt 5 — Data-gedreven O/U-gewichten

```
## Context
Prompts 1–4 zijn af. Optimizer is L-BFGS-B met multi-start.
O/U-gewichten in `bepaal_poisson_lambdas()` zijn hardcoded: 0.5 (wedstrijd) en 0.8 (team).

## Taak
Maak de O/U-gewichten configureerbaar en optimaliseer ze via backtest.

## Vereisten
1. Extraheer gewichten naar constanten of een config-dict bovenaan het bestand:
   `WEIGHT_MATCH_OU = 0.5`, `WEIGHT_TEAM_OU = 0.8`
2. Breid `backtest.py` uit met een grid-search modus:
   `python backtest.py --grid-search-weights`
   Zoekt over een redelijk raster (bijv. match_ou ∈ [0.2, 0.5, 0.8, 1.0], team_ou ∈ [0.5, 0.8, 1.0, 1.2])
3. Rapporteer beste combinatie op basis van gemiddelde poulepunten
4. Update defaults naar de beste gevonden waarden (documenteer in comment waarom)

## Acceptatiecriteria
- Grid-search output is een tabel met gewichten → gemiddelde punten
- Defaults zijn geüpdatet op basis van backtest-resultaten
- CLI-flags `--weight-match-ou` en `--weight-team-ou` om handmatig te overriden

## Aanname
Als de voorbeelddata te klein is voor betrouwbare grid-search, breid `data/backtest_voorbeeld.csv`
uit met minimaal 20 wedstrijden (WK 2022 groepsfase is publiek beschikbaar).
```

---

## Prompt 6 — Slimmere afhandeling van meerdere O/U-lijnen

```
## Context
Prompts 1–5 zijn af. O/U-gewichten zijn geoptimaliseerd via backtest.
`haal_polymarket_wedstrijden()` verzamelt alle O/U-lijnen en stopt ze allemaal in de loss,
wat conflicten kan geven.

## Taak
Selecteer en weeg O/U-lijnen slimmer in plaats van ze blind op te tellen.

## Vereisten
1. Per type (wedstrijd-O/U, team-O/U thuis, team-O/U uit): selecteer maximaal 1 lijn
2. Selectiecriteria (in volgorde):
   a) Beide kanten (under + over) beschikbaar
   b) Hoogste gecombineerde liquiditeit (als Polymarket volume/liquidity beschikbaar is in de API-response)
   c) Anders: dichtst bij verwacht totaal doelpunten (bijv. lijn 2.5 als λ_h + λ_a ≈ 2.5)
3. Als meerdere lijnen toch gebruikt worden, weeg ze omgekeerd evenredig met spread/volatility
4. Log bij `--verbose` welke lijnen geselecteerd zijn en welke genegeerd

## Acceptatiecriteria
- Geen dubbele/conflicterende O/U-lijnen meer in `bepaal_poisson_lambdas()` input
- ρ-variatie in output neemt toe (niet meer bijna altijd -0.05) — meet op huidige Polymarket-batch
- Backtest-punten ≥ resultaat vóór deze wijziging

## Bestanden
- `haal_polymarket_wedstrijden()` — parsing en selectie
- `bepaal_poisson_lambdas()` — accepteert al geselecteerde lijnen
```

---

## Prompt 7 — Dynamische EV-zoekruimte

```
## Context
Prompts 1–6 zijn af. Lambda-fitting is geoptimaliseerd.
`voorspel()` zoekt EV over scores 0-0 t/m 6-6, terwijl de matrix 0-9 loopt.
Bij hoge λ (bv. Brazilië–Haïti, λ ≈ 3.8) worden scores als 4-0 gemist.

## Taak
Maak de EV-zoekruimte dynamisch op basis van de berekende λ's.

## Vereisten
1. Bereken `max_score = min(9, ceil(max(λ_h, λ_a) + 2))`
2. Zoek EV over `range(max_score + 1)` voor beide teams
3. Ondergrens blijft 0
4. Als cumulatieve matrix-massa buiten het raster > 1% is, verhoog `max_score` met 1 (max 9)
5. Toon in verbose-modus het gekozen raster en de buiten-raster-massa

## Acceptatiecriteria
- Wedstrijden met λ > 2.5 kunnen adviezen als 3-0, 4-0 geven waar dat EV-optimaal is
- Lage-λ-wedstrijden gedragen zich identiek aan voorheen
- Backtest op hoge-scoring wedstrijden toont verbetering of gelijk resultaat
- Performance: brute-force over max 10×10 blijft acceptabel (< 1ms per wedstrijd)

## Bestanden
- `voorspel()` in `polymarket_voorspeller.py`
```

---

## Prompt 8 — Risico-inzicht en top-N EV-weergave

```
## Context
Prompts 1–7 zijn af. Dynamisch raster werkt.
`voorspel()` berekent al `top_5` maar toont die nauwelijks in CLI/export.

## Taak
Verrijk de output met risico-inzicht naast maximale EV.

## Vereisten
1. Bereken per top-N score (default N=5):
   - EV (bestaand)
   - P(exacte score) — kans uit matrix
   - P(≥1 punt), P(≥5 punten) — cumulatief over alle matrix-uitkomsten
   - EV-marge t.o.v. #2 (ΔEV)

2. Toon in CLI (`print_resultaat`) een compacte top-5 tabel:
   Uitslag | EV | Kans% | P(≥5pt) | ΔEV

3. Voeg toe aan `exporteer_naar_bestand()` als optionele sectie onderaan (of `--top5` flag)

4. In `exporteer_naar_html()`: toon top-5 in een inklapbare sectie per wedstrijdkaart

## Acceptatiecriteria
- Bij twijfelgevallen (gelijkwaardige odds) is zichtbaar dat EV-scores dicht bij elkaar liggen
- Geen breaking changes in bestaande export-formaat (nieuwe sectie is additief)
- `top_5` dict-structuur in `voorspel()` return wordt uitgebreid, niet herschreven
```

---

## Prompt 9 — Tie-break bij gelijke EV

```
## Context
Prompts 1–8 zijn af. Top-5 EV-tabel met risico-inzicht is zichtbaar.
`voorspel()` kiest nu de eerste score bij `ev > best_ev` (strikt groter).

## Taak
Implementeer een deterministische tie-break wanneer meerdere scores dezelfde EV hebben.

## Vereisten
1. Tie-break volgorde (in deze prioriteit):
   a) Hoogste P(exacte score) — kies de waarschijnlijkere score
   b) Bij gelijk: hoogste P(≥5 punten) — meer partiële puntenkans
   c) Bij gelijk: laagste som doelpunten (conservatiever, bv. 1-0 boven 2-1)
   d) Bij gelijk: laagste thuisdoelpunten

2. Maak tie-break strategie configureerbaar via `--tiebreak probability|conservative` (default: `probability`)

3. Documenteer in code-comment waarom deze volgorde gekozen is (verwijzing naar poule-scoring)

## Acceptatiecriteria
- Geen willekeurige score-keuze meer bij gelijke EV (test met synthetische matrix)
- Backtest-punten ≥ vorige implementatie
- Tie-break keuze zichtbaar in verbose output ("Gekozen via tie-break: probability")
```

---

## Prompt 10 — MOTD-scorerfactor kalibreren

```
## Context
Prompts 1–9 zijn af. EV-optimalisatie en tie-breaks zijn stabiel.
In `calc_ev_motd()` wordt `4 * 0.35` gebruikt als vaste kans op de juiste doelpuntenmaker
als het team scoort. Die 0.35 is een onbewezen aanname.

## Taak
Kalibreer de scorer-factor op basis van data en maak hem configureerbaar.

## Vereisten
1. Extraheer `SCORER_HIT_RATE = 0.35` als constante
2. Onderzoek (web of hardcoded tabel) historische MOTD-scorer-hit-rates voor WK/EK:
   - P(juiste scorer | team scoort minstens 1 doelpunt)
   - P(juiste "geen score" | team scoort 0)
3. Als backtest-data MOTD-scorer-resultaten bevat, optimaliseer de factor via grid-search
4. CLI: `--scorer-rate 0.35` om te overriden
5. Toon in MOTD-output de gebruikte factor en verwachte scorer-EV apart van score-EV

## Acceptatiecriteria
- Factor is gedocumenteerd met bron of backtest-resultaat
- `calc_ev_motd()` gebruikt de configureerbare constante
- MOTD-backtest (als data beschikbaar) toont verbetering t.o.v. vaste 0.35
```

---

## Prompt 11 — Scorer-tip loskoppelen van score-voorspelling

```
## Context
Prompt 10 is af. Scorer-hit-rate is gekalibreerd en configureerbaar.
Nu volgt de MOTD-scorer-tip automatisch uit `pred_h > 0` / `pred_a > 0`,
maar "geen score" kan hogere EV hebben terwijl je toch 1-0 voorspelt.

## Taak
Optimaliseer de scorer-tip (spits vs. geen score) onafhankelijk van de score-voorspelling.

## Vereisten
1. Na het kiezen van de optimale score (h, a), voer een aparte optimalisatie uit:
   - Voor thuis: vergelijk EV-contributie van "spits" vs "geen score" bij vaste (h, a)
   - Idem voor uit
2. Of: brute-force over (score, scorer_h, scorer_a) als performance het toelaat
   (max 7×7×2×2 = 196 combinaties — prima)
3. Update `voorspel()` return dict met aparte velden:
   - `scorer_ev_thuis`, `scorer_ev_uit`
   - `scorer_tip_onafhankelijk: bool` (True als tip afwijkt van `pred_h > 0` logica)

4. Toon in output wanneer de tip afwijkt van de simpele regel, met uitleg

## Acceptatiecriteria
- Scenario "1-0 + geen score uit" wordt correct geadviseerd als dat EV-optimaal is
- Bestaande MOTD-wedstrijden in `voorspellingen.txt` worden herberekend; noteer welke tips veranderen
- Backtest MOTD-punten ≥ vorige implementatie
```

---

## Prompt 12 — Extra Polymarket-markten benutten

```
## Context
Prompts 1–11 zijn af. Het model gebruikt 1X2 + O/U. Polymarket biedt mogelijk meer markten.

## Taak
Breid `haal_polymarket_wedstrijden()` uit om extra markttypen te parsen en in de lambda-fit te gebruiken.

## Vereisten
1. Onderzoek de Polymarket API-response voor WK-events (tag_id=100350) en identificeer:
   - Both Teams To Score (BTTS)
   - Win to nil / clean sheet
   - Eventueel correct score markets
2. Parse deze markten met dezelfde robuustheid als bestaande O/U-parsing
3. Voeg fit-termen toe in `bepaal_poisson_lambdas()`:
   - BTTS: P(h>0 AND a>0) vs markt
   - Clean sheet thuis: P(a=0) vs markt
   - Clean sheet uit: P(h=0) vs markt
4. Gewichten configureerbaar (default: 0.6), optimaliseer via backtest als data beschikbaar
5. `--verbose`: toon welke extra markten gevonden en gebruikt zijn per wedstrijd

## Acceptatiecriteria
- Minstens BTTS wordt geparsed en gebruikt als de markt beschikbaar is
- Wedstrijden zonder extra markten werken ongewijzigd (graceful degradation)
- Fit-residual daalt gemiddeld op wedstrijden met BTTS-data
- Backtest-punten ≥ vorige implementatie

## Niet doen
- JS-calculator nog niet aanpassen
```

---

## Prompt 13 — Tie-breaker-vragen modelleren

```
## Context
Prompts 1–12 zijn af. Het Poisson-model is rijk gefit.
De `--extra` tie-breaker output is nu statisch (31e min, 36e min, 411e min).

## Taak
Maak tie-breaker-voorspellingen afgeleid uit het model in plaats van hardcoded constanten.

## Vereisten
1. **Eerste doelpunt-minuut:**
   - Modelleer als exponential verdeeld over 90 minuten met rate λ_total = λ_h + λ_a
   - Mediaan-minuut = 90 * (1 - ln(2)) / λ_total, afgerond naar geheel getal
   - Clamp tussen 1 en 90

2. **Eerste gele kaart:**
   - Gebruik een eenvoudig basismodel: mediaan ~30-40 minuten, schaal met λ_total
   - Of: constante uit historisch WK-gemiddelde (documenteer bron)
   - Maak configureerbaar

3. **Eerste rode kaart:**
   - Zeer laag probability event; adviseer "geen rode kaart" (minuut 0 of ">90") als default
   - Of mediaan uit historische data (~300+ minuten equivalent)

4. Update `print_resultaat(toon_extra=True)` met berekende waarden + korte uitleg
5. Voeg optioneel toe aan export

## Acceptatiecriteria
- `--extra` output varieert per wedstrijd op basis van λ
- Hoge-xG-wedstrijden → vroegere eerste-doelpunt-minuut
- Lage-xG-wedstrijden → latere minuut
- Geen externe API nodig
```

---

## Prompt 14 — Python/JS-pariteit herstellen

```
## Context
Prompts 1–13 zijn af. Alle Python-verbeteringen (log-loss, power overround, L-BFGS-B,
O/U-selectie, extra markten) zijn actief in `polymarket_voorspeller.py`.
De embedded JS-calculator in `exporteer_naar_html()` is verouderd:
- Gebruikt MSE i.p.v. log-loss
- Mist wedstrijd-level O/U
- Mist extra markten (BTTS etc.)
- Gebruikt eigen Nelder-Mead i.p.v. L-BFGS-B

## Taak
Synchroniseer de JavaScript-calculator met de Python-implementatie.

## Vereisten
1. Port de volgende Python-logica naar JS:
   - Log-loss objective
   - Power-method overround
   - L-BFGS-B of verbeterde Nelder-Mead met multi-start (L-BFGS-B in JS is optioneel; multi-start Nelder-Mead is acceptabel als het binnen 100ms blijft)
   - Dynamische EV-zoekruimte
   - Tie-break logica
   - MOTD-scorer optimalisatie (prompt 11)
2. Voeg een pariteits-test toe: `test_js_parity.py` dat 10 representatieve invoer-sets door
   Python en een headless JS-runner stuurt en output vergelijkt (λ, ρ, advies, EV binnen 1e-4)
3. Of: genereer de JS-math automatisch vanuit Python (template) om drift te voorkomen

## Acceptatiecriteria
- Python en JS geven identieke adviezen voor dezelfde invoer (binnen tolerantie)
- Browser-calculator ondersteunt dezelfde markt-inputs als CLI
- `index.html` werkt in browser zonder console-errors
- Pariteits-test slaagt in CI (voeg stap toe aan `.github/workflows/update_predictions.yml` of aparte test-workflow)

## Bestanden
- `polymarket_voorspeller.py` (exporteer_naar_html sectie)
- `index.html` (gegenereerd)
- Nieuw: `test_js_parity.py`
```

---

## Prompt 15 — MOTD-detectie robuuster maken

```
## Context
Prompts 1–14 zijn af. Python en JS zijn gesynchroniseerd.
MOTD-wedstrijden worden gedetecteerd via hardcoded `MOTD_LIST` met fuzzy word-matching,
wat fout kan gaan (bv. "Japan" in meerdere sets).

## Taak
Maak MOTD-detectie betrouwbaarder en onderhoudbaarder.

## Vereisten
1. Vervang fuzzy word-matching door expliciete team-paren:
   `MOTD_PAIRS = [("England", "Croatia"), ("Netherlands", "Sweden"), ...]`
2. Normaliseer teamnamen via een mapping-dict (NL ↔ EN namen):
   `"Nederland" → "Netherlands"`, `"Kroatie" → "Croatia"`, etc.
3. Laad MOTD-lijst uit een extern bestand `data/motd.json` (makkelijk bij te werken per speelronde)
4. CLI: `--motd-file data/motd.json` en `--motd` voor handmatige override
5. Log waarschuwing als een MOTD-pair niet gevonden wordt in de Polymarket-batch

## Acceptatiecriteria
- Geen false positives door ambigue woorden ("Japan", "United")
- MOTD-lijst updaten vereist geen code-wijziging, alleen `data/motd.json`
- Bestaande MOTD-wedstrijden in huidige `voorspellingen.txt` worden correct herkend
- GitHub Actions workflow blijft werken (commit `data/motd.json`)
```

---

## Prompt 16 — Polymarket API-parsing hardenen

```
## Context
Prompts 1–15 zijn af. MOTD-detectie is robuust.
Team/markt-matching in `haal_polymarket_wedstrijden()` gebruikt regex + woordoverlap,
wat fragiel is bij afwijkende teamnamen of API-wijzigingen.

## Taak
Maak de Polymarket-data-pipeline robuuster en beter observeerbaar.

## Vereisten
1. Centraliseer teamnaam-normalisatie in `normalize_team_name(name) -> str`
   met een mapping voor alle WK-2026 teams (thuis + uit varianten)
2. Vervang word-overlap matching door:
   - Exacte match na normalisatie (primair)
   - Fuzzy match via `difflib.SequenceMatcher` met threshold 0.8 (fallback)
3. Valideer geparste data:
   - 1X2-kansen sommeren tot ~100% na overround-removal
   - λ_h, λ_a binnen [0.05, 5.0] na fit
   - Waarschuwing als fit-residual > drempel
4. Schrijf parsing-resultaten naar `data/parse_log.json` bij `--verbose` of altijd in CI
5. Voeg retry-logica toe voor API-calls (3 pogingen, exponential backoff)

## Acceptatiecriteria
- Geen enkele wedstrijd in de huidige batch mist 1X2-data door parsing-fouten
- Parse-log toont 0 warnings op huidige Polymarket-batch
- API-timeout faalt gracefully met duidelijke foutmelding
- Bestaande output (`voorspellingen.txt`) blijft consistent of verbetert (meer O/U ✓ markers)
```

---

## Prompt 17 — Backtest uitbreiden met echte WK-data

```
## Context
Prompts 1–16 zijn af. Het volledige algoritme is geoptimaliseerd.
De backtest draait op een kleine voorbeelddataset.

## Taak
Breid de backtest uit met een echte historische dataset en genereer een evaluatierapport.

## Vereisten
1. Verzamel data voor WK 2022 groepsfase (48 wedstrijden):
   - Werkelijke uitslagen (publiek beschikbaar)
   - 1X2-closing odds (bv. via football-data.co.uk CSV of handmatig ingevulde Polymarket-archief)
2. Sla op als `data/wk2022_groepsfase.csv`
3. Draai volledige backtest met alle geoptimaliseerde defaults
4. Genereer rapport `reports/backtest_wk2022.md` met:
   - EV-strategie vs. baselines (tabel)
   - Per scoring-type hit rates
   - Top-10 wedstrijden waar EV-strategie het meest/minst verschil maakte
   - Aanbevelingen voor verdere verbetering
5. CLI: `python backtest.py --data data/wk2022_groepsfase.csv --report reports/backtest_wk2022.md`

## Acceptatiecriteria
- Rapport is leesbaar en bevat concrete cijfers
- EV-strategie presteert meetbaar beter dan minstens 2 van 3 baselines
- Als EV-strategie slechter presteert: rapporteer eerlijk en stel rollback-voorstellen voor
```

---

## Prompt 18 — Optioneel: overdispersie-model (alleen bij bewezen meerwaarde)

```
## Context
Prompt 17 is af. Backtest op WK 2022 toont waar het huidige Poisson/Dixon-Coles-model
tekortschiet (als dat zo is). Voer deze prompt ALLEEN uit als het rapport structurele
tekortkomingen toont bij hoge-scoring of gelijkwaardige wedstrijden.

## Taak
Implementeer een Negatieve Binomiaal-variant als alternatief scoremodel en vergelijk via backtest.

## Vereisten
1. Voeg `calc_matrix_nb(lam_h, lam_a, rho, r)` toe met overdispersie-parameter r
2. Fit r als 4e parameter in `bepaal_poisson_lambdas()` (of aparte functie)
3. CLI: `--model poisson|negbinom` (default: `poisson`)
4. Backtest vergelijking op `data/wk2022_groepsfase.csv`
5. Alleen als default switchen als negbinom ≥0.1 punt/wedstrijd beter scoort

## Acceptatiecriteria
- Negbinom model produceert geldige probability-matrices (som = 1.0)
- Backtest-rapport vergelijkt beide modellen eerlijk
- Geen complexiteitstoename in de default flow als negbinom niet beter presteert
- JS-pariteit (prompt 14) wordt bijgewerkt als negbinom default wordt

## Niet doen
- Geen bivariate Poisson tenzij negbinom ook tekortschiet
- Geen ML/neural nets
```

---

## Prompt 19 — Codebase opschonen en modulair maken

```
## Context
Alle algoritme-prompts (1–18) zijn af. `polymarket_voorspeller.py` is ~2500+ regels
en mengt model, API, CLI, HTML-template en JS.

## Taak
Splits de monolith op in logische modules zonder gedragswijziging.

## Voorgestelde structuur
opta-voorspeller/
├── model/
│   ├── poisson.py        # poisson, dixon_coles, calc_matrix, bepaal_lambdas
│   ├── ev.py             # calc_ev_regular, calc_ev_motd, voorspel
│   └── overround.py      # normaliseer_kansen, power method
├── data/
│   ├── polymarket.py     # haal_polymarket_wedstrijden, team normalization
│   └── motd.json
├── export/
│   ├── text.py           # exporteer_naar_bestand
│   └── html.py           # exporteer_naar_html + JS template
├── backtest.py
├── test_js_parity.py
├── polymarket_voorspeller.py  # dunne CLI-entrypoint
└── data/wk2022_groepsfase.csv

## Vereisten
1. Refactor zonder gedragswijziging — alle tests en backtests moeten identieke output geven
2. `python polymarket_voorspeller.py -p` blijft werken als entrypoint
3. GitHub Actions workflow hoeft niet te wijzigen (zelfde commando)
4. Voeg `requirements.txt` toe met `requests`, `scipy`

## Acceptatiecriteria
- `backtest.py` en pariteits-test slagen
- `python polymarket_voorspeller.py -p -o voorspellingen.txt` produceert byte-identieke output
  (of verschil alleen in timestamps als die in output staan)
- Geen circular imports
- Elke module < 400 regels
```

---

## Overzicht

| # | Prompt | Afhankelijk van |
|---|--------|-----------------|
| 1 | Backtest-framework | — |
| 2 | Log-loss | 1 |
| 3 | Overround power method | 1–2 |
| 4 | Optimizer multi-start | 1–3 |
| 5 | O/U-gewichten grid-search | 1–4 |
| 6 | O/U-lijn selectie | 1–5 |
| 7 | Dynamisch EV-raster | 1–6 |
| 8 | Risico-inzicht / top-5 | 1–7 |
| 9 | Tie-break gelijke EV | 1–8 |
| 10 | MOTD-scorer kalibratie | 1–9 |
| 11 | Scorer-tip loskoppelen | 10 |
| 12 | Extra Polymarket-markten | 1–11 |
| 13 | Tie-breaker-vragen model | 1–12 |
| 14 | Python/JS-pariteit | 1–13 |
| 15 | MOTD-detectie | 1–14 |
| 16 | API-parsing hardenen | 1–15 |
| 17 | Backtest WK 2022 data | 1–16 |
| 18 | Negatieve binomiaal (optioneel) | 17 |
| 19 | Modulaire refactor | 1–18 |

**Tip:** Voer prompt 14 (JS-pariteit) pas uit nadat alle Python-algoritmewijzigingen stabiel zijn — anders moet je de JS twee keer porten. Prompt 19 is het beste als allerlaatste stap.
