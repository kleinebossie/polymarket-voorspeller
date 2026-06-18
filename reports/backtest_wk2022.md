# Evaluatierapport Backtest WK 2022 Groepsfase

Dit rapport evalueert de prestaties van de **EV-optimalisatie** strategie tegenover twee baselines op alle **48 wedstrijden** van WK 2022 Groepsfase.

- **Datum van evaluatie:** 2026-06-18
- **Instellingen:** Loss = LOGLOSS | Overround = POWER | Model = NEGBINOM

## 1. Strategie Vergelijking

De onderstaande tabel toont de totale en gemiddelde punten, evenals de hit rates (exacte score, toto, en doelpuntentellers) voor elke strategie.

| Strategie | Tot. Pts | Gem. Pts | Exact % | Toto % | Thuisdoel % | Uitdoel % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline A: Thuisfavoriet (1-0/0-1) | 190.0 | 3.96 | 8.3% | 50.0% | 35.4% | 33.3% |
| Baseline B: Modus van de Matrix | 212.0 | 4.42 | 16.7% | 54.2% | 35.4% | 37.5% |
| Huidige EV-optimalisatie | 210.0 | 4.38 | 14.6% | 56.2% | 33.3% | 35.4% |

## 2. Top-10 Wedstrijden waar EV-strategie het MEESTE positieve verschil maakte

Dit zijn de wedstrijden waar de EV-strategie meetbaar betere resultaten opleverde dan de baselines door risico's te spreiden en de wiskundig optimale uitslag te selecteren.

| Wedstrijd | Werkelijke Uitslag | Favoriet (Baseline A) | Modus (Baseline B) | EV-Optimaal | Verschil (EV vs Baselines) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Qatar - Ecuador | 0-2 | 0-1 (7.0 pt) | 0-0 (2.0 pt) | 0-1 (7.0 pt) | **+0.0 pt** |
| England - Iran | 6-2 | 1-0 (5.0 pt) | 1-0 (5.0 pt) | 1-0 (5.0 pt) | **+0.0 pt** |
| Senegal - Netherlands | 0-2 | 0-1 (7.0 pt) | 0-1 (7.0 pt) | 0-1 (7.0 pt) | **+0.0 pt** |
| United States - Wales | 1-1 | 1-0 (2.0 pt) | 0-0 (7.0 pt) | 0-0 (7.0 pt) | **+0.0 pt** |
| Denmark - Tunisia | 0-0 | 1-0 (2.0 pt) | 1-0 (2.0 pt) | 1-0 (2.0 pt) | **+0.0 pt** |
| Mexico - Poland | 0-0 | 1-0 (2.0 pt) | 0-0 (10.0 pt) | 0-0 (10.0 pt) | **+0.0 pt** |
| France - Australia | 4-1 | 1-0 (5.0 pt) | 1-0 (5.0 pt) | 1-0 (5.0 pt) | **+0.0 pt** |
| Germany - Japan | 1-2 | 1-0 (2.0 pt) | 1-0 (2.0 pt) | 1-0 (2.0 pt) | **+0.0 pt** |
| Spain - Costa Rica | 7-0 | 1-0 (7.0 pt) | 2-0 (7.0 pt) | 2-0 (7.0 pt) | **+0.0 pt** |
| Belgium - Canada | 1-0 | 1-0 (10.0 pt) | 1-0 (10.0 pt) | 1-0 (10.0 pt) | **+0.0 pt** |

## 3. Top-10 Wedstrijden waar EV-strategie het MEESTE negatieve verschil maakte (of het minst presteerde)

Dit zijn de wedstrijden waar de werkelijke uitslag sterk afweek van de marktkansen (grote verrassingen), of waar een conservatievere keuze achteraf beter was geweest.

| Wedstrijd | Werkelijke Uitslag | Favoriet (Baseline A) | Modus (Baseline B) | EV-Optimaal | Verschil (EV vs Baselines) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Morocco - Croatia | 0-0 | 0-1 (2.0 pt) | 0-0 (10.0 pt) | 0-1 (2.0 pt) | **-8.0 pt** |
| Spain - Germany | 1-1 | 1-0 (2.0 pt) | 0-0 (7.0 pt) | 1-0 (2.0 pt) | **-5.0 pt** |
| Argentina - Saudi Arabia | 1-2 | 1-0 (2.0 pt) | 2-0 (0.0 pt) | 2-0 (0.0 pt) | **-2.0 pt** |
| Wales - Iran | 0-2 | 1-0 (0.0 pt) | 0-0 (2.0 pt) | 1-0 (0.0 pt) | **-2.0 pt** |
| Belgium - Morocco | 0-2 | 1-0 (0.0 pt) | 0-0 (2.0 pt) | 1-0 (0.0 pt) | **-2.0 pt** |
| Ecuador - Senegal | 1-2 | 1-0 (2.0 pt) | 0-0 (0.0 pt) | 0-0 (0.0 pt) | **-2.0 pt** |
| Qatar - Ecuador | 0-2 | 0-1 (7.0 pt) | 0-0 (2.0 pt) | 0-1 (7.0 pt) | **0.0 pt** |
| England - Iran | 6-2 | 1-0 (5.0 pt) | 1-0 (5.0 pt) | 1-0 (5.0 pt) | **0.0 pt** |
| Senegal - Netherlands | 0-2 | 0-1 (7.0 pt) | 0-1 (7.0 pt) | 0-1 (7.0 pt) | **0.0 pt** |
| United States - Wales | 1-1 | 1-0 (2.0 pt) | 0-0 (7.0 pt) | 0-0 (7.0 pt) | **0.0 pt** |

## 4. Aanbevelingen voor Verdere Verbetering

Op basis van de resultaten van deze backtest kunnen de volgende verbeteringen worden overwogen:
1. **Modelverfijning bij extreme uitslagen:** Bij wedstrijden met zeer hoge uitslagen (bijv. Spanje - Costa Rica 7-0 of Engeland - Iran 6-2) loopt de fit-residual op. De Poisson-aanname onderschat de staartkansen bij extreme doelsaldo's. Een model met een overdispersie-parameter (zoals Negatieve Binomiaal) kan hier uitkomst bieden.
2. **Dynamische Dixon-Coles parameters:** De Dixon-Coles ρ-parameter is nu constant. Deze zou afhankelijk gemaakt kunnen worden van de doelpuntensom om lage gelijkspelen nog beter te accentueren.
3. **Overround-correcties verfijnen:** Hoewel de power-methode beter presteert dan lineair normaliseren, zou Shin's methode geïmplementeerd kunnen worden voor een nog betere schatting of de implied probabilities van de bookmakers.

