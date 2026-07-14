# TODO

- **Presentation-Notebook für die Abgabe**: `AUFGABENSTELLUNG.md` will Top-1/Top-5-Accuracy, Trainingskurven und generierte ABC-Tunes gezeigt bekommen, Prof mag Notebooks. `analysis.ipynb` ist aktuell nur für den Autoresearch-Loop selbst (val_bpb-Frontier über Experimente), nicht für die eigentliche Abgabe. Entweder erweitern oder neues Notebook.
- **Checkpoint-Saving in `train.py` fehlt komplett**: jeder 5-Minuten-Run ist wegwerfbar, nur der Code landet im Git-Commit, Gewichte sind danach weg. Für Musik-Samples/finale Metriken braucht's mindestens einen Checkpoint-Save für den "finalen" Lauf, sobald eine Architektur feststeht.
- **Notebook-getriebener Loop (inspiriert von Dimitris `notebook_runner.ipynb`), aber mit Claude statt Gemini**: falls gewünscht — Dimitris Version ruft die Gemini API direkt aus der Notebook-Zelle; müsste für Claude Code angepasst werden. Nicht spontan mitnehmen, erst klären ob/wie das gewünscht ist.
