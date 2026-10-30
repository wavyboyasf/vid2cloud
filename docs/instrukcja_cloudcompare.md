# Instrukcja: przygotowanie referencji TLS i dopasowania zgrubnego w CloudCompare

Do zrobienia ręcznie w GUI, raz: (a) TLS bez elementów ruchomych, (b) pary punktów dla
każdego nagrania, (c) macierz zgrubna. Reszta (wycinanie, SOR, ICP, C2C, statystyki) idzie
wsadowo: `bash eval/eval_all.sh`.

CloudCompare 2.13.2 (flatpak, ten sam co w trybie wsadowym). Menu podane po angielsku, tak jak
w domyślnym interfejsie. Ścieżki względem katalogu repozytorium.

**Zasada na całość:** niczego nie przesuwać, nie obracać, nie skalować i nie podpróbkowywać
w GUI. Pliki mają zostać w tym samym układzie współrzędnych, w którym zostały zapisane.
Jeśli przy otwieraniu pliku pojawi się okno **Global shift/scale**, odpowiedz **No** (współrzędne
TLS i rekonstrukcji są rzędu metrów, przesunięcie nie jest potrzebne, a zmieniłoby eksportowane
współrzędne punktów).

---

## a) TLS bez elementów ruchomych

TLS jest z 23.02.2026, nagrania z 21.09.2026 — krzesła, pulpity, instrumenty i przedmioty na
scenie mogły stać inaczej. Usuwamy je z referencji, żeby C2C nie mierzyło przestawionych mebli.

1. **File > Open** → `runs/2026-09-21/eval_cc/tls/tls_koncertowa_1cm.ply`.
2. W drzewie **DB Tree** zaznacz chmurę. Dla czytelności: w panelu **Properties** ustaw
   **Colors: RGB** (kolory ze skanera).
3. **Edit > Segment** (ikona nożyczek na pasku głównym, skrót **T**). Pojawia się mały pasek
   narzędzi segmentacji.
   - Spacja (albo przycisk **Pause segmentation**) przełącza między obracaniem widoku a rysowaniem.
   - W trybie rysowania: lewy przycisk dodaje wierzchołek wieloboku, prawy go zamyka.
   - **Segment Out** (ikona z minusem) ukrywa punkty wewnątrz wieloboku, **Segment In** — na zewnątrz.
   - Usuń krzesła, pulpity, statywy i wszystko, co da się przestawić. Ściany, podłoga, sufit,
     kolumny, framugi, kominki, stała zabudowa — zostają. Rysuj kilka wieloboków po kolei,
     obracając widok między nimi (najwygodniej z góry, a potem z boku — przyciski widoków
     standardowych, m.in. **Top view**, są na pionowym pasku *Viewing tools* po lewej stronie okna).
4. Zatwierdź zielonym **✓ (Confirm segmentation)**. CloudCompare dzieli chmurę na dwie: widoczną
   i ukrytą (w DB Tree pojawi się druga chmura z dopiskiem `.remaining` albo `.segmented`).
   Kliknij obie na przemian (pole wyboru przy nazwie), żeby sprawdzić, która jest bez mebli.
5. Zaznacz chmurę bez mebli → **File > Save** (Ctrl+S) → typ **PLY** → nazwa
   `runs/2026-09-21/eval_cc/tls/tls_koncertowa_1cm_static.ply` → format **Binary little endian**.
   Pola skalarne (intensywność) mogą zostać.
6. Pliku `tls_koncertowa_1cm.ply` nie nadpisuj — jest punktem wyjścia, gdyby trzeba było
   segmentować jeszcze raz.

`eval_cloud.py` domyślnie bierze wersję `_static`; jeśli jej nie ma, `eval_all.sh --dry-run`
pokazuje to jako brak.

---

## b) Pary punktów: rekonstrukcja ↔ TLS (raz na nagranie)

Jedna macierz na nagranie wystarcza dla wszystkich jego wariantów (adapter, powtórzenia,
przebiegi selekcji): zaczynają się od tej samej albo sąsiedniej klatki, więc mają prawie ten
sam układ (kotwica = pierwsza klatka kluczowa). ICP to doprecyzuje.

Nagrania i chmury do klikania:

| nazwa | chmura rekonstrukcji |
|---|---|
| `daleko_wolno` | `runs/2026-09-21/adapter/daleko_wolno/cloud.ply` |
| `blisko_z_gory` | `runs/2026-09-21/adapter/blisko_z_gory/cloud.ply` |
| `daleko_szybko_pion` | `runs/2026-09-21/adapter/daleko_szybko_pion/cloud.ply` |

Jakie punkty — **6 do 8 par**:
- naroża i krawędzie ścian, framugi drzwi i okien, narożniki kominków, podstawy i głowice
  kolumn/pilastrów, krawędź podestu — elementy stałe, ostre, łatwe do wskazania w obu chmurach;
- **nie plansze ArUco** (TLS ich nie ma — był robiony w lutym), nie krzesła ani pulpity;
- rozłożone po całym fragmencie sali, który widać w rekonstrukcji, i w trzech wymiarach:
  nie wszystkie na podłodze i nie wszystkie na jednej ścianie (punkty prawie współliniowe albo
  na jednej płaszczyźnie źle wyznaczają obrót).

Kroki dla jednego nagrania (np. `daleko_wolno`):

1. **File > Open** → chmura rekonstrukcji z tabeli i `tls_koncertowa_1cm_static.ply`. Obie
   chmury mają różne układy i skale — to normalne, widoki ustawiasz osobno (ukrywaj jedną
   polem wyboru w DB Tree).
2. Zapisz sobie (kartka, zrzut ekranu) listę 6–8 elementów w kolejności, w jakiej będziesz klikać.
3. Zaznacz **chmurę rekonstrukcji** w DB Tree → **Tools > Point list picking** (ikona listy
   punktów na pasku głównym).
4. Klikaj lewym przyciskiem punkty z listy, po kolei. Błędne kliknięcie usuwa ikona usuwania
   ostatniego punktu w oknie narzędzia.
5. W oknie narzędzia: ikona dyskietki (**Export**) → **Export to ASCII file** (dowolny wariant:
   `x,y,z`, z indeksem albo z etykietą — skrypt bierze trzy ostatnie pola) →
   `runs/2026-09-21/eval_cc/pairs/daleko_wolno_rec.txt`.
6. Zamknij narzędzie (**✓**). Zaznacz **chmurę TLS** → **Tools > Point list picking** →
   te same elementy **w tej samej kolejności** → eksport do
   `runs/2026-09-21/eval_cc/pairs/daleko_wolno_tls.txt`.

Pary są łączone wyłącznie po kolejności wierszy, więc kolejność w obu plikach musi być ta sama.
Katalog `runs/2026-09-21/eval_cc/pairs/` trzeba założyć, jeśli go nie ma.

---

## c) Macierz zgrubna

```bash
cd /home/wavy/dev/vid2cloud
/home/wavy/miniforge3/envs/vid2cloud-tools/bin/python eval/coarse_from_pairs.py daleko_wolno
```

Skrypt liczy przekształcenie przez podobieństwo (Umeyama, 7 parametrów: skala, obrót,
przesunięcie) i wypisuje skalę [m / jedn. chmury], odchyłkę na każdej parze i ich RMS.
Zapisuje:
- `runs/2026-09-21/eval_cc/coarse/daleko_wolno.txt` — macierz 4×4 (format `-APPLY_TRANS` CC),
- `runs/2026-09-21/eval_cc/coarse/daleko_wolno_report.json` — raport.

Jeśli jedna para odstaje wyraźnie od pozostałych, najczęstsze przyczyny to zamieniona
kolejność albo kliknięcie w inny element — popraw plik i uruchom ponownie. Macierz zgrubna ma
tylko ustawić chmury blisko siebie; dokładne dopasowanie robi ICP.

Uwaga: przy pierwszej ocenie nagrania `eval_cloud.py` zapisuje prostopadłościan wycinania
(`eval_cc/crop/<nazwa>.json`) razem z macierzą, z której go policzył. Jeśli później zmienisz
macierz zgrubną, usuń `eval_cc/crop/<nazwa>.json`, `<nazwa>_tls.ply` i `<nazwa>_tls.json`
(skrypt odmówi liczenia ze starym prostopadłościanem).

---

## d) Ocena wsadowa

```bash
bash eval/eval_all.sh --dry-run   # lista chmur i czego brakuje
bash eval/eval_all.sh             # liczy wszystko, co ma macierz zgrubną
```

Wyniki: `runs/2026-09-21/eval_cc/c2c/<źródło>/<…>/c2c_stats.json` i zbiorczo
`runs/2026-09-21/eval_cc/c2c_summary.csv`. Dokładne komendy CloudCompare każdej chmury są
w `cc_commands.txt` w jej katalogu.

Zrzuty z legendą barw: **File > Open** → `<katalog wyniku>/c2c.ply` → w **Properties** pole
skalarne `C2C absolute distances` jako aktywne → w sekcji skali barw zaznacz **Visible**
(legenda w rogu widoku). Do porównywania zrzutów ustaw ten sam zakres wyświetlania we wszystkich
(sekcja **SF display params** w panelu **Properties** — histogram z suwakami i polami liczbowymi).
