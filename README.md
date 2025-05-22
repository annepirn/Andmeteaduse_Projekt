# Andmeteaduse_Projekt
Autorid: Anne Pirn, Tõnis Reinik, Mirjam Reino

## Eesmärk
Uurida, kuidas on seotud kinnisvara hinnad, inflatsioon, keskmine tööhõive ja keskmine palk. Seejärel koostada mudel, mis ennustaks (regressioonmudel), kuidas mõjutavad üksteist inflatsiooni, keskmise ruutmeetri hinna ja keskmise palga muutused. 
Samuti luuakse 3-osaline töölaud:
### 1. **Hetkeolukord**
- Kuvab viimase kvartali andmed (THI, brutopalk, eluasemehinnaindeks).
- Maakondade võrdlus kaardil: €/m² ruutmeetrihinnad hooneliikide lõikes.
- Tööjõuturu ülevaade (nt töötuse määr, leibkondade arv).

### 2. **Muutused ajas**
- Ajalooliste trendide visualiseerimine kvartalite kaupa.
- Võrdlus brutopalga ja eluasemeindeksi vahel.
- Maakondade korrelatsioonid valitud tunnuste vahel ajas.

### 3. **Prognoosid**
- Prophet ja lineaarne regressioon näitajate (THI, brutopalk, hinnaindeks) tulevikutrendide ennustamiseks.
- SARIMA vs. regressioon prognoos tööjõu- ja majandusnäitajatele.
- Maakondade hinnaennustused hooneliikide lõikes aastaks 2025.

## Hüpoteesid

1. Keskmine kinnisvara ruutmeetrihind liigub keskmise palgaga positiivses korrelatsioonis.
2. Keskmine palk on korrelatsioonis tööhõive ja inflatsiooniga.

### Kasutatud tehnoloogiad
- **[Streamlit](https://streamlit.io/)** – interaktiivne veebirakendus
- **Pandas / Numpy / GeoPandas** – andmetöötlus
- **Matplotlib / Seaborn / Plotly** – visualiseerimine
- **scikit-learn / statsmodels / Prophet** – mudelid ja prognoosimine
- **Mapbox** – interaktiivsed kaardid

### Nõuded
Enne jooksutamist, paigalda vajalikud sõltuvused:
pip install -r requirements.txt 

### Andmed

Kõik puhastatud ning töölaual kasutatud failid on toodud [Data andmekaustas](https://github.com/annepirn/Andmeteaduse_Projekt/tree/main/Data) . 

Andmete saaamiseks kasutati Eesti Statistikaameti API võimalus ning kasutati järgnevaid andmetabeleid:
Tarbijahinnaindeks: https://andmed.stat.ee/et/stat/majandus__hinnad/IA02

Tööhõive: https://andmed.stat.ee/et/stat/sotsiaalelu__tooturg__tooturu-uldandmed__luhiajastatistika/TT4660

Keskmine palk: https://andmed.stat.ee/et/stat/sotsiaalelu__sissetulek/ST14

Registreeritud töötud:  https://andmed.stat.ee/et/stat/sotsiaalelu__tooturg__tooturu-uldandmed__aastastatistika/TT4645

Kinnisvara hindade saamikseks laaditi maa-ameti kodulehelt alla hinnastatistika korteritele ning eluhoonetele, valiti aastad 2018-2024:
https://www.maaamet.ee/kinnisvara/htraru/

Fail: Kinnisvara-hinnastatistika-2018-2024.csv

Faili ülesehitus: 
Aasta{"2018","2019","2020","2021","2022","2023","2024",”2025”}, Kvartal{“I”,”II”,”III”,”IV”}, Hoone liik{“korter_10_29.99”,”korter_30_40.99”,”korter_41_54.99”,”korter_55_69.99”,”korter_70_249.99”,””Elamu”,”Suvila”,”Muu”}, näitajad {“Korteriomanditeginute_Arv”, “Keskmine_m2”,”Kokku_tehinguhind”,”Minimaalne_tehingusumma”,”Maksimaalne_tehingsumma”,”Minimaalne_pinnaühikuhind”,”Maksimaalne_pinnaühikuhind”,”Mediaan_pinnaühikuhind”,”Keskmine_pinnaühikuhind”,”Standardhälve_pinnaühikuhind”} 


### Andmete teisendused
Andmetes esinevad kuunimed teisendati eestikeelsele kujule ning kvartaalse info puhul tehti kindlaks, et oleks olemas veerg, kus on esindatud nii kvartali number kui aasta number. 
Kuud olid algtabelis eraldi veergudena, need teisendati üheks kuuveeruks.
Puuduvate väärtuste veerud asendati NaN-iga.
Filtreeriti välja ainult maakonnad, mis päriselt eksisteerivad. 

Iga muudatuse järel prinditi välja tulemused, et kontrollida, kas see samm läks õigesti. 


### Andmete visualiseerimine

Andmete visualiseerimiseks kasutati streamlit.app töövahendit. 
Näidatakse hetkeolukorda, muutuseid ajas ning prognoose. 

Muudatuste jaoks defineeriti käesolev ajaühik, eelmine ajaühik ning year to date. Seejärel arvutati muutus kvartalis tarbija hinnaindeksi, keskmise brotopalga ning eluaseme hinnaindeksi jaoks.  

Visualiseerimiseks maakonniti kasutati [maakonnad.geojson](https://github.com/annepirn/Andmeteaduse_Projekt/blob/main/maakonnad.geojson) faili. 

Samuti loodi aasta filter ning muudatustele illustreerimiseks pandi juurde nooremärgised ning värvid. 

Muudatuste kuvamiseks loodi algkvartlai filter ning lõpp-kvartali filter. Samuti loodi hoone liigi filter, millega saab valida, kas tegu on majaga ning korteri puhul suuruse. 

Juhul kui miskit ei kuvata, kuvab töölaud sõnumi "Mõned aastad jäeti välja" ning annab ka märku, mitu juhtumit välja arvestati.

Jooniste loomisel võetakse arvesse valitud filtreid. Joonisteks kasutatakse joondiagramme, et näidata, kas seos uuritud näitaja ja aja vahel on lineaarne ning täpsemaks vaateks kasutatakse Eesti kaarti, millele on lisatud heatmap. 


