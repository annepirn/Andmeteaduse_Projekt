# Andmeteaduse_Projekt

Projekti eesmärk: Uurida, kuidas on seotud kinnisvara hinnad, inflatsioon, keskmine tööhõive ja keskmine palk. Seejärel koostada mudel, mis ennustaks (regressioonmudel), kuidas mõjutavad üksteist inflatsiooni, keskmise ruutmeetri hinna ja keskmise palga muutused.

Hüpoteesid:

1. Keskmine kinnisvara ruutmeetrihind liigub keskmise palgaga positiivses korrelatsioonis.
2. Keskmine palk on korrelatsioonis tööhõive ja inflatsiooniga.
3. Laenuportfelli kvaliteeti mõjutab kinnisvara hinnast, keskmisest palgast ja tööhõivest kõige rohkem tööhõive.

### Andmed

Andmete saaamiseks kasutati Eesti Statistikaameti API võimalus ning kasutati järgnevaid andmetabeleid:
Tarbijahinnaindeks: https://andmed.stat.ee/et/stat/majandus__hinnad/IA02

Tööhõive: https://andmed.stat.ee/et/stat/sotsiaalelu__tooturg__tooturu-uldandmed__luhiajastatistika/TT4660

Keskmine palk: https://andmed.stat.ee/et/stat/sotsiaalelu__sissetulek/ST14

Registreeritud töötud:  https://andmed.stat.ee/et/stat/sotsiaalelu__tooturg__tooturu-uldandmed__aastastatistika/TT4645

Kinnisvara hindade saamikseks laaditi maa-ameti kodulehelt alla hinnastatistika korteritele ning eluhoonetele:
https://www.maaamet.ee/kinnisvara/htraru/

Fail: Kinnisvara-hinnastatistika-2018-2024.csv

Faili ülesehitus: 
Aasta{"2018","2019","2020","2021","2022","2023","2024",”2025”}, Kvartal{“I”,”II”,”III”,”IV”}, Hoone liik{“korter_10_29.99”,”korter_30_40.99”,”korter_41_54.99”,”korter_55_69.99”,”korter_70_249.99”,””Elamu”,”Suvila”,”Muu”}, näitajad {“Korteriomanditeginute_Arv”, “Keskmine_m2”,”Kokku_tehinguhind”,”Minimaalne_tehingusumma”,”Maksimaalne_tehingsumma”,”Minimaalne_pinnaühikuhind”,”Maksimaalne_pinnaühikuhind”,”Mediaan_pinnaühikuhind”,”Keskmine_pinnaühikuhind”,”Standardhälve_pinnaühikuhind”} 



