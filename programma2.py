import sys
import codecs
import re
import nltk

def AnalisiLinguistica(frasi):
    tokensTOT = []
    ListaNomi = []
    tokensPOStot = []
    ListaLuoghi = []
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        analisi = nltk.ne_chunk(tokensPOS)
        for nodo in analisi: #ciclo l'albero scorrendo i nodi
            NE = ''
            NE2 = ''
            if hasattr(nodo, 'label'): #controlla se chunk è un nodo intermedio, uso label perchè uso NLTK3
                if nodo.label() in ["PERSON"]:
                    for partNE in nodo.leaves():
                        NE = NE+' '+partNE[0]
                    ListaNomi.append(NE) #Creo la Lista dei Nomi
                if nodo.label() in ["GPE"]:
                    for partNE2 in nodo.leaves():
                        NE2 = NE2+' '+partNE2[0]
                    ListaLuoghi.append(NE2) #Creo la Lista dei Luoghi
        tokensTOT = tokensTOT+tokens
        tokensPOStot = tokensPOStot+tokensPOS

    return tokensTOT, tokensPOStot, ListaNomi, ListaLuoghi

def EstraiDieciNomi(ListaNomiInput, FrasiInput):
    NomiSet = set(ListaNomiInput)
    DizionarioNomiOutput = {}
    DieciNomiOutput = []
    for i in NomiSet: #creo un dizionario Nomi/frequenze
        DizionarioNomiOutput[i] = ListaNomiInput.count(i)
    DizionarioNomiOutput = sorted(DizionarioNomiOutput.items(), key = lambda x : x[1], reverse=True)
    #ordino il dizionario per valori decrescenti di frequenza,
    #poi creo una lista di tuple dal dizionario e
    #poi mappo la lista di tuple per ottenere una lista di Dieci Nomi
    ListaDizionario = list(DizionarioNomiOutput)
    ListaApp1 = map(lambda x: x[0], ListaDizionario)
    ListaApp2 = list(ListaApp1)
    DieciNomiOutput = ListaApp2[0:10]
    return DieciNomiOutput

def CalcolaListaFrasi(Nome, FrasiInput):
    FrasiOutput = []
    #Rimuovo lo spazio davanti a ogni Nome
    Nome2 = Nome[1:len(Nome)]
    for i in FrasiInput:
        tokens = nltk.word_tokenize(i)
        for j in tokens:
            #confronto frase per frase ogni tokens con il Nome passato da Input e se il nome è presente nella frase, aggiungo la frase alla lista in output
           if(j==Nome2):
            FrasiOutput.append(i)
            break
    return FrasiOutput

def CalcolaLunghezzaFrasi(ListaFrasiNome):
    DizionarioFrasi = {}
    DizionarioFrasi1 = {}
    DizCopy = {}
    FraseMax = []
    FraseMin = []
    listApp = []
    listaApp1 = []
    for i in ListaFrasiNome:
        #creo un dizionario Frasi/lunghezza
        DizionarioFrasi[i] = len(i)
    if(DizionarioFrasi!={}):
       DizCopy = DizionarioFrasi
       #ordino due dizionari uno in ordine crescente, l'altro in ordine decrescente
       DizionarioFrasi = sorted(DizionarioFrasi.items(), key = lambda x : x[1], reverse=True)
       DizionarioFrasi1 = sorted(DizCopy.items(), key = lambda x : x[1])
       #creo due liste di tuple dai dizionari e trovo FraseMax e FraseMin come primi elementi delle due liste
       listaApp = list(DizionarioFrasi)
       FraseMax = listaApp[0][0]
       listaApp1 = list(DizionarioFrasi1)
       FraseMin = listaApp1[0][0]
    return "Restituisco la frase più lunga", FraseMax, "Restituisco la frase più breve", FraseMin

def CalcolaDieciLuoghi(ListaFrasiNome, Luoghi):
    LuoghiSet = set(Luoghi)
    DizionarioLuoghiOutput = {}
    DieciLuoghiOutput = []
    ListaDizionario = []
    tokenstot = []
    for j in ListaFrasiNome:
        tokens = nltk.word_tokenize(j)
        tokenstot = tokenstot + tokens
    for i in LuoghiSet:
        #se il luogo è presente almeno una volta lo aggiungo al Dizionario dei luoghi
        #Levo lo spazio davanti al Luogo
        i1= i[1:len(i)]
        if(tokenstot.count(i1)>0):
            #creo un dizionario luoghi/frequenza
          DizionarioLuoghiOutput[i] = tokenstot.count(i1)
    #ordino il dizionario per valori discendenti di frequenza
    DizionarioLuoghiOutput = sorted(DizionarioLuoghiOutput.items(), key = lambda x : x[1], reverse=True)
    return DizionarioLuoghiOutput[0:10]

def CalcolaDieciPerson(ListaFrasiNome, Person):
    PersonSet = set(Person)
    DizionarioPersonOutput = {}
    DieciPersonOutput = []
    tokenstot = []
    for j in ListaFrasiNome:
        tokens = nltk.word_tokenize(j)
        tokenstot = tokenstot + tokens
    for i in PersonSet:
        #se la persona è presente almeno una volta lo aggiungo al Dizionario delle persone
        #Levo lo spazio davanti al nome
        i1= i[1:len(i)]
        if(tokenstot.count(i1)>0):
            #creo un dizionario persona/frequenza
           DizionarioPersonOutput[i] = tokenstot.count(i1)
    #ordino il dizionario per valori discendenti di frequenza
    DizionarioPersonOutput = sorted(DizionarioPersonOutput.items(), key = lambda x : x[1], reverse=True)
    return DizionarioPersonOutput[0:10]

def CalcolaDieciSostantiviVerbi(ListaFrasiNome):
    DieciSostantivi = []
    DizionarioSostantivi = {}
    DizionarioVerbi = {}
    DieciVerbi = []
    tokensPOStot = []
    tokenstot = []
    for i in ListaFrasiNome:
        tokens = nltk.word_tokenize(i)
        tokensPOS = nltk.pos_tag(tokens)
        tokensPOStot = tokensPOStot + tokensPOS
        tokenstot = tokenstot + tokens
    for j in tokensPOStot:
        #controllo frase per frase nella lista delle tuple parola/POS-tag se sono presenti tag relativi a sostantivi o a verbi
        if(j[1] == "NN" or j[1] == "NNS" or j[1] == "NNP" or j[1] == "NNPS"):
           DizionarioSostantivi[j[0]] = tokenstot.count(j[0])
        if(j[1] == "VB" or j[1] == "VBD" or j[1] == "VBG" or j[1] == "VBN" or j[1] == "VBP" or j[1] == "VBZ"):
           DizionarioVerbi[j[0]] = tokenstot.count(j[0])
    #ordino i dizionari per valori decrescenti di frequenza
    DizionarioSostantivi = sorted(DizionarioSostantivi.items(), key = lambda x : x[1], reverse=True)
    DizionarioVerbi = sorted(DizionarioVerbi.items(), key = lambda x : x[1], reverse=True)
    print("Sostantivi più frequenti:")
    print(DizionarioSostantivi[0:10])
    print("Verbi più frequenti")
    print(DizionarioVerbi[0:10])
    return ""

def DateMesiGiorni(ListaFrasiNome):
    Date = []
    Mesi = []
    Giorni = []
    for i in ListaFrasiNome:
        #uso le espressioni regolari per trovare Date,Mesi e Giorni e trovo le date in formato numerico. Aggiungo a ogni giro del ciclo le date i mesi e i giorni trovati nella lista delle frasi del nome
        Date += (re.findall(r' \b\d\d[-/]\d\d[-/]\d\d\d\d\b', i))
        Mesi += (re.findall(r' \b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', i))
        Giorni += (re.findall(r'\b(?:Monday|Tuesday|Thursday|Wednesday|Friday|Saturday|Sunday)\b', i))
    return Date, Mesi, Giorni

def Markov(Corpus, Frasi, ListaFrasiNome):
    DistribuzioneFreq = nltk.FreqDist(Corpus)
    LunghezzaCorpus = len(Corpus)
    probabilita = 1.0
    DizionarioFrasi = {}
    DizionarioFrasiNome = {}
    ListaApp = []
    for j in Frasi:
          tokens = nltk.word_tokenize(j)
          for tok in tokens:
              #calcolo la probabilita
              probtoken = (DistribuzioneFreq[tok]*1.0/LunghezzaCorpus*1.0)
              probabilita = probabilita*probtoken
              #creo un dizionario frase/probabilita
              DizionarioFrasi[j] = probabilita
    #ordino il dizionario per valori decrescenti di probabilita, creo una lista di tuple dal dizionario e una lista di frasi dalla lista di tuple
    DizionarioFrasi = sorted(DizionarioFrasi.items(), key = lambda x : x[1], reverse=True)
    ListaApp = list(DizionarioFrasi)
    ListaApp1 = map(lambda x: x[0], ListaApp)
    ListaApp2 = list(ListaApp1)
    for k in ListaFrasiNome:
            tokens=nltk.word_tokenize(k)
            if(ListaApp2.count(k)>0 and len(tokens)>=8 and len(tokens)<=12):
                return k
            else:
                return "Nessuna frase trovata"

def main(file1, file2):
    fileInput1 = codecs.open(file1, "r","utf-8")
    fileInput2 = codecs.open(file2, "r","utf-8")
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)
    TestoTokenizzato1, TestoAnalizzatoPOS1, NomiPropri1, Luoghi1 = AnalisiLinguistica(frasi1)
    TestoTokenizzato2, TestoAnalizzatoPOS2, NomiPropri2, Luoghi2 = AnalisiLinguistica(frasi2)
    DieciNomiFreq1 = EstraiDieciNomi(NomiPropri1, frasi1)
    DieciNomiFreq2 = EstraiDieciNomi(NomiPropri2, frasi2)
    print("Confronto dei due corpora:")
    print("\n\nPRIMO CORPUS")
    for i in DieciNomiFreq1:
        print("\n\nNome:")
        print(i)
        ListaNome1 = CalcolaListaFrasi(i, frasi1)
        print()
        print(CalcolaLunghezzaFrasi(ListaNome1))
        print("\nDieci Luoghi più frequenti:")
        print(CalcolaDieciLuoghi(ListaNome1, Luoghi1))
        print("\nDieci Persone più frequenti:")
        print(CalcolaDieciPerson(ListaNome1, NomiPropri1))
        print("\nDieci Sostantivi e dieci verbi più frequenti:\n")
        CalcolaDieciSostantiviVerbi(ListaNome1)
        print("\nDate, Mesi, Giorni:\n")
        print(DateMesiGiorni(ListaNome1))
        print("\nCalcolo la frase lunga minimo 8 token e massimo 12 con probabilità più alta:\n")
        print(Markov(TestoTokenizzato1, frasi1, ListaNome1))
    print("\n\nSECONDO CORPUS")
    for j in DieciNomiFreq2:
        print("\n\nNome:\n")
        print(j)
        ListaNome2 = CalcolaListaFrasi(j, frasi2)
        print()
        print(CalcolaLunghezzaFrasi(ListaNome2))
        print("\nDieci Luoghi più frequenti:\n")
        print(CalcolaDieciLuoghi(ListaNome2, Luoghi2))
        print("\nDieci Persone più frequenti:\n")
        print(CalcolaDieciPerson(ListaNome2, NomiPropri2))
        print("\nDieci Sostantivi e dieci verbi più frequenti:\n")
        CalcolaDieciSostantiviVerbi(ListaNome2)
        print("\nDate, Mesi, Giorni:\n")
        print(DateMesiGiorni(ListaNome2))
        print("\nCalcolo la frase lunga minimo 8 token e massimo 12 con probabilità più alta:\n")
        print(Markov(TestoTokenizzato2, frasi2, ListaNome2))
    print("\n\n\nFINE ESECUZIONE")
main(sys.argv[1], sys.argv[2])
