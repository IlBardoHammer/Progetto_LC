import sys
import codecs
import nltk
import math
from nltk import bigrams

def CalcolaTOT(frasi):
    lunghezzaTOT = 0.0
    lunghezzaFRASI = len(frasi)
    for i in frasi:
    #divido la frase in token
        tokens = nltk.word_tokenize(i)
        lunghezzaTOT = lunghezzaTOT + len(tokens)
    #restituisco le due lunghezze
    return lunghezzaFRASI, lunghezzaTOT

def CalcolaLunghezzaMEDIA(frasi):
    lunghezzaMEDIA_Frasi = 0.0
    lunghezzaMEDIA_Parole = 0.0
    somma_Frasi = 0.0
    somma_Parole = 0.0
    for i in frasi:
        #divido la frase in tokens
        tokens = nltk.word_tokenize(i)
        #sommo la lunghezza delle frasi in termini di tokens
        somma_Frasi = somma_Frasi + len(tokens)
        for j in tokens:
            #calcolo la lunghezza della parola
            parola = len(j)
            somma_Parole = somma_Parole + parola
            C = somma_Frasi
            lunghezzaMEDIA_Parole = somma_Parole/C
            lunghezzaMEDIA_Frasi = somma_Frasi/len(frasi)
    #restituisco le lunghezze medie
    return lunghezzaMEDIA_Frasi, lunghezzaMEDIA_Parole

def GrandezzaVoc(frasi):
    Corpus = []
    jnum = 0
    CMille = []
    for i in frasi:
        #divido la frase in tokens
        tokens = nltk.word_tokenize(i)
        Corpus = Corpus + tokens
    for j in Corpus:
        #uso un contatore per contare ogni 1000 tokens
        jnum+=1
        #aggiungo parole a una variabile di appoggio
        CMille.append(j)
        if(jnum==1000 or jnum%1000==0):
        #stampo il vocabolario ogni 1000 tokens
           print(len(set(CMille)))
           #azzero il contenuto della variabile di appoggio
           CMille = []
           #ritorno una stringa vuota per non far stampare un "None" a fine funzione
    return ""


def DistribuzioneHapax(frasi):
    Hapax = []
    Corpus = []
    jnum = 0
    for i in frasi:
        #divido la frase in tokens
        tokens = nltk.word_tokenize(i)
        Corpus = Corpus + tokens
    for j in Corpus:
         jnum+=1
         Hapax.append(j)
         if(jnum==1000 or jnum%1000==0):
           #stampo la distribuzione degli hapax ogni 1000 tokens
           print(len(Hapax)/jnum)
           #azzero il contenuto della variabile di appoggio
           Hapax = []
    #ritorno una stringa vuota per non far stampare un "None" a fine funzione
    return ""

def RapportoSostantiviVerbi(frasi):
    Sostantivi = []
    Verbi = []
    tokensPOS = []
    for i in frasi:
        #divido la frase in tokens
        tokens = nltk.word_tokenize(i)
        #assegno a una variabile "tokensPOS" una lista di tuple
        tokensPOS = nltk.pos_tag(tokens)
        for j in tokensPOS:
        #riempo le liste (Sostantivi,Verbi) rispettivamente con sostantivi e verbi
            if(j[1] == "NN" or j[1] == "NNS" or j[1] == "NNP" or j[1] == "NNPS"):
                Sostantivi.append(j[0])
            if(j[1] == "VB" or j[1] == "VBD" or j[1] =="VBG" or j[1] == "VBN" or j[1] == "VBP" or j[1] == "VBZ"):
                Verbi.append(j[0])
    #Siccome il primo return contine un possibile /0 controllo che ci siano Verbi e se non ci sono vado nel secondo return
    if(len(Verbi)!=0):
       return len(Sostantivi)/len(Verbi)
    else:
       return "Nessun Verbo trovato"

def PosFreq(frasi):
    tokensPOS = []
    posfreq = []
    listaPOS = []
    TestoTokenizzato = []
    DizionariolistaPOS = {}
    for i in frasi:
        #divido la frase in tokens
        tokens = nltk.word_tokenize(i)
        #prendo in input i tokens di ogni frase e riempo la lista con una somma di liste di tuple
        tokensPOS = nltk.pos_tag(tokens)
        for j in tokensPOS:
            #controllo che i tokens non siano caratteri speciali, come la punteggitura e non li inserisco nella lista dei POS-tag
            if(j[1]!=j[0]):
              listaPOS.append(j[1])
    #uso un doppio for per scorrere la listaPOS e controllare gli elementi al suo interno, in modo da trovare i POS-tag più frequenti
    listaPOSset = set(listaPOS)
    for i in listaPOSset:
        DizionariolistaPOS[i] = listaPOS.count(i)
    DizionariolistaPOS = sorted(DizionariolistaPOS.items(), key = lambda x : x[1], reverse=True)
    ListaOutput = list(DizionariolistaPOS)
    ListaOutput0 = map(lambda x: x[0], ListaOutput)
    ListaOutput0 = list(ListaOutput0)
    return ListaOutput0[0:10]

def BigrammiPOSPMAX(frasi):
    TestoTokenizzato = []
    DieciBigramma = {}
    for i in frasi:
        #divido la frase in tokens
        tokens = nltk.word_tokenize(i)
        TestoTokenizzato = TestoTokenizzato + tokens
    #aggiungo i bigrammi dal Testo tokenizzato a una variabile
    bigrammi = list(bigrams(TestoTokenizzato))
    #Salvo in un'altra variabile tutti i bigrammi diversi, quindi utilizzerò un data-type set
    bigrammiDiversi = set(bigrammi)
    for j in bigrammiDiversi:
        #calcolo la frequenza del bigramma
        freqBigramma = bigrammi.count(j)
        #calcolo la frequenza della parola
        freqA = TestoTokenizzato.count(j[0])
        #calcolo la probabilità condizionata
        probCond = freqBigramma*1.0/freqA*1.0
        #creo un dizionario bigramma/probabilità
        DieciBigramma[j] = probCond
    #ordino il dizionario per valori decrescenti di probabilità    
    DieciBigramma = sorted(DieciBigramma.items(), key = lambda x : x[1], reverse=True)
    return DieciBigramma[0:10]

def BigrammiPOSLMI(frasi):
    TestoTokenizzato = []
    DieciBigramma = {}
    for i in frasi:
        #divido la frase in tokens
        tokens = (nltk.word_tokenize(i))
        TestoTokenizzato = TestoTokenizzato + tokens
    #aggiungo i bigrammi dal Testo tokenizzato a una variabile
    bigrammi = list(bigrams(TestoTokenizzato))
    #Salvo in un'altra variabile tutti i bigrammi diversi, quindi utilizzerò un data-type set
    bigrammiDiversi = set(bigrammi)
    for j in bigrammiDiversi:
        #calcolo la frequenza di un'altro bigramma
        freqBigramma = bigrammi.count(j)
        #calcolo la frequenza della parola
        freqA = TestoTokenizzato.count(j[0])
        #calcolo la probabilità condizionata
        probCond = freqBigramma*1.0/freqA*1.0
        #calcolo la LMI
        LMI = freqBigramma * math.log((probCond/(freqBigramma/len(TestoTokenizzato)*(freqA/len(TestoTokenizzato)))), 2)
        #creo un dizionario bigramma/LMI
        DieciBigramma[j] = LMI
    #ordino il dizionario per valori decrescenti di LMI
    DieciBigramma = sorted(DieciBigramma.items(), key = lambda x : x[1], reverse=True)
    return DieciBigramma[0:10]

def main(file1, file2):
    fileInput1 = codecs.open(file1, "r", "utf-8")
    fileInput2 = codecs.open(file2, "r", "utf-8")
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    lingua_inglese = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi1 = lingua_inglese.tokenize(raw1)
    frasi2 = lingua_inglese.tokenize(raw2)
    print("confronto dei due testi")
    print("\n\nPrimo file: il numero totale di frasi e di token")
    print(CalcolaTOT(frasi1))
    print("\nSecondo file: il numero totale di frasi e di token")
    print(CalcolaTOT(frasi2))
    print("\n\nPrimo File: Calcolo la lunghezza media della frasi in termini di token e la lunghezza media delle parole in termini di caratteri")
    print(CalcolaLunghezzaMEDIA(frasi1))
    print("\nSecondo File: Calcolo la lunghezza media della frasi in termini di token e la lunghezza media delle parole in termini di caratteri")
    print(CalcolaLunghezzaMEDIA(frasi2))
    print("\n\nPrimo File: Calcolo la grandezza del vocabolario all'aumentare del corpus per porzioni incrementali di 1000 token (1000 token, 2000 token, 3000 token, etc.)")
    print(GrandezzaVoc(frasi1))
    print("\nSecondo File: Calcolo la grandezza del vocabolario all'aumentare del corpus per porzioni incrementali di 1000 token (1000 token, 2000 token, 3000 token, etc.)")
    print(GrandezzaVoc(frasi2))
    print("\n\nPrimo File: Calcolo la distribuzione degli hapax all'aumentare del corpus per porzioni incrementali di 1000 token (1000 token, 2000 token, 3000 token, etc.)")
    print(DistribuzioneHapax(frasi1))
    print("\nSecondo File: Calcolo la distribuzione degli hapax all'aumentare del corpus per porzioni incrementali di 1000 token (1000 token, 2000 token, 3000 token, etc.)")
    print(DistribuzioneHapax(frasi2))
    print("\n\nPrimo file: il rapporto tra Sostantivi e Verbi")
    print(RapportoSostantiviVerbi(frasi1))
    print("\nSecondo file: il rapporto tra Sostantivi e Verbi")
    print(RapportoSostantiviVerbi(frasi2))
    print("\n\nPrimo file: Calcolo le 10 PoS (Part-of-Speech) più frequenti")
    print(PosFreq(frasi1))
    print("\nSecondo file: Calcolo le 10 PoS (Part-of-Speech) più frequenti")
    print(PosFreq(frasi2))
    print("\n\nPrimo file: Estraggo ed ordino i 10 bigrammi di PoS con probabilità condizionata massima, indicando anche la relativa probabilità")
    print(BigrammiPOSPMAX(frasi1))
    print("\nSecondo File: Estraggo ed ordino i 10 bigrammi di PoS con probabilità condizionata massima, indicando anche la relativa probabilità")
    print(BigrammiPOSPMAX(frasi2))
    print("\n\nPrimo file: Estraggo ed ordino i 10 bigrammi di PoS con forza associativa massima calcolata in termini di Local Mutual Information indicando anche la relativa forza associativa.")
    print(BigrammiPOSLMI(frasi1))
    print("\nSecondo file: Estraggo ed ordino i 10 bigrammi di PoS con forza associativa massima calcolata in termini di Local Mutual Information indicando anche la relativa forza associativa.")
    print(BigrammiPOSLMI(frasi2))
    print("\n\n\nFINE ESECUZIONE")
main(sys.argv[1], sys.argv[2])
