//+------------------------------------------------------------------+
//|                                        ExportMinuteData.mq5    |
//|                        Copyright 2024, Your Name                 |
//|                                       https://www.yourwebsite.com|
//+------------------------------------------------------------------+
#property strict

// Nom du fichier CSV
string fileName = "ExportMinuteData.csv";
int fileHandle;

// Fonction d'initialisation
int OnInit()
  {
   // Ouvrir le fichier en mode écriture
   fileHandle = FileOpen(fileName, FILE_WRITE | FILE_CSV);
   if (fileHandle == INVALID_HANDLE)
     {
      Print("Erreur d'ouverture du fichier: ", GetLastError());
      return INIT_FAILED;
     }

   // Écrire les en-têtes
   FileWrite(fileHandle, "Time,Open,High,Low,Close,Volume");

   // Exporter les données
   ExportHistoricalData();

   // Fermer le fichier
   FileClose(fileHandle);

   // Fin du script
   Print("Export terminé avec succès.");
   return INIT_SUCCEEDED;
  }

// Fonction pour exporter les données historiques
void ExportHistoricalData()
  {
   // Obtenir le symbole actuel
   string symbol = Symbol();
   
   // Période en minutes
   ENUM_TIMEFRAMES timeframe = PERIOD_M5;

   // Obtenir la date actuelle
   datetime currentTime = TimeCurrent();
   
   // Calculer la date 1 an dans le passé
   datetime startTime = currentTime - 365 * 24 * 3600; // 1 an

   // Allouer un tableau pour les données des bougies
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   // Copier les données des bougies
   int copied = CopyRates(symbol, timeframe, startTime, currentTime, rates);
   if (copied <= 0)
     {
      Print("Erreur de récupération des données : ", GetLastError());
      return;
     }

   Print("Nombre de bougies copiées : ", copied);
   
   if (copied > 0)
     {
      // Écrire les données dans le fichier CSV
      for (int i = 0; i < copied; i++)
        {
         FileWrite(fileHandle, 
                   TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES),
                   rates[i].open,
                   rates[i].high,
                   rates[i].low,
                   rates[i].close);
        }
     }
  }

// Fonction de nettoyage
void OnDeinit(const int reason)
  {
   // Rien à faire ici
  }
