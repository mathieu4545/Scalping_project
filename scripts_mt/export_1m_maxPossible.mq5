//+------------------------------------------------------------------+
//|                                        ExportMultiPairData.mq5   |
//|                        Copyright 2024, Your Name                 |
//|                                       https://www.yourwebsite.com|
//+------------------------------------------------------------------+
#property strict

// Liste des symboles (20 paires de devises les plus tradées)
//string symbols[] = {"EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", 
//                    "EURJPY", "GBPJPY", "EURGBP", "EURCHF", "EURCAD", "EURAUD", "AUDJPY", 
//                   "AUDCAD", "CADJPY", "CHFJPY", "NZDJPY", "GBPCHF", "EURNZD"};

string symbols[] = {"AUDNZD"};

// Période de temps (5 minutes)
ENUM_TIMEFRAMES timeframe = PERIOD_M10;

// Fonction d'initialisation
int OnInit()
  {
   // Exporter les données pour chaque symbole
   for(int i = 0; i < ArraySize(symbols); i++)
     {
      ExportHistoricalData(symbols[i]);
     }

   // Fin du script
   Print("Export terminé avec succès.");
   return INIT_SUCCEEDED;
  }

// Fonction pour exporter les données historiques
void ExportHistoricalData(string symbol)
  {
   // Obtenir la date actuelle
   datetime currentTime = TimeCurrent();
   
   // Calculer la date 1 an dans le passé
   datetime startTime = currentTime - 100 * 24 * 3600; // 1 an

   // Allouer un tableau pour les données des bougies
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   // Copier les données des bougies
   int copied = CopyRates(symbol, timeframe, startTime, currentTime, rates);
   if (copied <= 0)
     {
      Print("Erreur de récupération des données pour ", symbol, ": ", GetLastError());
      return;
     }

   Print("Nombre de bougies copiées pour ", symbol, ": ", copied);
   
   // Nom du fichier CSV avec format symbol_timeframe_debut_to_fin.csv
   string fileName = symbol + "_" + EnumToString(timeframe) + "_" +
                     TimeToString(startTime, TIME_DATE) + "_to_" +
                     TimeToString(currentTime, TIME_DATE) + ".csv";

   // Ouvrir le fichier en mode écriture
   int fileHandle = FileOpen(fileName, FILE_WRITE | FILE_CSV);
   if (fileHandle == INVALID_HANDLE)
     {
      Print("Erreur d'ouverture du fichier pour ", symbol, ": ", GetLastError());
      return;
     }

   // Écrire les en-têtes
   FileWrite(fileHandle, "Time,Open,High,Low,Close");
   
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

   // Fermer le fichier
   FileClose(fileHandle);
   Print("Export terminé pour ", symbol, ". Fichier: ", fileName);
  }

// Fonction de nettoyage
void OnDeinit(const int reason)
  {
   // Rien à faire ici
  }
