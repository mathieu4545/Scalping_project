//+------------------------------------------------------------------+
//|                                              ExportDataCsv.mq5  |
//|                        Copyright 2024, Your Name                  |
//|                                       https://www.yourwebsite.com|
//+------------------------------------------------------------------+
#property strict

// Nom du fichier CSV
string fileName = "ExportData.csv";
int fileHandle;

// Fonction d'initialisation
int OnInit()
  {
   // Vérifier si le fichier existe déjà
   bool fileExists = FileIsExist(fileName);

   // Ouvrir le fichier en mode écriture
   fileHandle = FileOpen(fileName, FILE_WRITE | FILE_CSV);
   if (fileHandle == INVALID_HANDLE)
     {
      Print("Erreur d'ouverture du fichier: ", GetLastError());
      return INIT_FAILED;
     }

   // Écrire les en-têtes si le fichier est nouveau
   if (!fileExists || FileSize(fileHandle) == 0)
     {
      FileWrite(fileHandle, "Time,Open,High,Low,Close,Volume");
     }

   // Fermer le fichier
   FileClose(fileHandle);

   // Démarrer le timer
   EventSetTimer(1); // Déclenche toutes les secondes

   return INIT_SUCCEEDED;
  }

// Fonction appelée à chaque tick
void OnTick()
  {
   // Obtenir les prix en temps réel
   datetime time = TimeCurrent();
   double open = iOpen(NULL, 0, 0);
   double high = iHigh(NULL, 0, 0);
   double low = iLow(NULL, 0, 0);
   double close = iClose(NULL, 0, 0);
   long volume = iVolume(NULL, 0, 0);

   // Réouvrir le fichier en mode ajout
   fileHandle = FileOpen(fileName, FILE_WRITE | FILE_CSV);
   if (fileHandle != INVALID_HANDLE)
     {
      // Ajouter les données
      FileWrite(fileHandle, TimeToString(time, TIME_DATE | TIME_MINUTES), open, high, low, close, volume);
      FileClose(fileHandle);
     }
   else
     {
      Print("Erreur d'ouverture du fichier: ", GetLastError());
     }
  }

// Fonction de nettoyage
void OnDeinit(const int reason)
  {
   // Arrêter le timer
   EventKillTimer();
  }
