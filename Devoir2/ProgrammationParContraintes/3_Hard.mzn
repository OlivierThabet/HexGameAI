% Vinet-Larouche Mattias (2212749)
% Thabet Olivier (2294559)

include "globals.mzn"; 

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre de villes à visiter + point de départ de la tournée
int: n;
 % Distance entre chaque villes (symmétrique)
array[1..n,1..n] of int: distance;
 % Distance minimale possible entre deux villes différentes
int: min_val = min([distance[i,j] | i,j in 1..n where distance[i,j] > 0]);
% Distance maximale possible entre deux villes différentes
int: max_val = max([distance[i,j] | i,j in 1..n]);

% Fenêtre de temps (1 = début, 2 = fin) durant laquelle il est possible de jouer pour chaque ville
array[1..n,1..2] of int: fenetre_arrivee;
% Temps maximum pour tout le trajet
int: max_trajet = max([fenetre_arrivee[i,2] | i in 1..n]);
% Fenêtre de temps la plus grande du trajet
int: max_fenetre = max([fenetre_arrivee[i,2] - fenetre_arrivee[i,1]| i in 2..n]);

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Ordre des villes visitées par la tournée (la dernière ville visitée doit être 1, pour revenir au point de départ).
array[1..n] of var 1..n: chemin_tournee;
% Circuit de la tournée, circuit_tournee[i] = j => j est le successeur de i
array[1..n] of var 1..n: circuit_tournee;
% Temps cumulé lors de l'arrivé à la ième ville de la tournée
array[1..n] of var 0..max_trajet: cumul_temps;
% Temps d'attente avant le spectacle en ville i
array[1..n] of var 0..max_fenetre: temps_attente;
% Temps d'attente total 
var 0..max_trajet: total_temps_attentes = sum(temps_attente);

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Temps de trajet à minimiser
var int: temps_trajet;

%-----------------------------------------------------------------------------%
% Étudiant
%-----------------------------------------------------------------------------%

% chemins
constraint chemin_tournee[n] = 1;
constraint alldifferent(chemin_tournee);
constraint circuit_tournee[chemin_tournee[n]] = chemin_tournee[1];
constraint forall(i in 1..n-1)(
    circuit_tournee[chemin_tournee[i]] = chemin_tournee[i+1]
);

% contraintes de distance pour boucler la boucle
constraint temps_attente[chemin_tournee[1]] = 
    max(0, fenetre_arrivee[chemin_tournee[1], 1] - distance[1, chemin_tournee[1]]);
constraint cumul_temps[1] = 
    distance[1, chemin_tournee[1]] + temps_attente[chemin_tournee[1]];
constraint distance[1, chemin_tournee[1]] <= fenetre_arrivee[chemin_tournee[1], 2];

% contraintes de distance sur les autres villes
constraint forall(i in 2..n)(
    % temps attente
    temps_attente[chemin_tournee[i]] = max(0,
        fenetre_arrivee[chemin_tournee[i], 1]
        - cumul_temps[i-1]
        - distance[chemin_tournee[i-1], chemin_tournee[i]])
    /\
    % temps cumul
    cumul_temps[i] = cumul_temps[i-1]
                   + distance[chemin_tournee[i-1], chemin_tournee[i]]
                   + temps_attente[chemin_tournee[i]]
    /\
    % fenetre sortie
    cumul_temps[i-1] + distance[chemin_tournee[i-1], chemin_tournee[i]]
        <= fenetre_arrivee[chemin_tournee[i], 2]
);

constraint temps_trajet = cumul_temps[n];
solve minimize temps_trajet;

%-----------------------------------------------------------------------------%

output [
  "Chemin de la tournée                   : \(chemin_tournee)\n",
  "Circuit de la tournée                  : \(circuit_tournee)\n",
  "Temps cumulés avant d'entrer en ville  : \(cumul_temps)\n",
  "Temps d'attente avant l'entrée en ville: \(temps_attente)\n",
  "Temps d'attente total                  : \(total_temps_attentes)\n",
  "Temps de trajet total                  : \(temps_trajet)\n",
];