% Vinet-Larouche Mattias (2212749)
% Thabet Olivier (2294559)

include "globals.mzn";

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre d'acteurs et de rôles
int: n;
% Couleurs des costumes
enum costumes; 
% Costume démandé pour un rôle i (première dimension) par un acteur j (seconde dimension)
array[1..n,1..n] of costumes: choix_costume;

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Variables de décision pour assigner un acteur à un rôle
array [1..n] of var 1..n: assignation_role;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

constraint all_different(assignation_role);
array[1..n] of var costumes: assignation_costumes;
constraint all_different(assignation_costumes);
constraint forall(i in 1..n)(assignation_costumes[i] = choix_costume[assignation_role[i], i]);
constraint forall(i in 1..(n-1))(abs(assignation_role[i]-assignation_role[i+1]) > 1);
%-----------------------------------------------------------------------------%

output	
  ["L'acteur " ++ format(i) ++ 
   " prendra le rôle " ++ format(fix(assignation_role[i])) ++ 
   " (costume " ++ format(choix_costume[fix(assignation_role[i]), i]) ++ ")" ++"\n" 
  | i in 1..n];