% NOM_ETUDIANT_1 (MATRICULE_1)
% NOM_ETUDIANT_2 (MATRICULE_2)

include "globals.mzn";

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Nombre devant satisfaire les critères
var 1000..9999: nombre;

% Décomposition du nombre en un ensemble de chiffres
array[1..4] of var 1..9: chiffres;

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Variable à minimiser
var int: p;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
%1234
constraint chiffres[4] mod 2 = 0;
constraint forall(k in 1..4)(chiffres[k] != 0);
constraint all_different(chiffres);
constraint chiffres[1] > chiffres[2];
constraint chiffres[3] < chiffres[4];
constraint sum(i in 1..4)(chiffres[i])>15;
p = chiffres[2] * chiffres[3] * chiffres [4];

nombre = chiffres[4] + chiffres[3]*10 + chiffres[2]*100 + chiffres[1]*1000;

solve minimize p;

%-----------------------------------------------------------------------------%

output [
  show(nombre)
];