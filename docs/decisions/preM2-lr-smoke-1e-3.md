# Smoke lr=1e-3 : raison et plan de validation à 1e-5

Statut : DÉCISION CONSIGNÉE (pré-M2, audit correctif A7).
Référence : `SmokeConfig.lr` (`train/loop.py`), `MasterOptimizer`
(`train/master.py`).

## Pourquoi 1e-3 dans le smoke (`SmokeConfig.lr`, `train/loop.py`)

À `lr=1e-5` + paramètres bf16 purs, un update AdamW typique (~1e-5)
s'arrondit à zéro (ULP bf16 ~4e-3 à magnitude 1 ; mesuré : 0,0 % de
paramètres modifiés — test `test_master_accumulates_when_bf16_rounds`).
Le smoke doit OBSERVER des mises à jour (porte câblage) : `lr=1e-3` rend
les updates du modèle visibles en 5 pas. Ce n'est PAS une recommandation
d'entraînement : le pilote utilisera 1e-5.

## Ce qui rend 1e-5 viable (implémenté ici, `train/master.py`)

Poids maîtres FP32 (`MasterOptimizer`) : gradients bf16 → accumulation
FP32 → recopie. Testé : grads finis non nuls, modèle modifié à lr
visible, bit-exact reproductible, accumulation prouvée à 1e-5 même
quand la recopie bf16 arrondit.

## Plan de validation à 1e-5 (M2, pilote)

1. Smoke long (−N pas) à 1e-5 + maîtres : assert non plus sur
   l'empreinte modèle par pas (arrondi), mais sur l'empreinte MAÎTRES
   (strictement monotone) + dérive bornée modèle-vs-maîtres.
2. Comparer trajectoires de perte 1e-5+maîtres vs 1e-3 : même direction
   initiale sur 50 pas (corrélation des KL par ligne > 0.99), sinon
   investiguer (pas de conclusion qualité).
3. Passage en fp32 pur exclu (×2 mémoire modèle + ×4 états — hors budget
   0.8B+enseignant sur 34 Go avec activations).

## Audit correctif A7

Le smoke utilise `lr=1e-3` pour rendre les mises à jour OBSERVABLES
(5 pas, 16 prompts, L_max=128). Le test de reprise F-008 (A5) vérifie
que la reprise après interruption conserve l'état des maîtres FP32 et
reproduit le même chemin d'optimisation.
