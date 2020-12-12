# Projet application - Traitement de vidéos

![Supported Python Versions](https://img.shields.io/badge/Python->=3.8-blue.svg?logo=python&logoColor=white) ![Made withJupyter](https://img.shields.io/badge/Jupyter-6.1.5-orange.svg?logo=jupyter&logoColor=white)

_Auteurs:_ [Simon Audrix](mailto:saudrix@ensc.fr) & [Gabriel Nativel-Fontaine](mailto:gnativ910e@ensc.fr)

Ce dépôt contient notre travail dans le cadre du module **Projets applicatifs - Vidéo** du parcours **Intelligence Artificielle** inscrit dans la 3ème année du cursus d'ingénieur au sein de l'[Ecole Nationale Supérieure de Cognitique](http://www.ensc.fr).

## Partie 1: Classification d'images

Dans un premier temps, il nous a été demandé de classifier des images à l'aide d'un réseau de neurones.

Ces images sont des images de tailles 227 par 227 en couleurs, représentant des objets du quotidien pouvant apparaître dans une cuisine. Les images sont des images rognées pour ne contenir que l'objet que l'on souhaite classifier, réalisées à partir du jeu de données _Grasping In The Wild (GITW)_.

Dans cette première partie, on compte 5 classes:

- Bol
- Paquet de sucre
- Canette de Coca Cola
- Bouteille de lait
- Paquet de riz

<p align="center" display="flex">
   <img src='img/lab_bol.png' width=19% />
   <img src='img/lab_sucre.png' width=19% />
   <img src='img/lab_coca.png' width=19%/>
   <img src='img/lab_milk.png' width=19%/>
   <img src='img/lab_taureau_aile.png' width=19% />
</p>

### Pré-traitement des données

Comme avant tout traitement et analyse sur des données, il est nécessaire d'appliquer un prétraitement garantissant le même résultat pour toutes les images, améliorant ainsi la généralisation à des images n'appartenant pas au jeu de données initial (changements d'endroits, de luminosité, ...).

Le jeu de données contient 4736 images pour le set d'entrainement et 3568 images pour le set de test. C'est peu par rapport à ce que l'on sait des réseaux neurones. Il peut donc être nécessaire de faire de l'augmentation sur notre jeu de données, en appliquant des traitement comme des rotations des miroirs, ... on peut entrainer notre réseau sur plus d'images tout en améliorant encore ses performances de généralisation.

### Réseau custom

Le code de ce réseau est disponible dans un jupyter notebook:  [notebooks/nn_custom.ipynb](https://github.com/3a-ia-ensc/Video/blob/main/notebooks/nn_custom.ipynb).

Nous avons commencé par créer un réseau _"from scratch"_ , un simple réseau de convolutions. 

<p align="center">
   <img src='img/nn_custom.png' />
</p>

#### Résultats



### Réseaux pré-entraînés

Le code de ce réseau est disponible dans un jupyter notebook:  [notebooks/nn_vgg.ipynb](https://github.com/3a-ia-ensc/Video/blob/main/notebooks/nn_vgg.ipynb).

Nous avons comparé les résultats obtenus avec un réseau créé à partir d'un réseau déjà existant, ainsi nous avons utilisé l'architecture du réseau VGG pré-entraîné sur la base de données ImageNet.

<p align="center">
   <img src='img/nn_custom.png' />
</p>

#### Résultats

