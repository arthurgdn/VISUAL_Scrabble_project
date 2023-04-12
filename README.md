### Projet Computer Vision - Scrabble

## Introduction
Ce projet de Vision par Ordinateur consiste en la mise en place d'un assistant au Scrabble. En donnant en entrée, une grille de Scrabble et le tirage de lettre effectué, on renvoie le mot qui donne le score le plus important.

## Fine-tuning du modèle
Pour détecter correctemment les lettres sur le Scrabble nous avons besoin de fine-tuner le modèle. En effet, celui-ci a du mal à distinguer les "I". 

On lui donne ainsi en entrée les images I et I2. 

Puis on fine-tune le modèle avec la commande suivante
```python -m Vision.ViT_finetuning```

Une fois que l'entrainement a fini de tourner votre modèle s'enregistre dans le dossier modèle. 

## Obtenir le meilleur mot

Pour être assister au Scrabble, il vous faut ensuite exécuter la commande 
```python main.py``` 
On vous demande alors votre image. Indiquez par exemple "board1.jpeg". 

L'algorithme détecte alors la grille et vous en donne un aperçu. 

On vous demande alors les lettres que vous avez pioché. Entrez vos 7 lettres (par exemple : AEBNERT). 

Le programme vous renvoie alors le mot avec sa position ainsi que la nouvelle grille générée en jouant ce mot. 

Attention ! Nous n'avons pas traité le cas de la lettre joker.

