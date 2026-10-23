# TP de NLP (CY Tech, ING3 IA)

Mes TP du cours de traitement automatique du langage. Tous utilisent le même corpus, [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (environ 1,1 million de caractères de pièces de Shakespeare), sous deux angles différents.

## Contenu

Le cours est organisé en deux séries de TP. Chaque TP est un notebook autonome, avec ses explications et ses réponses aux questions.

### Série Transformers : construire un GPT de caractères pièce par pièce

| Notebook | Sujet |
|---|---|
| `transformer_tp1_bigramme.ipynb` | du texte brut aux entiers, couples (contexte, cible), mini-batchs, modèle bigramme, génération avant et après entraînement |
| `transformer_tp2_self_attention.ipynb` | de la moyenne causale à une tête d'attention : masque, softmax, requêtes / clés / valeurs, mise à l'échelle par √d_h |
| `transformer_tp3_layernorm_mlp.ipynb` | LayerNorm écrite à la main et comparée à `nn.LayerNorm`, MLP par token, connexion résiduelle |
| `transformer_tp4_mini_transformer.ipynb` | assemblage du mini-Transformer (4 blocs, 4 têtes, 210 000 paramètres), entraînement de 5 000 itérations, génération, expériences sur le contexte, les têtes et le dropout |

### Série classification : qui parle ?

La tâche : à partir d'une réplique, retrouver lequel des cinq personnages les plus bavards la prononce (Duke Vincentio, Gloucester, Menenius, Petruchio, Romeo). Le découpage train / validation / test (70 / 15 / 15, stratifié, `random_state=42`) est le même dans tous les notebooks, ce qui rend les scores comparables.

| Notebook | Sujet |
|---|---|
| `nlp_tp2_preparation_donnees.ipynb` | extraction des répliques, choix de la tâche, split stratifié, vocabulaire de mots, `Dataset` et `DataLoader` PyTorch |
| `nlp_tp3_prompting_llm.ipynb` | classification sans entraînement par un LLM (Qwen2.5-3B-Instruct) : trois prompts, sortie JSON validée, comparaison avec une baseline TF-IDF + régression logistique |

## Lancer les notebooks

Les TP Transformers 1 à 3 et le TP 2 de classification tournent sur CPU en quelques secondes. Le TP 4 entraîne un vrai petit modèle : compter quelques minutes sur CPU, nettement moins sur un GPU.

```bash
pip install torch pandas scikit-learn matplotlib jupyter
jupyter notebook
```

Le TP 3 (prompting) charge un modèle de 3 milliards de paramètres et demande un GPU. Il est prévu pour Google Colab avec un GPU T4 (*Exécution → Modifier le type d'exécution*). Une option permet aussi de passer par l'API Anthropic à la place du modèle local.

Les graines aléatoires sont fixées : en relançant les notebooks sur CPU, on retrouve les chiffres cités dans les réponses. Sur GPU, les valeurs du TP 4 peuvent différer un peu (les calculs n'y sont pas faits dans le même ordre). La partie du TP 3 à température 0,8 varie, elle, d'une exécution à l'autre par construction.

## Quelques résultats

Sur le test (115 répliques), la baseline TF-IDF + régression logistique atteint un F1 macro de 0,64. Le LLM en few-shot, sans aucun entraînement, plafonne à 0,42 et prédit « Duke Vincentio » beaucoup trop souvent. Pour une tâche aussi étroite, un modèle classique entraîné reste à la fois meilleur et des milliers de fois plus rapide.

Jalis Klingler
