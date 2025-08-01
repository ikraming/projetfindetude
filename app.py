import rdflib
import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric
from networkx import is_connected
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from gensim.models import Word2Vec
import networkx as nx
from collections import defaultdict
from hnswlib import Index
import os
import time
import gc
import nltk

import threading
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import torch
import numpy as np
import traceback
from colorama import init, Fore
init()
# Variables globales pour stocker le modèle pré-entraîné
MODELS_LOADED = False
MODELS_LOADING = False
rdf2vec_model = None
rgcn_model = None
entity_to_idx = None
embeddings_dict = None
combined_graph = None
query_parser = None
initialization_progress = "En attente..."

# Créer l'application Flask avec le bon chemin pour les templates
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

nltk.download('punkt')
# Fixer la graine aléatoire pour la reproductibilité(fonction random produise a chaque fois des resultat different ce grain make it donne les meme valeur a chaque execution)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


import time
import os
import gc
import rdflib
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random


def load_all_kg_files_once(file_list, visualize=True, max_nodes=50, output_dir="kg_visualizations", preprocess=True):
    """
    Charge tous les fichiers .nt pour construire un KG unifié en une seule fois avec prétraitement automatique

    Args:
        file_list: Liste des fichiers .nt à charger
        visualize: Si True, génère une visualisation du KG
        max_nodes: Nombre maximum de nœuds à afficher dans la visualisation
        output_dir: Répertoire où sauvegarder les visualisations
        preprocess: Si True, effectue le prétraitement automatiquement après le chargement

    Returns:
        tuple: (triples, graph) - Liste des triplets (prétraités si preprocess=True) et objet graphe RDF
    """
    start_time = time.time()

    print(f"  Loading all {len(file_list)} KG files at once...")

    # Vérifier l'existence de tous les fichiers avant de commencer
    existing_files = []
    total_size = 0

    for filename in file_list:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            if file_size > 0:
                existing_files.append(filename)
                total_size += file_size
                print(f"   {filename} ({file_size / (1024 * 1024):.2f} MB)")
            else:
                print(f"    {filename} is empty, ignored")
        else:
            print(f"   {filename} does not exist, ignored")

    if not existing_files:
        print("  No valid file found!")
        return None, None

    print(f" Total load: {len(existing_files)} files ({total_size / (1024 * 1024):.2f} MB)")

    try:
        add_log(f"\n ◐ Loading of  {len(existing_files)} kg files...")
        print(" ➤ loading...")

        # Initialiser le graphe RDF
        graph = rdflib.Graph()

        # Charger chaque fichier
        total_triples = 0
        for nt_file in existing_files:
            file_start_time = time.time()
            try:
                temp_graph = rdflib.Graph()
                temp_graph.parse(nt_file, format='nt')
                file_triple_count = len(temp_graph)

                # Ajouter au graphe principal
                graph += temp_graph
                total_triples += file_triple_count

                add_log(f" ▲  {nt_file}: {file_triple_count} triples in {time.time() - file_start_time:.2f}s")

                # Libérer la mémoire
                del temp_graph
                gc.collect()

            except Exception as e:
                add_log(f" Error while loading {nt_file}: {str(e)}")

        # Extraire tous les triplets du graphe combiné
        triples = [(str(s).strip(), str(p).strip(), str(o).strip()) for s, p, o in graph]

        load_time = time.time() - start_time

        if triples:
            add_log(f"\n ✔ Combined KG successfully built – Total: {len(triples)} triples ({load_time:.2f}s)")
            add_log(f" ->  Speed: {len(triples) / load_time:.0f} triples/seconde")

            # Afficher quelques exemples avant prétraitement
            add_log(" ➤ Examples of loaded triplets (before preprocessing):")
            for i, triplet in enumerate(triples[:5]):
                add_log(f"  {i + 1}. {triplet}")

            # ========== PRÉTRAITEMENT AUTOMATIQUE ==========
            if preprocess:
                print(f"\n ➤ Automatic preprocessing initiated...")
                add_log(f"\n ➤  Starting automatic preprocessing...")

                # Appeler la fonction de prétraitement
                processed_triples = preprocess_triples_minimal(triples, verbose=True)

                if processed_triples:
                    triples = processed_triples  # Remplacer les triplets originaux par les triplets prétraités
                    add_log(f" ✔ Preprocessing completed successfully!")
                    add_log(f" ➤ Final triplets after preprocessing: {len(triples)}")

                    # Afficher quelques exemples après prétraitement
                    print("  Examples of triplets after preprocessing:")
                    for i, triplet in enumerate(triples[:5]):
                        add_log(f"  {i + 1}. {triplet}")
                else:
                    add_log(" ➤ Preprocessing failed, original triplets retained")
            else:
                add_log("➤  Preprocessing disabled (preprocess=False)")

            # Analyser la distribution des types d'éléments (sur les données finales)
            element_stats = analyze_kg_elements(triples, output_dir)
            print(f"\n ***********Analyse des éléments du KG {'(après prétraitement)' if preprocess else ''}:**************")
            print(f"  - Unique entities: {element_stats['entities']:,}")
            print(f"  - Unique relations / predicates: {element_stats['properties']:,}")
            print(f"  - Unique literals: {element_stats['literals']:,}")
            print(f"  - Total triples: {len(triples):,}")

            # Générer la visualisation si demandée (sur les données finales)
            if visualize:
                try:
                    print(f" ➤ Generating visualization...")
                    generate_kg_visualization(triples, max_nodes, output_dir)
                    add_log(f"➤  Visualization successfully generated")
                except Exception as e:
                    add_log(f"  Error while generating visualization: {str(e)}")

            # Temps total incluant le prétraitement
            total_time = time.time() - start_time
            add_log(
                f"\n ➤ Total time (loading + {'preprocessing + ' if preprocess else ''}analyse): {total_time:.2f}s")

            return triples, graph
        else:
            add_log("  no triple loaded")
            return None, None

    except Exception as e:
        add_log(f"  error while loading: {str(e)}")
        add_log(f"  Trace: {traceback.format_exc()}")
        return None, None


#pretraitement
def preprocess_triples_minimal(triples, verbose=True):
    """
    Prétraitement minimal des triplets RDF avec détection RÉELLE des problèmes
    """
    import sys
    import re
    import time
    start_time = time.time()

    if verbose:
        print(f"\n" + "=" * 60)
        print(" ➤ start of preprocessing")
        print("=" * 60)
        print(f" - Initiale triples: {len(triples)}")
        sys.stdout.flush()

        # DIAGNOSTIC: Afficher le type et contenu réel des premiers triplets
        add_log("\n  DIAGNOSTIC – Initial raw triplets:")
        for i, triplet in enumerate(triples[:10]):
            add_log(f"  {i + 1}. Type: {type(triplet)} | Content: {triplet}")
            if isinstance(triplet, (tuple, list)) and len(triplet) >= 3:
                add_log(f"      Subjet: '{triplet[0]}' | Prédicate: '{triplet[1]}' | Object: '{triplet[2]}'")
        sys.stdout.flush()

    # Étape 1: Suppression des triplets avec éléments vides/None
    add_log("\n ➤ Étape 1/3: Removing triplets with empty elements...")
    sys.stdout.flush()

    valid_triples = []
    empty_count = 0
    empty_examples = []

    for i, triplet in enumerate(triples):
        if i % 10000 == 0 and i > 0:
            print(f"   Processing: {i}/{len(triples)} triples...")
            sys.stdout.flush()

        is_empty, reason = is_triple_empty_detailed(triplet)
        if not is_empty:
            valid_triples.append(triplet)
        else:
            empty_count += 1
            if len(empty_examples) < 10:
                empty_examples.append((triplet, reason))

    if empty_examples:
        add_log(f"\n    Examples of empty triplets detected:")
        for j, (triplet, reason) in enumerate(empty_examples):
            add_log(f"      {j + 1}. {reason}: {triplet}")

    print(f" ✔ Step 1 finished: {empty_count} empty triples deleted")
    print(f"    Remaining non-empty triplets: {len(valid_triples)}")
    sys.stdout.flush()

    # Étape 2: Suppression des triplets malformés
    add_log("\n ➤ Step 2/3: delete of malformed triples...")
    sys.stdout.flush()

    well_formed_triples = []
    malformed_count = 0
    malformed_examples = []

    for i, triplet in enumerate(valid_triples):
        if i % 10000 == 0 and i > 0:
            print(f"   Validation: {i}/{len(valid_triples)} triples...")
            sys.stdout.flush()

        is_malformed, reason = is_triple_malformed_detailed(triplet)
        if not is_malformed:
            well_formed_triples.append(triplet)
        else:
            malformed_count += 1
            if len(malformed_examples) < 10:
                malformed_examples.append((triplet, reason))

    if malformed_examples:
        add_log(f"\n    exemple of malformed triples detected :")
        for j, (triplet, reason) in enumerate(malformed_examples):
            add_log(f"      {j + 1}. {reason}: {triplet}")

    print(f" ✔ Step 2 finished: {malformed_count} triples malformed deleted")
    print(f"    Well-formed triplets left: {len(well_formed_triples)}")
    sys.stdout.flush()

    # Étape 3: Suppression des doublons
    add_log("\n ➤ Step 3/3: Removing duplicates...")
    sys.stdout.flush()

    # Créer un set pour la déduplication avec comparaison exacte
    seen_triples = set()
    unique_triples = []
    duplicates_removed = 0
    duplicate_examples = []

    for triplet in well_formed_triples:
        # Créer une signature exacte du triplet
        triplet_signature = create_triplet_signature(triplet)

        if triplet_signature not in seen_triples:
            seen_triples.add(triplet_signature)
            unique_triples.append(triplet)
        else:
            duplicates_removed += 1
            if len(duplicate_examples) < 5:
                duplicate_examples.append(triplet)

    if duplicate_examples:
        add_log(f"\n    exemple of duplicate triples deleted:")
        for j, triplet in enumerate(duplicate_examples):
            add_log(f"      {j + 1}. {triplet}")

    print(f" ✔ Step 3 finished: {duplicates_removed} deleted deplicates")
    print(f"    final unique triples: {len(unique_triples)}")

    total_time = time.time() - start_time
    add_log(f"\n ➤ time of preprocessing: {total_time:.2f}s")
    print("=" * 60)
    add_log(" ✔ preprocessing finished")
    print("=" * 60)

    # Résumé des suppressions
    total_removed = empty_count + malformed_count + duplicates_removed
    add_log(f"\n **************Summary of Removals:***************")
    add_log(f"   • empty triples: {empty_count}")
    add_log(f"   • malformed triples: {malformed_count}")
    add_log(f"   • duplicates: {duplicates_removed}")
    add_log(f"   • Total deleted: {total_removed}")
    add_log(f"   • Preserved triplets: {len(unique_triples)}")
    add_log(f"   • Retention rate: {len(unique_triples) / len(triples) * 100:.1f}%")

    if verbose and unique_triples:
        print("\n ➤ Examples of triplets after preprocessing:")
        for i, triplet in enumerate(unique_triples[:5]):
            print(f"  {i + 1}. {triplet}")

    return unique_triples


def is_triple_empty_detailed(triplet):
    """
    Vérifie si un triplet est vide et retourne la raison
    Returns: (is_empty: bool, reason: str)
    """
    # Vérifier la structure
    if not isinstance(triplet, (tuple, list)):
        return True, "Not a tuple/list"

    if len(triplet) != 3:
        return True, f"Incorrect length: {len(triplet)}"

    subject, predicate, obj = triplet

    # Vérifier chaque élément
    for i, (elem, name) in enumerate([(subject, "Subjet"), (predicate, "Prédicate"), (obj, "Object")]):
        if elem is None:
            return True, f"{name} is None"

        # Convertir en string
        elem_str = str(elem).strip()

        # Vérifications d'éléments vides
        if not elem_str:
            return True, f"{name} empty"

        if elem_str == '.':
            return True, f"{name} Is just a dot"

        if elem_str in ['""', "''", '" "', "' '"]:
            return True, f"{name} Empty quotation marks"

        # Vérification spéciale pour l'objet manquant
        if i == 2 and (elem_str.endswith(' .') or elem_str == ''):
            return True, f"Missing or incomplete object"

    return False, ""


def is_triple_malformed_detailed(triplet):
    """
    Vérifie si un triplet est malformé - AMÉLIORÉ
    Le critère principal: le sujet NE DOIT PAS commencer par un littéral
    Returns: (is_malformed: bool, reason: str)
    """
    if not isinstance(triplet, (tuple, list)) or len(triplet) != 3:
        return True, " invalide structure"

    subject, predicate, obj = triplet

    # Convertir en strings pour l'analyse
    subject_str = str(subject).strip()
    predicate_str = str(predicate).strip()
    obj_str = str(obj).strip()

    # CRITÈRE PRINCIPAL: Le sujet ne doit PAS être un littéral
    if is_subject_literal(subject_str):
        return True, f"Invalid subject: starts with a literal instead of a URI: '{subject_str[:50]}...'"

    # Vérifications supplémentaires du prédicat
    if not is_valid_predicate(predicate_str):
        return True, f" invalide predicate: '{predicate_str}'"

    return False, ""


def is_subject_literal(subject_str):
    """
    Détecte si un sujet est un littéral (ne commence pas par une URI)
    """
    if not subject_str:
        return True

    # Les URIs valides commencent par:
    # 1. http:// ou https://
    if subject_str.startswith(('http://', 'https://')):
        return False

    # 2. <http://... (URI complète avec < >)
    if subject_str.startswith('<http') and subject_str.endswith('>'):
        return False

    # 3. Préfixe namespace (ex: dbr:, owl:, etc.)
    if ':' in subject_str and not subject_str.startswith('"'):
        # Vérifier que c'est bien un préfixe (pas trop d'espaces)
        if subject_str.count(' ') <= 1:  # Tolérer un espace
            return False

    # 4. URIs relatives commençant par /
    if subject_str.startswith('/'):
        return False

    # Tout le reste est considéré comme un littéral
    return True


def is_valid_predicate(predicate_str):
    """
    Validation simple des prédicats
    """
    if not predicate_str:
        return False

    # Les prédicats valides:
    # 1. URIs complètes
    if predicate_str.startswith(('http://', 'https://', '<http')):
        return True

    # 2. Préfixes namespace
    if ':' in predicate_str and not predicate_str.startswith('"'):
        return True

    # 3. Prédicats RDF/RDFS/OWL standards
    standard_predicates = [
        'rdf:type', 'rdfs:label', 'rdfs:comment', 'owl:sameAs',
        'a'  # raccourci pour rdf:type
    ]
    if predicate_str in standard_predicates:
        return True

    return False


def create_triplet_signature(triplet):
    """
    Crée une signature unique pour détecter les doublons exacts
    """
    if not isinstance(triplet, (tuple, list)) or len(triplet) != 3:
        return str(triplet)

    # Normaliser chaque élément (enlever espaces en début/fin)
    subject = str(triplet[0]).strip()
    predicate = str(triplet[1]).strip()
    obj = str(triplet[2]).strip()

    # Créer une signature exacte
    return f"{subject}|||{predicate}|||{obj}"


# Fonction de diagnostic améliorée
def diagnose_triplets_improved(triples, sample_size=100):
    """
    Analyse améliorée des triplets pour identifier les problèmes
    """
    print(f"\n  Improved Diagnosis of {min(sample_size, len(triples))} TRIPLETS:")

    issues_found = {
        'empty': [],
        'malformed': [],
        'duplicates': [],
        'valid': []
    }

    seen_signatures = set()

    for i, triplet in enumerate(triples[:sample_size]):
        print(f"\n--- Triplet {i + 1} ---")
        print(f"Content: {triplet}")
        #soit un tuple a 3 elements
        if isinstance(triplet, (tuple, list)) and len(triplet) >= 3:
            print(f"  Subjet: '{triplet[0]}'")
            print(f"  Predicate: '{triplet[1]}'")
            print(f"  Object: '{triplet[2]}'")

        # Test vide
        is_empty, empty_reason = is_triple_empty_detailed(triplet)
        if is_empty:
            print(f" empty: {empty_reason}")
            #append pour les liste ajouter un elemnt a la fin dune list
            issues_found['empty'].append((i, triplet, empty_reason))
            continue

        # Test malformé
        is_malformed, malformed_reason = is_triple_malformed_detailed(triplet)
        if is_malformed:
            print(f"  MALFORMED: {malformed_reason}")
            issues_found['malformed'].append((i, triplet, malformed_reason))
            continue

        # Test doublon
        signature = create_triplet_signature(triplet)
        if signature in seen_signatures:
            print(f" DETECTED DEPLICATES")
            issues_found['duplicates'].append((i, triplet))
        else:
            seen_signatures.add(signature)
            print(f" VALIDE")
            issues_found['valid'].append((i, triplet))

    # Résumé
    print(f"\n *************** Diagnostic Summary:****************")
    print(f"  • empty triples: {len(issues_found['empty'])}")
    print(f"  • malformed triples: {len(issues_found['malformed'])}")
    print(f"  • Duplicates: {len(issues_found['duplicates'])}")
    print(f"  • valide triples: {len(issues_found['valid'])}")

    return issues_found


def analyze_kg_elements(triples, output_dir="kg_visualizations"):
    """
    Analyse les types d'éléments dans le KG de manière simplifiée
    Dans un fichier .nt : chaque ligne = triplet (sujet, prédicat, objet)
    - Prédicat (élément 2) = toujours une relation/propriété
    - Sujet et Objet (éléments 1 et 3) = entité (URI) ou littéral (pas URI)

    Args:
        triples: Liste des triplets (sujet, prédicat, objet)
        output_dir: Répertoire de sortie pour les graphiques

    Returns:
        dict: Statistiques des éléments
    """

    def is_uri(element):
        """Vérifie si un élément est une URI (donc une entité)"""
        if not element:
            return False

        element = str(element).strip()

        # Cas 1: URI avec crochets <http://...>
        if element.startswith('<') and element.endswith('>') and '://' in element:
            return True

        # Cas 2: URI sans crochets http://...
        if element.startswith(('http://', 'https://', 'ftp://', 'urn:')):
            return True

        # Cas 3: URI relative ou autres formats
        if '://' in element and not element.startswith('"') and not element.endswith('"'):
            return True

        return False

    # Compteurs pour chaque type d'éléments uniques
    entities = set()  # URIs uniques (sujet ou objet)
    properties = set()  # Prédicats uniques (toujours élément 2)
    literals = set()  # Littéraux uniques (sujet ou objet, pas URI)

    # Compteurs pour les occurrences totales
    total_entity_occurrences = 0
    total_property_occurrences = 0
    total_literal_occurrences = 0

    add_log("\n -- Simplified analysis of element types...--")

    # Debug: Afficher quelques triplets pour comprendre le format
    add_log("\n ▲ Debug – Initial triplets for analysis:")
    for i, (subject, predicate, obj) in enumerate(triples[:3]):
        add_log(f" ■  Triplet {i + 1}:")
        add_log(f" ■   Subjet: '{subject}' (URI: {is_uri(subject)})")
        add_log(f" ■   Predicate: '{predicate}' (URI: {is_uri(predicate)})")
        add_log(f" ■  Object: '{obj}' (URI: {is_uri(obj)})")

    for subject, predicate, obj in triples:
        # Élément 2 (prédicat) = toujours une propriété/relation
        properties.add(predicate)
        total_property_occurrences += 1

        # Élément 1 (sujet) : vérifier s'il est à la position 1 ET si c'est une URI
        if is_uri(subject):
            entities.add(subject)
            total_entity_occurrences += 1
        else:
            literals.add(subject)
            total_literal_occurrences += 1

        # Élément 3 (objet) : vérifier s'il est à la position 3 ET si c'est une URI
        if is_uri(obj):
            entities.add(obj)
            total_entity_occurrences += 1
        else:
            literals.add(obj)
            total_literal_occurrences += 1

    # Statistiques finales
    stats = {
        'entities': len(entities),
        'properties': len(properties),
        'literals': len(literals),
        'total_triples': len(triples),
        # Statistiques d'occurrences
        'entity_occurrences': total_entity_occurrences,
        'property_occurrences': total_property_occurrences,
        'literal_occurrences': total_literal_occurrences
    }

    # Afficher quelques exemples pour vérification
    add_log(f"\n  -- Examples of detected elements:--")
    if properties:
        sample_props = list(properties)[:5]
        add_log(f" ■ unique proprities ({len(properties)}): {sample_props}")
    if entities:
        sample_entities = list(entities)[:3]
        add_log(f" ■ unique entities ({len(entities)}): {sample_entities}")
    if literals:
        sample_literals = list(literals)[:3]
        add_log(f" ■ unique literals ({len(literals)}): {sample_literals}")

    add_log(f"\n ******************* Occurences Statistiques:*****************")
    add_log(f"  ■  enties Occurrences : {total_entity_occurrences:,}")
    add_log(f"  ■  proprities Occurrences : {total_property_occurrences:,}")
    add_log(f"  ■  leterals Occurrences : {total_literal_occurrences:,}")

    # Vérification de cohérence
    total_elements = total_entity_occurrences + total_property_occurrences + total_literal_occurrences
    expected_elements = len(triples) * 3  # 3 éléments par triplet
    add_log(f"\n - Vérification: {total_elements} analyzed elements / {expected_elements} Expected")

    # Générer les graphiques
    create_element_distribution_chart(stats, output_dir)

    return stats

def create_element_distribution_chart(stats, output_dir):
    """
    Crée des graphiques pour la distribution des éléments

    Args:
        stats: Dictionnaire avec les statistiques des éléments
        output_dir: Répertoire de sortie
    """

    # Créer le répertoire si nécessaire
    os.makedirs(output_dir, exist_ok=True)

    # Créer une figure avec plusieurs sous-graphiques
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Graphique en secteurs - Éléments uniques
    unique_labels = []
    unique_sizes = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    unique_data = [
        ('Entities', stats['entities']),
        ('Propriéties/Relations', stats['properties']),
        ('Literals', stats['literals'])
    ]

    for label, count in unique_data:
        if count > 0:
            unique_labels.append(f"{label}\n({count:,})")
            unique_sizes.append(count)

    if unique_sizes:
        wedges1, texts1, autotexts1 = ax1.pie(unique_sizes, labels=unique_labels,
                                              colors=colors[:len(unique_sizes)],
                                              autopct='%1.1f%%', startangle=90,
                                              explode=[0.05] * len(unique_sizes))

        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        ax1.set_title('Distribution of unique elements', fontsize=14, fontweight='bold')

    # 2. Graphique en secteurs - Occurrences totales
    occurrence_labels = []
    occurrence_sizes = []

    occurrence_data = [
        ('Entities', stats['entity_occurrences']),
        ('Propriéties/Relations', stats['property_occurrences']),
        ('Literals', stats['literal_occurrences'])
    ]

    for label, count in occurrence_data:
        if count > 0:
            occurrence_labels.append(f"{label}\n({count:,})")
            occurrence_sizes.append(count)

    if occurrence_sizes:
        wedges2, texts2, autotexts2 = ax2.pie(occurrence_sizes, labels=occurrence_labels,
                                              colors=colors[:len(occurrence_sizes)],
                                              autopct='%1.1f%%', startangle=90,
                                              explode=[0.05] * len(occurrence_sizes))

        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        ax2.set_title('Distribution of total occurences', fontsize=14, fontweight='bold')

    # 3. Graphique en barres - Comparaison
    categories = ['Entities', 'Proprities', 'Literals']
    unique_counts = [stats['entities'], stats['properties'], stats['literals']]
    occurrence_counts = [stats['entity_occurrences'], stats['property_occurrences'], stats['literal_occurrences']]

    x = range(len(categories))
    width = 0.35

    bars1 = ax3.bar([i - width / 2 for i in x], unique_counts, width,
                    label='unique elements', color='lightcoral', alpha=0.8)
    bars2 = ax3.bar([i + width / 2 for i in x], occurrence_counts, width,
                    label='total occurences', color='skyblue', alpha=0.8)

    ax3.set_xlabel('Types of \'elements')
    ax3.set_ylabel('Nombre')
    ax3.set_title('Comparaison: Uniques vs Occurrences', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.set_yscale('log')  # Échelle logarithmique pour mieux voir les différences

    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height):,}', ha='center', va='bottom', fontsize=8)

    # 4. Tableau de statistiques détaillées
    ax4.axis('off')

    table_data = [
        ['Entités uniques', f'{stats["entities"]:,}', f'{stats["entity_occurrences"]:,}'],
        ['Propriétés/Relations uniques', f'{stats["properties"]:,}', f'{stats["property_occurrences"]:,}'],
        ['Littéraux uniques', f'{stats["literals"]:,}', f'{stats["literal_occurrences"]:,}'],
        ['TOTAL', f'{stats["entities"] + stats["properties"] + stats["literals"]:,}',
         f'{stats["total_triples"]:,} triplets']
    ]

    table = ax4.table(cellText=table_data,
                      colLabels=['Type', 'Éléments Uniques', 'Occurrences Totales'],
                      cellLoc='center',
                      loc='center',
                      colColours=['lightblue'] * 3)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    #position de titre un peut bas  par le y
    ax4.set_title('Detailed Statistics of the Knowledge Graph',
                  fontweight='bold', y=0.8, fontsize=14)

    # Ajuster la mise en page
    plt.tight_layout()

    # Sauvegarder
    output_path = os.path.join(output_dir, 'kg_elements_distribution_fixed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    add_log(f" -- Distribution chart saved: {output_path}")

    # Afficher le graphique
    plt.show()
    plt.close()


def generate_detailed_kg_statistics(triples, output_dir="kg_visualizations"):
    """
    Génère des statistiques détaillées sur le KG

    Args:
        triples: Liste des triplets
        output_dir: Répertoire de sortie
    """

    add_log("\n *** Generating detailed KG statistics...***")

    # Analyse des domaines
    domains = Counter()
    for subject, predicate, obj in triples:
        # Nettoyer les URIs et extraire les domaines
        for element in [subject, predicate, obj]:
            clean_element = element.strip('<>')
            if clean_element.startswith('http'):
                domain = urlparse(clean_element).netloc
                domains[domain] += 1

    # Top 10 des domaines
    top_domains = domains.most_common(10)

    # Analyse des prédicats les plus fréquents
    predicates = Counter(predicate for _, predicate, _ in triples)
    top_predicates = predicates.most_common(10)

    # Créer un rapport
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'kg_detailed_stats.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== detailed report of  kg ===\n\n")
        f.write(f"total number of  triplets: {len(triples):,}\n")
        f.write(f" unique number of proprities: {len(set(p for _, p, _ in triples)):,}\n")
        f.write(f"unique number of domains: {len(domains)}\n\n")

        f.write("Top 10 Domains:\n")
        for domain, count in top_domains:
            f.write(f"  {domain}: {count:,} éléments\n")

        f.write("\nTop 10 Properties/Predicates:\n")
        for predicate, count in top_predicates:
            f.write(f"  {predicate}: {count:,} occurrences\n")

    add_log(f" --Detailed report saved: {report_path}")

    return {
        'domains': dict(domains),
        'predicates': dict(predicates),
        'top_domains': top_domains,
        'top_predicates': top_predicates
    }



def generate_kg_visualization(triples, max_nodes=50, output_dir="kg_visualizations"):
    """
    Génère une visualisation graphique d'une partie du Knowledge Graph

    Args:
        triples: Liste des triplets (sujet, prédicat, objet)
        max_nodes: Nombre maximum de nœuds à afficher
        output_dir: Répertoire de sortie pour les fichiers
    """
    start_viz_time = time.time()
    add_log(f"\n  Génération of the  visualisation of KG...")

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Créer un graphe NetworkX
    G = nx.Graph()

    # Compter la fréquence des entités pour sélectionner les plus importantes
    entity_freq = defaultdict(int)
    predicate_freq = defaultdict(int)

    for subj, pred, obj in triples:
        entity_freq[subj] += 1
        entity_freq[obj] += 1
        predicate_freq[pred] += 1

    # Sélectionner les entités les plus fréquentes
    top_entities = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)
    selected_entities = set([entity for entity, freq in top_entities[:max_nodes]])

    # Ajouter les triplets contenant ces entités
    edges_added = 0
    edge_labels = {}

    for subj, pred, obj in triples:
        if subj in selected_entities and obj in selected_entities:
            # Simplifier les URIs pour l'affichage
            subj_label = simplify_uri(subj)
            obj_label = simplify_uri(obj)
            pred_label = simplify_uri(pred)

            G.add_edge(subj_label, obj_label)
            edge_labels[(subj_label, obj_label)] = pred_label
            edges_added += 1

            if edges_added >= max_nodes * 2:  # Limiter le nombre d'arêtes
                break

    add_log(f"   Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Créer la visualisation
    plt.figure(figsize=(16, 12))
    plt.clf()

    # Calculer la disposition des nœuds
    if G.number_of_nodes() > 0:
        try:
            # Utiliser spring_layout pour une disposition agréable
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G, seed=42)

        # Calculer les tailles des nœuds basées sur leur degré
        node_sizes = [max(300, G.degree(node) * 100) for node in G.nodes()]

        # Dessiner les nœuds
        nx.draw_networkx_nodes(G, pos,
                               node_size=node_sizes,
                               node_color='lightblue',
                               alpha=0.7,
                               edgecolors='navy')

        # Dessiner les arêtes
        nx.draw_networkx_edges(G, pos,
                               alpha=0.5,
                               edge_color='gray',
                               width=1)

        # Ajouter les labels des nœuds
        nx.draw_networkx_labels(G, pos,
                                font_size=8,
                                font_weight='bold')

        # Ajouter les labels des arêtes (prédicats) - seulement pour les arêtes importantes
        important_edges = list(G.edges())[:20]  # Limiter à 20 labels d'arêtes
        important_edge_labels = {edge: edge_labels.get(edge, '')
                                 for edge in important_edges if edge in edge_labels}

        nx.draw_networkx_edge_labels(G, pos,
                                     important_edge_labels,
                                     font_size=6,
                                     alpha=0.7)

    plt.title(f"Visualisation og Knowledge Graph\n({G.number_of_nodes()} nodes, {G.number_of_edges()} relations)",
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    # Sauvegarder la visualisation
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # PNG haute résolution
    png_filename = os.path.join(output_dir, f"kg_visualization_{timestamp}.png")
    plt.savefig(png_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    # SVG pour la scalabilité
    svg_filename = os.path.join(output_dir, f"kg_visualization_{timestamp}.svg")
    plt.savefig(svg_filename, format='svg', bbox_inches='tight')

    plt.close()

    add_log(f"   Visualisation saved :")
    print(f"    - PNG: {png_filename}")
    print(f"    - SVG: {svg_filename}")
    add_log(f"   Visualisation generated in {time.time() - start_viz_time:.2f}s")


def simplify_uri(uri):
    """
    Simplifie une URI pour l'affichage en gardant seulement la partie finale
    """
    #ca existe dan srdf ou owl form
    if '#' in uri:
        return uri.split('#')[-1]
    elif '/' in uri:
        return uri.split('/')[-1]
    return uri[:30] + "..." if len(uri) > 30 else uri

# depth nombre de sauts
# Étape 2 : Convertir le KG en embeddings avec RDF2Vec amélioré pour gérer de grands graphes
def generate_random_walks(triples, num_walks=10, depth=4, max_entities=None):
    """
    Génère des marches aléatoires à partir des triplets du KG avec gestion de mémoire améliorée
    """
    start_time = time.time()

    # Construire un graphe orienté à partir des triplets
    add_log("\n ** Graph construction for random walks...**")
    G = nx.DiGraph()
    relation_dict = defaultdict(list)

    # Ajouter les arêtes au graphe
    for s, p, o in triples:
        G.add_edge(s, o, relation=p)
        relation_dict[(s, o)].append(p)

    # Identifier toutes les entités uniques
    entities = list(set([s for s, _, _ in triples] + [o for _, _, o in triples]))

    # Limiter le nombre d'entités si spécifié (pour la gestion de mémoire)
    if max_entities and len(entities) > max_entities:
        add_log(f"  Limitation at {max_entities} entities in {len(entities)} for walks generation")
        entities = entities[:max_entities]

    add_log(
        f"\n ️ ■ Génération of {num_walks} Depth-based random walks {depth} for {len(entities)} entities...")

    walks = []
    entities_processed = 0
    report_interval = max(1, len(entities) // 10)  # Rapport tous les 10%

    for entity in entities:
        if entity in G:
            for i in range(num_walks):
                walk = [entity]
                current = entity
                for j in range(depth):
                    neighbors = list(G.neighbors(current))
                    if not neighbors:
                        break
                    next_node = random.choice(neighbors)
                    # Ajouter la relation dans la marche
                    relations = relation_dict[(current, next_node)]
                    relation = random.choice(relations) if relations else "UNKNOWN_RELATION"
                    walk.append(relation)
                    walk.append(next_node)
                    current = next_node
                if len(walk) > 1:  # Garder uniquement les marches non triviales
                    walks.append(walk)

        entities_processed += 1
        if entities_processed % report_interval == 0:
            progress = entities_processed / len(entities) * 100
            add_log(f"  ◆ Progression: {progress:.1f}% ({entities_processed}/{len(entities)} traited entities)")

    add_log(f" ○ generated {len(walks)} Random walks in {time.time() - start_time:.2f}s")
    if walks:
        add_log(f"---Exemple of  a walk: {walks[0]}---")

    return walks, entities


def save_rdf2vec_embeddings_to_csv(model, entities, filename="rdf2vec_embeddings.csv"):
    """
    Sauvegarde les embeddings RDF2Vec dans un fichier CSV
    """
    import pandas as pd
    import os

    start_time = time.time()
    add_log(f"\n ✔ saved rdf2vec embeddings in {filename}...")

    try:
        # Créer le répertoire de sauvegarde si nécessaire
        save_dir = "embeddings_output"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            add_log(f" ✔ Répertoire créé: {save_dir}")

        filepath = os.path.join(save_dir, filename)

        # Préparer les données pour le CSV
        embeddings_data = []
        entities_with_embeddings = 0
        entities_without_embeddings = 0

        for entity in entities:
            if entity in model.wv:
                embedding = model.wv[entity].tolist()  # Convertir en liste Python
                row = {'entity': entity}
                # Ajouter chaque dimension d'embedding comme colonne
                for i, value in enumerate(embedding):
                    row[f'dim_{i}'] = value
                embeddings_data.append(row)
                entities_with_embeddings += 1
            else:
                entities_without_embeddings += 1

        # Créer le DataFrame et sauvegarder
        if embeddings_data:
            df = pd.DataFrame(embeddings_data)
            df.to_csv(filepath, index=False, encoding='utf-8')

            # Statistiques de sauvegarde
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            vector_size = len(embeddings_data[0]) - 1  # -1 pour la colonne 'entity'

            add_log(f" ✔ Embeddings RDF2Vec saved Successfully!")
            add_log(f"  ▲  File: {filepath}")
            add_log(f"  ▲  entities with embeddings: {entities_with_embeddings}")

            add_log(f"  ▲ Dimension of vectors: {vector_size}")
            add_log(f"  ▲ file size: {file_size_mb:.2f} MB")
            add_log(f"  ▲ Teme of saving sauvegarde: {time.time() - start_time:.2f}s")

            # Aperçu des premières lignes
            add_log(f"\n ▲ Data preview saved:")
            add_log(f" ▲  Columns: {list(df.columns[:6])}{'...' if len(df.columns) > 6 else ''}")
            add_log(f" ▲ first entity: {df.iloc[0]['entity']}")
            add_log(f" ▲ first dims: {[f'{df.iloc[0]["dim_" + str(i)]:.4f}' for i in range(min(5, vector_size))]}")

            return filepath, entities_with_embeddings, vector_size
        else:
            add_log(" No embedding found to save")
            return None, 0, 0

    except Exception as e:
        add_log(f" Error while saving: {str(e)}")
        return None, 0, 0

#window 5 he see 5 before him and 5 after him  and min count is how much the word appear in embeddings

def train_rdf2vec(triples, vector_size=100, window=5, min_count=1, sg=1, workers=4, max_entities=None,
                  batch_size=10000, save_embeddings=True):
    """
    Entraîne un modèle RDF2Vec sur les marches aléatoires du KG avec gestion de la mémoire améliorée
    et sauvegarde automatique des embeddings
    """
    start_time = time.time()

    # Générer les marches aléatoires avec limitation possible du nombre d'entités
    walks, entities = generate_random_walks(triples, num_walks=8, depth=4, max_entities=max_entities)

    # Entraîner le modèle Word2Vec sur les marches
    add_log("\n ■ Training the RDF2Vec (Word2Vec) model on random walks... ")
    model = Word2Vec(sentences=walks,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     sg=sg,  # Skip-gram (sg=1) vs CBOW (sg=0)
                     workers=workers,#nombre de cpu travail en paralelle
                     seed=SEED)

    # Créer un dictionnaire entity_to_id
    entity_to_id = {entity: idx for idx, entity in enumerate(entities)}

    #Afficher quelques exemples d'embeddings
    add_log("\n --- Some entity embeddings (RDF2Vec):")
    sample_count = 0
    for entity in entities:
        if sample_count >= 5:
            break
        if entity in model.wv:
            embedding = model.wv[entity]
            add_log(f"{entity}: {embedding[:5]}... (dim: {len(embedding)})")
            sample_count += 1
        else:
            if sample_count < 5:
                add_log(f"{entity}: Not found in the model")
                sample_count += 1

    # Sauvegarder les embeddings en CSV si demandé
    if save_embeddings:
        csv_filename = f"rdf2vec_embeddings_dim{vector_size}.csv"
        filepath, entities_saved, vector_dim = save_rdf2vec_embeddings_to_csv(model, entities, csv_filename)

        if filepath:
            add_log(f" ✔ RDF2Vec embeddings saved: {filepath}")
        else:
            add_log("  Error while saving embeddings")

    add_log(f"\n ▲ Modèle RDF2Vec entrained in  {time.time() - start_time:.2f}s ***")
    add_log(f"   ▲ Dimension og embeddings: {vector_size}")
    add_log(f"   ▲ Number of  entities with embedding: {len(model.wv.index_to_key)}")

    return model, entity_to_id, entities


# Étape 3 : Construire le graphe pour RGCN avec gestion de la mémoire améliorée
def build_rgcn_data(triples, model, entity_to_id, embedding_dim=100, batch_process=True):
    """
    Construit les données pour le modèle RGCN avec une meilleure gestion de la mémoire
    """
    start_time = time.time()

    # Extraire les entités uniques
    entities = set([s for s, _, _ in triples] + [o for _, _, o in triples])
    entity_to_idx = {entity.strip(): idx for idx, entity in enumerate(entities)}

    # Extraire les types de relations et créer un dictionnaire
    relations = set([p for _, p, _ in triples])
    relation_to_idx = {relation.strip(): idx for idx, relation in enumerate(relations)}
    num_relations = len(relation_to_idx)

    add_log(f"\n ■  Construction of  RGCN data :")
    add_log(f"   ▲ entities number: {len(entity_to_idx)}")
    add_log(f"   ▲ Nombre of relations types: {num_relations}")

    # Créer les edge_index et edge_type pour RGCN
    edge_index_list = []
    edge_type_list = []

    # Traitement par lots si activé veut dire les divise a des partie et traiter separament
    if batch_process and len(triples) > 170000:
        batch_size = 100000
        num_batches = (len(triples) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(triples))
            batch_triples = triples[start_idx:end_idx]

            for s, p, o in batch_triples:
                if s in entity_to_idx and o in entity_to_idx:
                    edge_index_list.append((entity_to_idx[s], entity_to_idx[o]))
                    edge_type_list.append(relation_to_idx[p])

            if (i + 1) % 5 == 0 or (i + 1) == num_batches:
                progress = (i + 1) / num_batches * 100
                add_log(f" ■  Progression: {progress:.1f}% (lot {i + 1}/{num_batches}, {len(edge_index_list)} arêtes)")
    else:
        # Traitement en une seule fois
        for s, p, o in triples:
            if s in entity_to_idx and o in entity_to_idx:
                edge_index_list.append((entity_to_idx[s], entity_to_idx[o]))
                edge_type_list.append(relation_to_idx[p])

    # Conversion en tensors
    add_log("\n ■ Conversion data to  tensors...")
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)

    add_log(f"\n ■ edge data:")
    add_log(f"   ■ Dimensions edge_index: {edge_index.size()}")
    add_log(f"   ■ Dimensions edge_type: {edge_type.size()}")

    # Créer les embeddings initiaux à partir du modèle RDF2Vec
    add_log("\n ▲ Creation of initial embeddings ...")
    entity_embeddings = []

    for e in entity_to_idx.keys():
        if e in model.wv:
            embedding = model.wv[e]
        else:
            # Utiliser un embedding aléatoire pour les entités non trouvées
            embedding = np.random.randn(model.vector_size)
        entity_embeddings.append(embedding)

    # Conversion en tensor
    embeddings = torch.tensor(np.vstack(entity_embeddings), dtype=torch.float)

    add_log(f"\n ✔ RGCN data built in {time.time() - start_time:.2f}s ")
    add_log(f"   ▲ Dimension of initial embeddings : {embeddings.shape}")

    return Data(x=embeddings, edge_index=edge_index, edge_type=edge_type), entity_to_idx, num_relations


# Étape 4 : Modèles RGCN
# Classe RGCN améliorée avec mécanisme d'attention avancé et normalisation
# blocks couch
#DROUPOUT éviter l’overfitting
# num bases :Nombre de bases pour la factorisation des relations (optionnel, utile dans R-GCN pour réduire le nombre de paramètres).
class RGCNAdvanced(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 dropout=0.2, num_bases=None, num_blocks=2, attention_heads=8,
                 use_layer_norm=True, residual_connections=True):
        super().__init__()

        self.num_blocks = num_blocks
        self.use_layer_norm = use_layer_norm
        self.residual_connections = residual_connections

        # Couche initiale de projection
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)

        # Blocks RGCN (plusieurs couches)
        self.rgcn_blocks = torch.nn.ModuleList()
        self.attention_blocks = torch.nn.ModuleList()

        if use_layer_norm:
            self.layer_norms = torch.nn.ModuleList()

        # Construire les blocks
        for i in range(num_blocks):
            # Si c'est le dernier block, la sortie a une dimension différente
            block_out_dim = out_channels if i == num_blocks - 1 else hidden_channels

            # Couche RGCN
            self.rgcn_blocks.append(
                RGCNConv(hidden_channels, block_out_dim, num_relations, num_bases=num_bases)
            )

            # Mécanisme d'attention
            self.attention_blocks.append(
                torch.nn.MultiheadAttention(block_out_dim, num_heads=attention_heads)
            )

            # Layer normalization
            if use_layer_norm:
                self.layer_norms.append(torch.nn.LayerNorm(block_out_dim))

        # Projection finale
        self.output_proj = torch.nn.Linear(out_channels, in_channels)

        # Dropout
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        # Projection initiale
        x = F.relu(self.input_proj(x))

        # Traitement par blocks
        for i in range(self.num_blocks):
            # Sauvegarde pour les connexions résiduelles
            if self.residual_connections and i > 0:
                res = x

            # Appliquer RGCN
            x = self.rgcn_blocks[i](x, edge_index, edge_type)

            # Activer avec ReLU sauf pour le dernier bloc
            if i < self.num_blocks - 1:
                x = F.relu(x)

            # Appliquer l'attention
            x = x.unsqueeze(0)  # Ajouter dimension batch
            x, _ = self.attention_blocks[i](x, x, x)
            x = x.squeeze(0)  # Supprimer dimension batch

            # Appliquer layer norm si activé
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            # Appliquer dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Ajouter connexion résiduelle si ce n'est pas le premier block
            if self.residual_connections and i > 0:
                # S'assurer que les dimensions correspondent
                if res.shape == x.shape:
                    x = x + res

        # Projection finale vers la dimension d'origine
        x = self.output_proj(x)

        # Normaliser les embeddings pour de meilleures similarités cosinus
        x = F.normalize(x, p=2, dim=1)

        return x


import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import matplotlib.patches as mpatches


def plot_loss_curve(losses, total_epochs, eval_every, save_path=None):
    """
    Affiche la courbe de loss à la fin de l'entraînement
    """
    if len(losses) < 2:
        return

    # Créer la figure
    plt.figure(figsize=(12, 8))

    # Subplot 1: Courbe de loss principale
    plt.subplot(2, 2, 1)
    epochs_range = range(1, len(losses) + 1)

    # Tracer la courbe complète
    plt.plot(epochs_range, losses, 'b-', linewidth=2, alpha=0.7, label='Loss')

    # Ajouter une moyenne mobile pour lisser la courbe
    if len(losses) > 10:
        window_size = min(20, len(losses) // 4)
        moving_avg = []
        for i in range(len(losses)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(losses), i + window_size // 2)
            moving_avg.append(sum(losses[start_idx:end_idx]) / (end_idx - start_idx))

        plt.plot(epochs_range, moving_avg, 'r-', linewidth=3, alpha=0.8, label=f'Moyenne mobile ({window_size})')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Zoom sur les dernières époques
    plt.subplot(2, 2, 2)
    if len(losses) > 20:
        recent_losses = losses[-20:]
        recent_epochs = list(range(len(losses) - 19, len(losses) + 1))
        plt.plot(recent_epochs, recent_losses, 'g-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Époque')
        plt.ylabel('Loss')
        plt.title('Zoom: 20 last epoch')
        plt.grid(True, alpha=0.3)

    # Subplot 3: Statistiques de convergence
    plt.subplot(2, 2, 3)
    if len(losses) > 5:
        # Calculer la dérivée (taux de changement)
        derivatives = [losses[i] - losses[i - 1] for i in range(1, len(losses))]
        der_epochs = range(2, len(losses) + 1)

        plt.plot(der_epochs, derivatives, 'purple', linewidth=2, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Δ Loss')
        plt.title('Loss Change Rate')
        plt.grid(True, alpha=0.3)

    # Subplot 4: Informations textuelles
    plt.subplot(2, 2, 4)
    plt.axis('off')

    # Calculer des statistiques
    if losses:
        current_loss = losses[-1]
        min_loss = min(losses)
        max_loss = max(losses)
        final_epoch = len(losses)

        # Amélioration globale
        improvement = "N/A"
        if len(losses) > 1:
            initial_loss = losses[0]
            improvement = f"{((initial_loss - current_loss) / initial_loss * 100):.2f}%"

        # Texte d'information
        info_text = f"""
final statistiques   of training:

 Total epoch: {final_epoch}/{total_epochs}
  final loss: {current_loss:.6f}
 minimale  Loss: {min_loss:.6f}
maximale  Loss  : {max_loss:.6f}

 Overall Improvement: {improvement}
 Évaluation of all  {eval_every} epochs
        """

        plt.text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))

    plt.tight_layout()

    # Sauvegarder si un chemin est fourni
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Afficher
    plt.show()


# Fonction d'entraînement améliorée avec affichage de la courbe à la fin
def train_rgcn_improved(model, data, num_epochs=100, patience=10, lr=0.001, device='cpu'):
    """
    Entraîne le modèle RGCN avec un pipeline amélioré:
    - Scheduler de taux d'apprentissage
    - Perte combinée (MSE + triplet loss)
    - Évaluation régulière
    - Early stopping amélioré
    - Affichage de la courbe de loss à la fin de l'entraînement
    """
    add_log("\n 	■  Entraînement amélioré du modèle RGCN:***")

    # Définir l'optimiseur
    #sert à éviter l’overfitting (régularisation L2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Scheduler pour réduire le taux d'apprentissage si le loss nameliore pas
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Fonctions de perte
    mse_loss = torch.nn.MSELoss()
    cos_sim = torch.nn.CosineSimilarity(dim=1)

    # Fonction de triplet loss pour améliorer la qualité des embeddings
    #margin seul de min et max loss triples
    def triplet_loss(embeddings, edge_index, margin=0.3):
        # Sélection aléatoire de triplets (ancre, positif, négatif)
        src, dst = edge_index

        # Sélectionner des ancres et positifs parmi les arêtes existantes
        batch_size = min(1000, len(src))
        indices = torch.randint(0, len(src), (batch_size,), device=device)

        anchors = embeddings[src[indices]]
        positives = embeddings[dst[indices]]

        # Sélectionner des négatifs aléatoires
        neg_indices = torch.randint(0, embeddings.shape[0], (batch_size,), device=device)
        negatives = embeddings[neg_indices]

        # Calculer les similarités
        pos_sim = cos_sim(anchors, positives)
        neg_sim = cos_sim(anchors, negatives)

        # Triplet loss: max(0, neg_sim - pos_sim + margin)
        loss = torch.relu(neg_sim - pos_sim + margin)

        return loss.mean()

    # Paramètres d'entraînement
    best_loss = float('inf')
    patience_counter = 0 #Sert à compter combien de fois de suite la perte n’a pas amélioré.
    eval_every = max(1, num_epochs // 20)  # ~20 évaluations pendant l'entraînement evaluation of the model tous les 5 epoch

    # Facteur de pondération pour la Triplet Loss
    triplet_weight = 0.3

    # Historique de perte
    losses = []

    # Boucle d'entraînement
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calcul des pertes
        reconstruction_loss = mse_loss(output, data.x)
        embedding_loss = triplet_loss(output, data.edge_index)

        # Perte combinée
        loss = reconstruction_loss + triplet_weight * embedding_loss

        # Backward pass
        loss.backward()

        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Enregistrer la perte
        losses.append(loss.item())

        # Évaluer et afficher la progression
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            add_log(f" 	■  Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}, "
                    f"MSE: {reconstruction_loss.item():.6f}, Triplet: {embedding_loss.item():.6f}, "
                    f"LR: {current_lr:.1e}")

            # Mettre à jour le scheduler
            scheduler.step(loss)

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                add_log(f" --- Early stopping à l'epoch {epoch + 1}")
                # Restaurer le meilleur modèle
                model.load_state_dict(best_model_state)
                break

    # Afficher la courbe de loss à la fin de l'entraînement
    plot_loss_curve(losses, num_epochs, eval_every, save_path="final_loss_curve.png")

    # Retourner l'historique des pertes
    return losses, model


# Étape 6 : Évaluation du modèle RGCN avec métriques et visualisation
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os
from datetime import datetime


def evaluate_model(model, data, entity_to_idx, sample_size=1000):
    """
    Évalue le modèle avec diverses métriques, optimisé pour les grands graphes
    Inclut la visualisation des courbes Precision@k et Recall@k
    """
    model.eval()
    with torch.no_grad():
        # Utiliser un échantillon pour les grands graphes
        use_sampling = len(entity_to_idx) > sample_size
        if use_sampling:
            print(f"\n  ÉEvaluation on a sample of {sample_size} entities (in {len(entity_to_idx)} in total)")
            entity_indices = random.sample(range(len(entity_to_idx)), sample_size)
        else:
            print(f"\n  Evaluation on all the {len(entity_to_idx)} entities")
            entity_indices = range(len(entity_to_idx))

        # Obtenir les embeddings mis à jour
        output = model(data)
        output_np = output.cpu().numpy()

        # Prepare data for metric calculation
        # Si échantillonnage, ne prendre que les entités échantillonnées
        if use_sampling:
            y_true = (data.x[entity_indices].cpu().numpy() > 0).astype(int)
            y_pred = (output_np[entity_indices] > 0).astype(int)
        else:
            y_true = (data.x.cpu().numpy() > 0).astype(int)
            y_pred = (output_np > 0).astype(int)

        # Calculate Mean Reciprocal Rank (MRR)
        mrr = calculate_mrr(y_true, y_pred)
        add_log(f" ✔Mean Reciprocal Rank (MRR): {mrr:.4f}")

        # Calculate Precision@k and Recall@k pour différentes valeurs de k
        k_values = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]
        precision_values = []
        recall_values = []

        add_log("\n **********Metric computation for different values of k...*****************")
        for k in k_values:
            precision_at_k, recall_at_k = calculate_precision_recall_at_k(y_true, y_pred, k=k)
            precision_values.append(precision_at_k)
            recall_values.append(recall_at_k)
            if k in [5, 10, 20]:  # Afficher seulement quelques valeurs clés
                add_log(f"✔ Precision@{k}: {precision_at_k:.4f}, Recall@{k}: {recall_at_k:.4f}")

        # Calculate Mean Average Precision (MAP)
        map_score = calculate_map(y_true, y_pred)
        add_log(f"✔ Mean Average Precision (MAP): {map_score:.4f}")

        # Calculate nDCG@k pour différentes valeurs de k
        ndcg_values = []
        add_log("\n  ************ nDCG metric computation for different values of k...********")
        for k in k_values:
            ndcg_at_k = calculate_ndcg_at_k(y_true, y_pred, k=k)
            ndcg_values.append(ndcg_at_k)
            if k in [5, 10, 20]:  # Afficher seulement quelques valeurs clés
                add_log(f" ✔ nDCG@{k}: {ndcg_at_k:.4f}")

        # Créer la courbe Precision@k, Recall@k et nDCG@k
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"precision_recall_ndcg_curves_{len(entity_to_idx)}_entities_{timestamp}.png"
        create_precision_recall_ndcg_curves(k_values, precision_values, recall_values,
                                            ndcg_values, save_path=filename)

        return map_score


def create_precision_recall_ndcg_curves(k_values, precision_values, recall_values, ndcg_values,
                                        save_path="precision_recall_ndcg_curves.png"):
    """
    Crée, affiche et sauvegarde les courbes de Precision@k, Recall@k et nDCG@k
    """
    plt.figure(figsize=(12, 6))

    # Graphique unique: Precision@k, Recall@k et nDCG@k
    plt.plot(k_values, precision_values, 'b-o', label='Precision@k', linewidth=2, markersize=6)
    plt.plot(k_values, recall_values, 'r-s', label='Recall@k', linewidth=2, markersize=6)
    plt.plot(k_values, ndcg_values, 'g-^', label='nDCG@k', linewidth=2, markersize=6)
    plt.xlabel('k', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Precision@k, Recall@k et nDCG@k', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(min(k_values), max(k_values))

    # Ajouter des annotations pour quelques points clés
    for i, k in enumerate([5, 10, 20]):
        if k in k_values:
            idx = k_values.index(k)
            plt.annotate(f'P@{k}={precision_values[idx]:.3f}',
                         xy=(k, precision_values[idx]),
                         xytext=(k + 2, precision_values[idx] + 0.05),
                         fontsize=9, alpha=0.8)
            plt.annotate(f'R@{k}={recall_values[idx]:.3f}',
                         xy=(k, recall_values[idx]),
                         xytext=(k + 2, recall_values[idx] - 0.05),
                         fontsize=9, alpha=0.8)
            plt.annotate(f'nDCG@{k}={ndcg_values[idx]:.3f}',
                         xy=(k, ndcg_values[idx]),
                         xytext=(k + 2, ndcg_values[idx]),
                         fontsize=9, alpha=0.8)

    plt.tight_layout()

    # Sauvegarder la figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    add_log(f" ✔  Courbes saved in: {save_path}")

    plt.show()
    plt.close()  # Fermer la figure pour libérer la mémoire

    # Afficher un résumé des résultats
    add_log(f"\n ******************* Summary of  metrics:***********************")
    print(
        f" --Best Precision@k: {max(precision_values):.4f} (k={k_values[precision_values.index(max(precision_values))]})")
    print(f" -- Best Recall@k: {max(recall_values):.4f} (k={k_values[recall_values.index(max(recall_values))]})")
    print(f" -- Best nDCG@k: {max(ndcg_values):.4f} (k={k_values[ndcg_values.index(max(ndcg_values))]})")


def calculate_mrr(y_true, y_pred, max_rank=100):
    """Calcule Mean Reciprocal Rank (MRR) optimisé pour les grands graphes"""
    batch_size = 1000
    ranks = []

    for i in range(0, min(y_true.shape[0], 5000), batch_size):  # Limiter à 5000 éléments pour les grands graphes
        batch_end = min(i + batch_size, y_true.shape[0])
        for j in range(i, batch_end):
            # Sort predictions for the current entity and get indices
            ranked_indices = np.argsort(y_pred[j, :])[::-1][:max_rank]  # Ne prendre que les top max_rank
            # Check if the true entity is in the ranked indices
            true_relevant = np.where(y_true[j, :] > 0)[0]

            if len(true_relevant) > 0:
                # Trouver le premier rang pertinent
                min_rank = float('inf')
                for rel_idx in true_relevant:
                    if rel_idx in ranked_indices:
                        rank = np.where(ranked_indices == rel_idx)[0][0] + 1
                        min_rank = min(min_rank, rank)

                if min_rank < float('inf'):
                    ranks.append(1 / min_rank)
                else:
                    ranks.append(0)
            else:
                ranks.append(0)

    return np.mean(ranks) if ranks else 0


def calculate_precision_recall_at_k(y_true, y_pred, k=10):
    """Calcule Precision@k et Recall@k de manière optimisée"""
    precision_scores = []
    recall_scores = []

    batch_size = 1000
    for i in range(0, min(y_true.shape[0], 5000), batch_size):  # Limiter à 5000 pour les grands graphes
        batch_end = min(i + batch_size, y_true.shape[0])
        for j in range(i, batch_end):
            # Sort predictions for the current entity
            ranked_indices = np.argsort(y_pred[j, :])[::-1][:k]  # Top k prédictions

            # Trouver les éléments pertinents dans la vérité terrain
            relevant_items = np.where(y_true[j, :] > 0)[0]

            # Calculer Precision@k: proportion des prédictions top-k qui sont pertinentes
            if len(ranked_indices) > 0:
                tp = len(set(ranked_indices) & set(relevant_items))
                precision = tp / len(ranked_indices)
            else:
                precision = 0
            precision_scores.append(precision)

            # Calculer Recall@k: proportion des éléments pertinents qui sont dans le top-k
            if len(relevant_items) > 0:
                tp = len(set(ranked_indices) & set(relevant_items))
                recall = tp / len(relevant_items)
            else:
                recall = 0
            recall_scores.append(recall)

    return np.mean(precision_scores), np.mean(recall_scores)


def calculate_ndcg_at_k(y_true, y_pred, k=10):
    """
    Calcule Normalized Discounted Cumulative Gain (nDCG@k) de manière optimisée
    """
    ndcg_scores = []

    batch_size = 1000
    for i in range(0, min(y_true.shape[0], 5000), batch_size):  # Limiter à 5000 pour les grands graphes
        batch_end = min(i + batch_size, y_true.shape[0])
        for j in range(i, batch_end):
            # Obtenir les indices triés par score de prédiction (ordre décroissant)
            ranked_indices = np.argsort(y_pred[j, :])[::-1][:k]

            # Calculer DCG@k
            dcg = 0
            for rank, idx in enumerate(ranked_indices):
                # Gain = 1 si l'élément est pertinent, 0 sinon
                gain = 1 if y_true[j, idx] > 0 else 0
                # DCG = somme des gains pondérés par log2(rang+2)
                dcg += gain / np.log2(rank + 2)

            # Calculer IDCG@k (DCG idéal)
            # Trier les vraies valeurs par ordre décroissant pour obtenir le meilleur DCG possible
            true_sorted = np.sort(y_true[j, :])[::-1][:k]
            idcg = 0
            for rank, gain in enumerate(true_sorted):
                if gain > 0:
                    idcg += gain / np.log2(rank + 2)

            # nDCG@k = DCG@k / IDCG@k
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0

            ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0


def calculate_map(y_true, y_pred, max_k=100):
    """Calcule Mean Average Precision (MAP) de manière optimisée"""
    ap_scores = []

    batch_size = 10000
    for i in range(0, min(y_true.shape[0], 5000), batch_size):  # Limiter pour les grands graphes
        batch_end = min(i + batch_size, y_true.shape[0])
        for j in range(i, batch_end):
            # Obtenir les prédictions triées
            ranked_indices = np.argsort(y_pred[j, :])[::-1][:max_k]

            # Récupérer les valeurs pertinentes
            relevant_items = set(np.where(y_true[j, :] > 0)[0])

            if not relevant_items:
                ap_scores.append(0)
                continue

            # Calculer AP
            precisions = []
            num_correct = 0

            for k, idx in enumerate(ranked_indices):
                if idx in relevant_items:
                    num_correct += 1
                    precisions.append(num_correct / (k + 1))

            if precisions:
                ap_scores.append(sum(precisions) / len(relevant_items))
            else:
                ap_scores.append(0)

    return np.mean(ap_scores) if ap_scores else 0
# Fonction pour exporter les embeddings finaux
def export_embeddings(embeddings, entity_to_idx, filename="embeddings_rgcn_rdf2vec.csv", return_dict=True):
    """
    Exporte les embeddings finaux dans un fichier CSV et optionnellement retourne un dictionnaire ou DataFrame

    Args:
        embeddings (numpy.ndarray): Tableau des embeddings
        entity_to_idx (dict): Dictionnaire mappant les entités à leurs indices
        filename (str, optional): Nom du fichier CSV de sortie. Defaults to "embeddings_rgcn_rdf2vec.csv".
        return_dict (bool, optional): Si True, retourne un dictionnaire. Si False, retourne un DataFrame. Defaults to True.

    Returns:
        dict or pandas.DataFrame: Dictionnaire ou DataFrame des embeddings
    """
    try:
        df_data = []
        embeddings_dict = {}

        for entity, idx in entity_to_idx.items():
            # Récupérer l'embedding pour cette entité
            embedding_values = embeddings[idx]

            # Créer un dictionnaire pour chaque entité avec ses dimensions
            embedding_dict = {f'dim_{i}': val for i, val in enumerate(embedding_values)}

            # Ajouter à un dictionnaire global
            embeddings_dict[entity] = embedding_dict

            # Préparer les données pour le DataFrame
            row = {'entity': entity, **embedding_dict}
            df_data.append(row)

        # Créer le DataFrame et exporter
        import pandas as pd
        df = pd.DataFrame(df_data)
        df.to_csv(filename, index=False)

        add_log(f"\n ✔ Embeddings successfully exported to '{filename}' ")
        add_log(f"   -- Number of entities : {len(df)}")
        add_log(f"   -- Dimensions: {len(df.columns) - 1}")

        # Retourner soit le dictionnaire, soit le DataFrame selon le paramètre
        return embeddings_dict if return_dict else df

    except Exception as e:
        add_log(f"\n  Error while exporting embeddings: {str(e)}")
        return None



import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



class SemanticQueryParser:
    """
    Classe pour analyser sémantiquement une requête sans utiliser d'embeddings.
    Effectue la tokenisation, supprime les mots vides, et détermine l'intention et les entités cibles.
    """

    def __init__(self, download_nltk=True):
        """
        Initialise le parser avec les ressources NLTK nécessaires
        """
        if download_nltk:
            try:
                # Check if punkt is available (often includes punkt_tab)
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                add_log(" - Téléchargement des ressources NLTK nécessaires...")
                nltk.download('punkt')  # Download punkt if not found

            try:
                # Explicitly check and download punkt_tab if needed
                # punkt_tab is required by PunktSentenceTokenizer which is used by word_tokenize
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                add_log(" - Téléchargement spécifique de 'punkt_tab'...")
                nltk.download('punkt_tab')  # Download punkt_tab specifically

            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                add_log("⏳ Téléchargement des mots vides...")
                nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english') + stopwords.words('french'))

        # Dictionnaire des mots-clés pour l'analyse sémantique par catégorie
        # Ajout de mots-clés liés aux médicaments pour les requêtes DrugBank
        # Dans l'initialisation de la classe SemanticQueryParser
        self.semantic_keywords = {
            "intent": {
                "query": ["what", "how", "tell", "find", "search", "where", "who", "which"],
                "list": ["list", "show", "all", "give"],
                "comparison": ["compare", "difference", "between", "versus", "vs"],
                "relation": ["related", "connection", "linked", "associated"],
                "interaction": ["interact", "interaction", "interacts", "combine", "combination"]
            },
            "entities": {
                "disease": ["disease", "illness", "condition", "disorder", "pathology", "syndrome"],
                "drug": ["drug", "medication", "medicine", "treatment", "prescription", "pill", "tablet", "drugbank",
                         "db"],
                "code": ["code", "icd", "icd9", "icd-9", "identifier"],
                "category": ["category", "type", "classification", "class", "group"],
                "name": ["name", "label", "called", "title", "term", "corresponding"],
                "description": ["description", "about", "info", "information", "details", "summary" ,"describe" , "explain","tell" ],
                "vaccine": ["vaccine", "vaccination", "rotavirus", "immunization"]

            }
        }

        add_log(" ✔ SemanticQueryParser successfully initialized ")

    # Méthode pour extraire les identifiants DrugBank
    def extract_drugbank_ids(self, tokens):
        """
        Extrait les identifiants DrugBank potentiels (format: DB suivi de chiffres)
        """
        add_log("\n ::::: Step 2b: Extraction of DrugBank identifiers:::::")

        drugbank_ids = []

        # Recherche des identifiants DB potentiels (DB suivi de chiffres)
        pattern = r'DB\d+'

        for token in tokens["all_tokens"]:
            if re.match(pattern, token, re.IGNORECASE):
                drugbank_ids.append(token.upper())  # Normaliser en majuscules
                add_log(f"   Potential DrugBank ID found: {token.upper()}")

        # Recherche de mentions explicites comme "drugbank:DB00530"
        for i, token in enumerate(tokens["all_tokens"]):
            if token.lower() == "drugbank" and i + 1 < len(tokens["all_tokens"]):
                next_token = tokens["all_tokens"][i + 1]
                if next_token.startswith(":"):
                    next_token = next_token[1:]  # Enlever le ":"
                if re.match(r'DB\d+', next_token, re.IGNORECASE):
                    drugbank_id = next_token.upper()
                    if drugbank_id not in drugbank_ids:
                        drugbank_ids.append(drugbank_id)
                        add_log(f"  -- Explicit DrugBank ID found: {drugbank_id}")

        if not drugbank_ids:
            add_log("   Aucun ID DrugBank trouvé")

        return drugbank_ids

    def clean_text(self, text):
        """
        Nettoie le texte en supprimant la ponctuation sauf les points dans les nombres décimaux
        """
        text = text.lower()

        # Remplacer les points non décimaux par des espaces
        text = re.sub(r'(?<!\d)\.(?!\d)', ' ', text)  # Supprime les points qui ne sont pas entre des chiffres
        text = re.sub(r'[^\w\s\.]', ' ', text)  # Supprime toute autre ponctuation sauf les points

        text = re.sub(r'\s+', ' ', text).strip()  # Normaliser les espaces
        return text

    def tokenize(self, query):
        """
        Tokenise la requête et supprime les mots vides
        """
        add_log("\n :::: Step 1: Query tokenization::::")
        add_log(f"  -- Original query: '{query}'")

        # Nettoyage du texte
        clean_query = self.clean_text(query)
        add_log(f" --  Cleaned query: '{clean_query}'")

        # Tokenisation
        # Use word_tokenize which relies on PunktSentenceTokenizer and thus punkt_tab
        tokens = word_tokenize(clean_query)
        add_log(f" --  Obtained tokens: {tokens}")

        # Suppression des mots vides
        meaningful_tokens = [token for token in tokens if token not in self.stop_words]
        add_log(f" --  Tokens after stopword removal: {meaningful_tokens}")

        return {"all_tokens": tokens, "meaningful_tokens": meaningful_tokens}

    def extract_numeric_entities(self, tokens):
        """
        Extrait les entités numériques comme les codes ICD
        """
        add_log("\n  :::: Step 2: Extraction of numerical entities::::")

        numeric_entities = []

        # Recherche des codes potentiels (nombres, avec éventuellement des points)
        pattern = r'\d+(\.\d+)?'

        for token in tokens["all_tokens"]:
            if re.match(pattern, token):
                numeric_entities.append(token)
                add_log(f" --  Potential code found: {token}")

        if not numeric_entities:
            add_log("  -- No numerical entity found")

        return numeric_entities

    def determine_intent(self, tokens):
        """
        Détermine l'intention principale de la requête
        """
        add_log("\n  ::::: Step 3: Intent determination:::::")

        intent_scores = {intent: 0 for intent in self.semantic_keywords["intent"]}

        # Compter les occurrences de mots-clés d'intention
        for token in tokens["all_tokens"]:
            for intent, keywords in self.semantic_keywords["intent"].items():
                if token in keywords:
                    intent_scores[intent] += 1

        # Trouver l'intention avec le score le plus élevé
        top_intent = max(intent_scores, key=intent_scores.get)

        if intent_scores[top_intent] > 0:
            add_log(f" --  Main intent detected: {top_intent} (score: {intent_scores[top_intent]})")
        else:
            add_log(" --  No clear intent detected, using default 'query")
            top_intent = "query"

        return top_intent

    def identify_entity_types(self, tokens):
        """
        Identifie les types d'entités mentionnés dans la requête
        """
        add_log("\n ::::: Step 4: Identification of entity types:::::")

        entity_scores = {entity: 0 for entity in self.semantic_keywords["entities"]}

        # Compter les occurrences de mots-clés d'entités
        for token in tokens["all_tokens"]:
            for entity, keywords in self.semantic_keywords["entities"].items():
                if token in keywords:
                    entity_scores[entity] += 1

        # Trouver les types d'entités avec un score > 0
        mentioned_entities = {entity: score for entity, score in entity_scores.items() if score > 0}

        if mentioned_entities:
            add_log(f" --  Detected entity types: {mentioned_entities}")
        else:
            add_log(" --  No specific entity type detected")

        return mentioned_entities

    def search_in_embeddings(self, query_analysis, embeddings_dict, combined_graph=None):
        """
        Recherche les cibles de la requête dans le dictionnaire des embeddings
        et extrait les triplets associés à l'entité la plus similaire

        Args:
            query_analysis (dict): Résultat de l'analyse de la requête
            embeddings_dict (dict): Dictionnaire contenant les embeddings exportés
            combined_graph (rdflib.Graph, optional): Le graphe RDF combiné et enrichi

        Returns:
        dict: Résultats trouvés avec leurs embeddings et les triplets associés
        """
        add_log("\n :::: Search in the embeddings...::::")

        search_results = {}
        top_entity = None
        top_similarity = -1

        # Si aucune cible n'a été identifiée, le notifier
        if not query_analysis["search_targets"]:
            add_log("  --  No search target identified")
            return search_results

        # Pour chaque cible de recherche
        for target in query_analysis["search_targets"]:
            add_log(f"  --  Search for: {target}")

            # Extraire le type et la valeur (ex: "code:225.0" -> type="code", value="225.0")
            if ":" in target:
                target_type, target_value = target.split(":", 1)
            else:
                target_type, target_value = "entity", target

            # Si c'est un code ICD, formater selon le format spécifique
            if target_type == "code":
                formatted_target = f"http://purl.bioontology.org/ontology/ICD9CM/{target_value}"
                add_log(f" --   Formatted ICD code: {formatted_target}")

                # Rechercher le code ICD formaté dans les embeddings
                if formatted_target in embeddings_dict:
                    search_results[formatted_target] = embeddings_dict[formatted_target]
                    add_log(f"  --  ICD code found in the embeddings")
                    # Garder la trace de l'entité trouvée
                    top_entity = formatted_target
                    # Afficher les 3 premières dimensions de l'embedding pour illustration
                    preview = {k: v for k, v in list(embeddings_dict[formatted_target].items())[:3]}
                    add_log(f" --   Embedding overview: {preview}...")
                else:
                    add_log(f"  --  ICD code not found in the embeddings")

                    # Recherche alternative avec une correspondance partielle
                    alternative_found = False
                    for key in embeddings_dict:
                        if target_value in key:
                            search_results[key] = embeddings_dict[key]
                            add_log(f"  --  Alternative found: {key}")
                            top_entity = key
                            alternative_found = True
                            break

                    if not alternative_found:
                        add_log(f" --   No alternative found for{target_value}")

            # Si c'est un ID DrugBank, formater selon le format spécifique
            elif target_type == "drugbank":
                formatted_target = f"http://bio2rdf.org/drugbank:{target_value}"
                add_log(f"  --  Formatted DrugBank ID: {formatted_target}")

                # Rechercher l'ID DrugBank formaté dans les embeddings
                if formatted_target in embeddings_dict:
                    search_results[formatted_target] = embeddings_dict[formatted_target]
                    add_log(f"  --  DrugBank ID found in the embeddings")
                    # Garder la trace de l'entité trouvée
                    top_entity = formatted_target
                    # Afficher les 3 premières dimensions de l'embedding pour illustration
                    preview = {k: v for k, v in list(embeddings_dict[formatted_target].items())[:3]}
                    add_log(f"  --  Embedding preview: {preview}...")
                else:
                    add_log(f"  --  DrugBank ID not found in the embeddings")

                    # Recherche alternative avec une correspondance partielle
                    alternative_found = False
                    for key in embeddings_dict:
                        if target_value in key:
                            search_results[key] = embeddings_dict[key]
                            add_log(f"  --  Alternative found: {key}")
                            top_entity = key
                            alternative_found = True
                            break

                    if not alternative_found:
                        add_log(f"  --  No alternative found for {target_value}")

            # NOUVEAU: Si c'est un nom de médicament pour interaction vaccinale
            elif target_type == "drug_name":
                formatted_target = f"http://pdd.wangmengsd.com/namedrug/{target_value}"
                add_log(f"  --  Formatted drug name: {formatted_target}")

                # Rechercher le nom de médicament formaté dans les embeddings
                if formatted_target in embeddings_dict:
                    search_results[formatted_target] = embeddings_dict[formatted_target]
                    add_log(f"  --  Drug name found in the embeddings")
                    top_entity = formatted_target
                    preview = {k: v for k, v in list(embeddings_dict[formatted_target].items())[:3]}
                    add_log(f"  --  Embedding overview: {preview}...")
                else:
                    add_log(f" --   Drug name not found in the embeddings")

                    # Recherche alternative avec une correspondance partielle
                    alternative_found = False
                    for key in embeddings_dict:
                        if target_value.lower() in key.lower():
                            search_results[key] = embeddings_dict[key]
                            add_log(f"  --  Alternative found: {key}")
                            top_entity = key
                            alternative_found = True
                            break

                    if not alternative_found:
                        add_log(f"  --  No alternative found for {target_value}")

            # Si c'est une autre entité (maladie, etc.), rechercher directement
            else:
                if target_value in embeddings_dict:
                    search_results[target_value] = embeddings_dict[target_value]
                    add_log(f"  --  Entity found in the embeddings")
                    top_entity = target_value
                    # Afficher les 3 premières dimensions de l'embedding pour illustration
                    preview = {k: v for k, v in list(embeddings_dict[target_value].items())[:3]}
                    add_log(f"  --  Embedding overview: {preview}...")
                else:
                    add_log(f"  --  Entity not found in the embeddings")

                    # Recherche alternative
                    for key in embeddings_dict:
                        if target_value in key:
                            search_results[key] = embeddings_dict[key]
                            add_log(f"  --  Alternative found: {key}")
                            top_entity = key
                            break

        # Résumé des résultats
        if search_results:
            add_log(f"\n ✔ Results found: {len(search_results)} entity")

            # Calculer la similarité avec les autres entités si des résultats ont été trouvés
            similarity_results = self.calculate_similarity_hnsw(search_results, embeddings_dict, top_k=5)

            # Récupérer l'entité avec la plus haute similarité
            if similarity_results and len(similarity_results) > 0:
                top_entity = similarity_results[0]["entity"]
                top_similarity = similarity_results[0]["score"]
                add_log(f"\n  -- Most similar entity: {top_entity} (score: {top_similarity:.4f})")

                # Si le graphe RDF est fourni, extraire les triplets associés
                if combined_graph is not None and top_entity:
                    related_triplets = self.extract_related_triplets(top_entity, combined_graph)
                    search_results["top_entity"] = top_entity
                    search_results["top_similarity"] = top_similarity
                    search_results["related_triplets"] = related_triplets
                else:
                    add_log("\n -- RDF graph not provided, unable to extract associated triples")
        else:
            add_log("\n  -- No result found in the embeddings")

        return search_results

    def calculate_similarity_hnsw(self, query_embeddings, embeddings_dict, top_k=5):
        """
        Calcule la similarité cosinus entre les embeddings de la requête et tous les embeddings
        du dictionnaire en utilisant l'algorithme HNSW pour une recherche efficace des plus proches voisins.
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        import hnswlib
        import matplotlib.pyplot as plt
        import traceback

        add_log("\n ::::: Similarity computation with HNSW...:::::")

        # Vérifier si les entrées sont valides
        if not query_embeddings or not embeddings_dict:
            add_log("  --  Invalid embeddings for similarity computation")
            return []

        # Extraire l'entité de requête et son embedding
        query_entity = list(query_embeddings.keys())[0]
        query_embedding_dict = query_embeddings[query_entity]

        # Convertir le dictionnaire d'embedding en array numpy
        query_embedding = np.array([value for key, value in sorted(query_embedding_dict.items())])

        add_log(f" --   Query: {query_entity}")
        add_log(f"  --  Embedding dimensions: {len(query_embedding)}")

        # Préparer la liste des entités et leurs embeddings
        entity_list = []
        embeddings_list = []

        # Convertir tous les embeddings du dictionnaire en arrays numpy
        for entity, embedding_dict in embeddings_dict.items():
            entity_list.append(entity)
            # Trier les dimensions pour s'assurer qu'elles sont dans le bon ordre
            embedding_array = np.array([value for key, value in sorted(embedding_dict.items())])
            embeddings_list.append(embedding_array)

        # Convertir la liste d'embeddings en un tableau 2D numpy
        all_embeddings = np.vstack(embeddings_list)

        # Nombre total d'entités
        num_elements = len(entity_list)
        dim = len(query_embedding)

        add_log(f"  --  HNSW index construction for {num_elements} entities...")

        # Créer l'index HNSW
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16, random_seed=42)
        index.add_items(all_embeddings, np.arange(num_elements))
        index.set_ef(50)

        add_log(" --   Search for the most similar entities...")

        # Rechercher les k plus proches voisins
        labels, distances = index.knn_query(query_embedding.reshape(1, -1), k=top_k)

        # Préparer les résultats
        results = []

        add_log("\n *********** Top most similar entities:*********")
        for idx, (label, distance) in enumerate(zip(labels[0], distances[0])):
            # Convertir la distance en score de similarité cosinus (1 - distance)
            similarity_score = 1 - distance
            entity = entity_list[label]

            results.append({
                "entity": entity,
                "score": similarity_score,
                "rank": idx + 1
            })

            # Afficher les résultats avec plus de précision
            add_log(f" --  #{idx + 1}: {entity} (score: {similarity_score:.6f})")

            # Afficher un aperçu de l'embedding (3 premières dimensions)
            embedding_preview = {k: v for k, v in list(embeddings_dict[entity].items())[:3]}
            add_log(f"   --    Overview: {embedding_preview}...")

        add_log(f"\n ✔ HNSW search completed successfully! ({len(results)} results) ")

        # ======================== BAR PLOT CORRIGÉ ========================
        try:
            add_log("\n -- Generation of the similarity bar plot...")

            # FORCER L'AFFICHAGE DES LOGS AVANT LE PLOT
            import sys
            sys.stdout.flush()

            # Extraire les données pour le graphique
            entities = [result["entity"] for result in results]
            scores = [result["score"] for result in results]

            # Créer des labels plus courts
            short_labels = []
            for i, entity in enumerate(entities):
                if len(entity) > 50:
                    if '/' in entity:
                        short_label = entity.split('/')[-1]
                    elif ':' in entity:
                        short_label = entity.split(':')[-1]
                    else:
                        short_label = entity[:30] + "..."
                else:
                    short_label = entity
                short_labels.append(f"#{i + 1}: {short_label}")

            # FERMER TOUTES LES FIGURES PRÉCÉDENTES
            plt.close('all')

            # Configuration du graphique
            fig, ax = plt.subplots(figsize=(12, 8))

            # Créer le bar plot avec des couleurs dégradées
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(scores)))
            bars = ax.bar(range(len(scores)), scores, color=colors, alpha=0.8,
                          edgecolor='black', linewidth=0.5)

            # Personnaliser le graphique
            ax.set_xlabel('Entities (Ranked by Similarity)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cosine Similarity Score', fontsize=12, fontweight='bold')

            # Titre avec requête tronquée si nécessaire
            query_display = query_entity[:60] + "..." if len(query_entity) > 60 else query_entity
            ax.set_title(f'Top {top_k} Most Similar Entities\nQuery: {query_display}',
                         fontsize=14, fontweight='bold', pad=20)

            # Définir les positions et labels des x
            ax.set_xticks(range(len(short_labels)))
            ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=10)

            # CORRECTION PRINCIPALE : Ajuster l'axe Y pour montrer les différences
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            # Si les scores sont très proches, zoomer sur la plage de variation
            if score_range < 0.01:  # Si la différence est très petite
                y_min = max(0, min_score - 0.005)  # Laisser un peu d'espace en bas
                y_max = min(1, max_score + 0.005)  # Laisser un peu d'espace en haut
            else:
                y_min = max(0, min_score - score_range * 0.1)
                y_max = min(1, max_score + score_range * 0.1)

            ax.set_ylim(y_min, y_max)

            # CORRECTION : Afficher plus de décimales pour montrer les différences
            for i, (bar, score) in enumerate(zip(bars, scores)):
                height = bar.get_height()
                # Calculer la position du texte en fonction de la nouvelle échelle
                text_y = height + (y_max - y_min) * 0.02

                # Afficher avec plus de précision (5 décimales)
                ax.text(bar.get_x() + bar.get_width() / 2., text_y,
                        f'{score:.5f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

            # Améliorer la mise en page
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # CORRECTION : Formatter l'axe Y pour montrer plus de décimales
            from matplotlib.ticker import FormatStrFormatter
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

            plt.tight_layout()

            # Sauvegarder avec nom unique basé sur la requête et timestamp
            import time
            timestamp = int(time.time())
            query_hash = hash(query_entity) % 10000
            plot_filename = f"similarity_plot_{query_hash}_{timestamp}.png"

            plt.savefig(plot_filename, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')

            add_log(f"  --  Graph saved: {plot_filename}")

            # AFFICHAGE NON-BLOQUANT
            plt.show(block=False)
            plt.draw()

            add_log("  --  Bar plot successfully generated!")
            add_log(f"  --  Score range: {min_score:.6f} - {max_score:.6f}")

            # FORCER L'AFFICHAGE DES LOGS APRÈS LE PLOT
            sys.stdout.flush()

        except Exception as plot_error:
            add_log(f"  -- Erreur lors de la génération du bar plot: {str(plot_error)}")
            add_log(f"  -- Détails: {traceback.format_exc()}")

        return results

    def extract_query_response(self, query_analysis, related_triplets):
        """
        Extrait la réponse pertinente à la requête de l'utilisateur en faisant correspondre
        l'intention et le type d'entité avec les prédicats disponibles dans les triplets.

        Args:
            query_analysis (dict): L'analyse de la requête
            related_triplets (dict): Les triplets associés à l'entité trouvée

        Returns:
             dict: Résultat structuré contenant la réponse à la requête
        """
        add_log("\n  :::::: Extraction of the query response...:::::")

        # Dictionnaire de correspondance entre les intentions/types d'entités et les prédicats
        query_to_predicate_mapping = {
            # Les requêtes sur les catégories de maladies
            "category_disease": {
                "predicates": [
                    "hasDiseaseLabel",
                    "hasCategory",
                    "type",
                    "prefLabel",
                    "broader"
            ],
                "description": "la catégorie de maladie"
        },



            # NOUVEAU: Les requêtes sur les noms de médicaments (DrugBank)
            "drug_name": {
                "predicates": [
                    "name",
                    "label",
                    "title",
                    "prefLabel",
                    "corresponding"
            ],
                "description": "le nom du médicament"
        },
            # NOUVEAU: Les requêtes sur les descriptions de médicaments
            "drug_description": {
                "predicates": [
                    "description",
                    "comment",
                    "abstract",
                    "summary",
                    "describe",
                    "explain",
                    "tell"
            ],
                "description": "la description du médicament"
        },
            "drug_vaccine_interaction": {
            "predicates": [
                "Vaccine_Interactions",
                "Rotavirus_Vaccine_Interactions",
                "vaccine_interaction",
                "interaction",
                "Interactions"
        ],
            "description": "l'interaction vaccinale du médicament"
    },
    }

        # Déterminer le type de requête basé sur l'analyse sémantique
        query_type = None

        # Debug: Afficher les tokens pour diagnostiquer
        add_log(f"   -- Tokens of the query: {query_analysis.get('tokens', {}).get('all_tokens', [])}")
        add_log(f"   -- Intent detected: {query_analysis.get('intent', 'None')}")
        add_log(f"   -- entities types: {query_analysis.get('entity_types', {})}")

        # **CORRECTION PRINCIPALE**: Vérifier d'abord si des interactions vaccinales sont disponibles dans les triplets
        if "vaccine_interaction_info" in related_triplets and related_triplets["vaccine_interaction_info"]:
            add_log(" --   Vaccine interactions detected in the triples - Type: drug_vaccine_interaction")
            query_type = "drug_vaccine_interaction"

        # PRIORITÉ 2: Vérifier si la requête concerne explicitement une interaction vaccinale
        elif (("interaction" in query_analysis.get("intent", "") or
               any(token.lower() in ["interact", "interaction", "interacts", "combine", "combination"]
                   for token in query_analysis.get("tokens", {}).get("all_tokens", []))) and
              ("vaccine" in query_analysis.get("entity_types", {}) or
               any(token.lower() in ["vaccine", "vaccination", "rotavirus", "immunization"]
                   for token in query_analysis.get("tokens", {}).get("all_tokens", [])))):
            add_log(" --   Vaccine interaction keywords detected - Type: drug_vaccine_interaction")
            query_type = "drug_vaccine_interaction"

        # **NOUVELLE CONDITION**: Si c'est un nom de médicament et qu'il y a des interactions vaccinales disponibles
        elif ("drug_names" in query_analysis and query_analysis["drug_names"] and
              "vaccine_interaction_info" in related_triplets and related_triplets["vaccine_interaction_info"]):
            add_log("  --  Drug name with available vaccine interactions - Type: drug_vaccine_interaction")
            query_type = "drug_vaccine_interaction"

        # **CONDITION GÉNÉRIQUE AMÉLIORÉE**: Si l'entité est un médicament et qu'il y a des interactions vaccinales
        elif (any("namedrug" in target for target in query_analysis.get("search_targets", [])) and
              "vaccine_interaction_info" in related_triplets and related_triplets["vaccine_interaction_info"]):
            add_log(" --   Drug entity with vaccine interactions - Type: drug_vaccine_interaction")
            query_type = "drug_vaccine_interaction"






        # Vérifier si la requête concerne explicitement une description de médicament
        elif "drug" in query_analysis.get("entity_types", {}) and "description" in query_analysis.get("entity_types", {}):
            query_type = "drug_description"
        # Vérifier si les mots liés à la description sont présents dans les tokens
        elif any(token.lower() in ["description", "about", "info", "information", "details", "summary"]
                 for token in query_analysis.get("tokens", {}).get("all_tokens", [])):
            query_type = "drug_description"
        # Détecter les requêtes sur les noms de médicaments
        elif "drug" in query_analysis.get("entity_types", {}) and "name" in query_analysis.get("entity_types", {}):
            query_type = "drug_name"
        # Détecter la requête sur la catégorie de maladie
        elif ("category" in query_analysis.get("entity_types", {}) and
              "disease" in query_analysis.get("entity_types", {})):
            query_type = "category_disease"
        # Si la requête contient "symptom" ou synonymes
        elif any(token in ["symptom", "symptoms", "sign", "signs", "exhibit"]
                 for token in query_analysis.get("tokens", {}).get("all_tokens", [])):
            query_type = "symptom_disease"
        # Si la requête contient "treatment", "medication", etc.
        elif any(token in ["treatment", "medication", "medicine", "drug", "cure"]
                 for token in query_analysis.get("tokens", {}).get("all_tokens", [])):
            query_type = "treatment_disease"
        # Si la requête contient "patient", "person", etc.
        elif any(token in ["patient", "person", "individual", "who", "people"]
                 for token in query_analysis.get("tokens", {}).get("all_tokens", [])):
            query_type = "patient_disease"
        # Par défaut pour les médicaments (DrugBank IDs) sans autres indications: nom du médicament
        elif "drugbank_ids" in query_analysis and query_analysis["drugbank_ids"] and query_type is None:
            query_type = "drug_name"
        # Par défaut, si type indéterminé mais maladie présente, on suppose une requête de catégorie
        elif "disease" in query_analysis.get("entity_types", {}) and query_type is None:
            query_type = "category_disease"

        # Résultat par défaut (aucune correspondance trouvée)
        result = {
            "query_type": query_type,
            "found": False,
            "response": None,
            "description": "Information non trouvée"
    }

        # Si type de requête identifié, chercher les triplets correspondants
        if query_type:
            add_log(f"  --  Query type identified: {query_type}")
            add_log(f" --   Description: {query_to_predicate_mapping[query_type]['description']}")

            # Prédicats à rechercher
            target_predicates = query_to_predicate_mapping[query_type]["predicates"]
            response_values = []

            # PRIORITÉ: Chercher d'abord dans les informations d'interaction vaccinale
            if query_type == "drug_vaccine_interaction" and "vaccine_interaction_info" in related_triplets:
                add_log("   -- Search in vaccine interactions...")
                for info in related_triplets["vaccine_interaction_info"]:
                    pred_name = info["property"].split("/")[-1]
                    for target_pred in target_predicates:
                        if target_pred.lower() in pred_name.lower():
                            response_values.append({
                                "predicate": pred_name,
                                "value": info["value"]
                            })
                            add_log(f"  --  Found in vaccine interaction: {pred_name} -> {info['value']}")
                            break  #  Arrêter après la première correspondance
                    if response_values:
                        break



            # Chercher dans les relations sortantes (où l'entité est le sujet)
            if not response_values:  # Seulement si pas encore trouvé
                add_log("  --  Search in triples where the entity is the subject...")
                for triplet in related_triplets.get("as_subject", []):
                    predicate = triplet["predicate"]
                    pred_name = predicate.split("/")[-1]
                    for target_pred in target_predicates:
                        if target_pred.lower() in pred_name.lower():
                            response_values.append({
                                "predicate": pred_name,
                                "value": triplet["object"]
                            })
                            add_log(f"  --  found: {pred_name} -> {triplet['object']}")
                            break
                    if response_values:
                        break

            # Chercher dans les relations entrantes (où l'entité est l'objet)
            # Uniquement pour les requêtes de type patient
            if query_type == "patient_disease":
                add_log("  --  Search in triples where the entity is the object...")
                for triplet in related_triplets.get("as_object", []):
                    predicate = triplet["predicate"]
                    pred_name = predicate.split("/")[-1]
                    for target_pred in target_predicates:
                        if target_pred.lower() in pred_name.lower():
                            response_values.append({
                                "predicate": pred_name,
                                "value": triplet["subject"]
                            })
                            add_log(f"  --  found: {pred_name} -> {triplet['subject']}")
                            break
                    if response_values:
                        break

            # Si des informations ICD spécifiques sont disponibles
            if "icd_info" in related_triplets and not response_values:
                add_log(" --   Search in ICD information...")
                for info in related_triplets["icd_info"]:
                    pred_name = info["property"].split("#")[-1]

                    for target_pred in target_predicates:
                        if target_pred.lower() in pred_name.lower():
                            response_values.append({
                                "predicate": pred_name,
                                "value": info["value"]
                        })
                            add_log(f"  --  Found in ICD info: {pred_name} -> {info['value']}")

            # Si des informations DrugBank spécifiques sont disponibles
            if "drugbank_info" in related_triplets and not response_values:
                add_log("  --  Search in DrugBank information...")
                for info in related_triplets["drugbank_info"]:
                    pred_name = info["property"].split("/")[-1]

                    for target_pred in target_predicates:
                        if target_pred.lower() in pred_name.lower():
                            response_values.append({
                                "predicate": pred_name,
                                "value": info["value"]
                        })
                            add_log(f"  --  Found in DrugBank info: {pred_name} -> {info['value']}")

            # Mettre à jour le résultat avec les valeurs trouvées
            if response_values:
                result["found"] = True
                result["response"] = response_values
                result["description"] = query_to_predicate_mapping[query_type]["description"]
                add_log(f" --   Answer found: {len(response_values)} valeur(s)")
            else:
                add_log(f"  --  No information found for this type of query")
        else:
            add_log(f"  --  Query type not identified")

        # Formuler une réponse en langage naturel
        if result["found"]:
            natural_response = f"For your search regarding {result['description']} "

            # Ajouter le code ICD si disponible
            if query_analysis.get("numeric_entities"):
                natural_response += f"the ICD code{query_analysis['numeric_entities'][0]}, "

            # Ajouter l'ID DrugBank si disponible
            if query_analysis.get("drugbank_ids"):
                natural_response += f"the DrugBank identifier {query_analysis['drugbank_ids'][0]}, "

            # Ajouter les noms de médicaments si disponibles
            if query_analysis.get("drug_names"):
                natural_response += f"of the medication {', '.join(query_analysis['drug_names'])}, "

            # Ajouter les résultats trouvés
            natural_response += f"I found the following information:\n"

            for i, resp in enumerate(result["response"]):
                natural_response += f"- {resp['value']}"
                if i < len(result["response"]) - 1:
                    natural_response += "\n"

            result["natural_response"] = natural_response
        else:
            result["natural_response"] = "I did not find any information matching your query."

        return result

    def extract_related_triplets(self, top_entity, combined_graph):
        """
        Extrait tous les triplets reliés à l'entité ayant la plus haute similarité
        dans le graphe KG enrichi, y compris les interactions vaccinales spécifiques.

        Args:
            top_entity (str): URI de l'entité avec la plus haute similarité
            combined_graph (rdflib.Graph): Le graphe RDF combiné et enrichi

        Returns:
            dict: Dictionnaire contenant les triplets où l'entité apparaît comme sujet ou objet,
              ainsi que les informations spécifiques (DrugBank, ICD, interactions vaccinales)
        """
        from rdflib import URIRef, Literal

        add_log(f"\n   :::::: Extraction of triples for the entity: {top_entity}::::::")

        # Convertir l'entité en URIRef pour la requête SPARQL
        entity_uri = URIRef(top_entity)

        # Dictionnaire pour stocker les triplets
        triplets = {
            "as_subject": [],
            "as_object": [],
            "vaccine_interaction_info": []  # Nouveau: pour stocker les interactions vaccinales
    }

        # 1. Extraire tous les triplets où l'entité est le sujet
        add_log("  --  Search for triples where the entity is the subject...")
        for s, p, o in combined_graph.triples((entity_uri, None, None)):
            # Convertir les URIRef et Literal en chaînes de caractères
            predicate = str(p)
            if isinstance(o, URIRef):
                object_value = str(o)
            else:
                object_value = o.toPython()  # Pour les littéraux

            triplets["as_subject"].append({
                "predicate": predicate,
                "object": object_value
        })

        add_log(f" ✔  found {len(triplets['as_subject'])} triples where the entity is the subject")

        # 2. Extraire tous les triplets où l'entité est l'objet
        add_log(" ::::: Search for triples where the entity is the object...::::::")
        for s, p, o in combined_graph.triples((None, None, entity_uri)):
            # Convertir en chaînes de caractères
            subject = str(s)
            predicate = str(p)

            triplets["as_object"].append({
                "subject": subject,
                "predicate": predicate
        })

        add_log(f"  ✔ found {len(triplets['as_object'])} triples where the entity is the object")

        # 3. Recherche spécifique pour les entités DrugBank
        if "bio2rdf.org/drugbank" in top_entity.lower():
            add_log("  --  DrugBank-type entity detected, searching for specific information...")
            drugbank_id = top_entity.split(":")[-1] if ":" in top_entity else ""

            drug_properties = [
                URIRef("http://example.org/name"),
                URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
                URIRef("http://example.org/description"),
                URIRef("http://www.w3.org/2000/01/rdf-schema#comment")
        ]

            triplets["drugbank_info"] = []
            for prop_uri in drug_properties:
                for s, p, o in combined_graph.triples((entity_uri, prop_uri, None)):
                    value = str(o) if isinstance(o, URIRef) else o.toPython()
                    triplets["drugbank_info"].append({
                        "property": str(p),
                        "value": value
                })

        # 4. NOUVEAU: Recherche spécifique pour les entités de médicaments (interactions vaccinales)
        if "pdd.wangmengsd.com/namedrug" in top_entity.lower():
            add_log("   --  Drug entity detected, searching for vaccine interactions...")
            drug_name = top_entity.split("/")[-1] if "/" in top_entity else ""

            vaccine_interaction_properties = [
                URIRef("http://pdd.wangmengsd.com/property/Rotavirus_Vaccine_Interactions"),
                URIRef("http://pdd.wangmengsd.com/property/Vaccine_Interactions"),
                URIRef("http://example.org/vaccine_interaction")
        ]

            # Rechercher les propriétés spécifiques
            for prop_uri in vaccine_interaction_properties:
                for s, p, o in combined_graph.triples((entity_uri, prop_uri, None)):
                    value = str(o) if isinstance(o, URIRef) else o.toPython()
                    triplets["vaccine_interaction_info"].append({
                        "property": str(p),
                        "value": value
                })
                    add_log(f" --   Vaccine interaction found: {str(p)} -> {value}")

            # Recherche générale des interactions contenant "vaccine" dans le prédicat
            for s, p, o in combined_graph.triples((entity_uri, None, None)):
                predicate_str = str(p).lower()
                if "vaccine" in predicate_str or "interaction" in predicate_str:
                    value = str(o) if isinstance(o, URIRef) else o.toPython()
                    if not any(item["property"] == str(p) and item["value"] == value
                              for item in triplets["vaccine_interaction_info"]):
                        triplets["vaccine_interaction_info"].append({
                            "property": str(p),
                            "value": value
                    })

        # 5. Statistiques et résumé
        total_triplets = (len(triplets["as_subject"]) + len(triplets["as_object"]) +
                         len(triplets.get("drugbank_info", [])) +
                         len(triplets["vaccine_interaction_info"]))

        add_log(f"\n ****** Résumé de l'extraction:******")
        add_log(f"   -- Entity: {top_entity}")
        add_log(f"   -- Total number of triples: {total_triplets}")
        if triplets["vaccine_interaction_info"]:
            add_log(f"   -- Vaccine interactions found: {len(triplets['vaccine_interaction_info'])}")

        return triplets

    def extract_drug_names(self, tokens):
        """
        Extrait les noms de médicaments potentiels de la requête
        """
        add_log("\n :::::: Step 2c: Extraction of drug names::::::")

        drug_names = []

        # Liste de noms de médicaments connus (à étendre selon vos besoins)
        # CORRECTION: Mettre tous les noms en minuscules pour la comparaison
        known_drugs = [
            "methimazole", "linezolid", "theophylline", "fludrocortisone",
            "decitabine", "aldesleukin", "fluocinonide", "prednisone",
            "daunorubicin", "sirolimus", "betamethasone" , "Etoposide", "Beclomethasone_dipropionate"
        ]

        # CORRECTION: Recherche dans TOUS les tokens (pas seulement meaningful_tokens)
        for token in tokens["all_tokens"]:
            token_lower = token.lower()

            # Vérifier si le token correspond à un médicament connu
            if token_lower in known_drugs:
                # CORRECTION: Garder la casse originale ou capitaliser proprement
                drug_name = token.capitalize()  # ou token si vous voulez garder la casse originale
                drug_names.append(drug_name)
                add_log(f" -- Drug name found: {drug_name}")

        # CORRECTION: Ajouter une recherche plus flexible (correspondance partielle)
        if not drug_names:
            add_log(" --  Search by partial match...")
            for token in tokens["all_tokens"]:
                token_lower = token.lower()
                for known_drug in known_drugs:
                    # Correspondance partielle (le token contient le nom du médicament ou vice versa)
                    if (len(token_lower) > 3 and
                            (token_lower in known_drug or known_drug in token_lower)):
                        drug_name = token.capitalize()
                        if drug_name not in drug_names:  # Éviter les doublons
                            drug_names.append(drug_name)
                            add_log(f" --  Drug name found (correspondance partielle): {drug_name}")

        if not drug_names:
            add_log("  --  No recognized drug name found")

        return drug_names

    def parse_query(self, query):
        """
        Analyse complète de la requête pour en extraire le sens, y compris les noms de médicaments
        et les interactions vaccinales.

        Args:
            query (str): La requête de l'utilisateur

        Returns:
            dict: Analyse sémantique complète avec:
            - tokens
            - entités numériques
            - IDs DrugBank
            - noms de médicaments
            - intention
            - types d'entités
            - interprétation
            - cibles de recherche
        """
        add_log("\n ****** Semantic analysis of the query:******", query)

        # Étape 1: Tokenisation et nettoyage
        tokens = self.tokenize(query)

        # Étape 2: Extraction des entités
        numeric_entities = self.extract_numeric_entities(tokens)
        drugbank_ids = self.extract_drugbank_ids(tokens)
        drug_names = self.extract_drug_names(tokens)  # Nouvelle extraction des noms de médicaments

        # Étape 3: Détermination de l'intention
        intent = self.determine_intent(tokens)

        # Étape 4: Identification des types d'entités
        entity_types = self.identify_entity_types(tokens)

        # Construire le résultat final
        query_analysis = {
            "original_query": query,
            "tokens": {
                "all_tokens": tokens["all_tokens"],
                "meaningful_tokens": tokens["meaningful_tokens"]
        },
            "numeric_entities": numeric_entities,
            "drugbank_ids": drugbank_ids,
            "drug_names": drug_names,  # Ajout des noms de médicaments
            "intent": intent,
            "entity_types": entity_types,
            "search_targets": []
    }

        # Interprétation en langage naturel
        interpretation = f"This query is a {intent} "

        if "category" in entity_types:
            interpretation += "concerning the category "
        if "disease" in entity_types:
            interpretation += "of a disease "
        if "drug" in entity_types:
            interpretation += "of a drug "
        if "name" in entity_types and "drug" in entity_types:
            interpretation += "with his name "
        if numeric_entities:
            interpretation += f"with the code '{numeric_entities[0]}' "
        if drugbank_ids:
            interpretation += f"with the DrugBank identifier '{drugbank_ids[0]}' "
        if drug_names:
            interpretation += f"with the drug(s) {', '.join(drug_names)} "

        query_analysis["interpretation"] = interpretation

        # Détermination des cibles de recherche
        search_targets = []

        # Cibles pour les codes ICD
        if numeric_entities and "code" in entity_types:
            search_targets.append(f"code:{numeric_entities[0]}")

        # Cibles pour les IDs DrugBank
        if drugbank_ids:
            search_targets.append(f"drugbank:{drugbank_ids[0]}")

        # NOUVEAU: Cibles pour les noms de médicaments (interactions vaccinales)
        if drug_names and ("interaction" in entity_types or
                          any(token.lower() in ["interact", "interaction", "interacts", "combine"]
                              for token in tokens["all_tokens"])):
            for drug_name in drug_names:
                search_targets.append(f"drug_name:{drug_name}")

        # Ajout des cibles à l'analyse
        query_analysis["search_targets"] = search_targets

        # Log des résultats
        add_log("\n *** ****Overview of the analysis:*******")
        add_log(f" ---  Interprétation: {interpretation}")
        if search_targets:
            add_log(f" ✔   Research targets: {search_targets}")
        else:
            add_log(" --  No research target identified")

        return query_analysis


# Imports nécessaires
import threading
import time
import queue
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory

# Variables globales pour les logs
log_queue = queue.Queue()
log_history = []
log_lock = threading.Lock()  # Pour éviter les conflits d'accès concurrent


def add_log(message, log_type="info"):
    """
    Ajouter un log à la queue et à l'historique de manière thread-safe
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"

    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "type": log_type,
        "formatted_message": formatted_message,
        "datetime": datetime.now().isoformat()
    }

    # Utiliser un verrou pour éviter les conflits
    with log_lock:
        log_queue.put(log_entry)
        log_history.append(log_entry)

        # GARDER TOUS LES LOGS - pas de limitation

    # Afficher aussi dans la console normale
    print(formatted_message)


import re
import numpy as np
from typing import List, Dict, Tuple
import time
import traceback
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# Imports pour BERT médical
try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    BERT_AVAILABLE = True
except ImportError:
    print("  Transformers not available. Installation requise: pip install transformers torch")
    BERT_AVAILABLE = False


class MedicalBERTEvaluator:
    """
    Évaluateur utilisant un modèle BERT médical pour l'encodage sémantique
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        """
        Initialise l'évaluateur BERT médical

        Args:
            model_name: Nom du modèle BERT médical à utiliser
                       Options:
                       - "emilyalsentzer/Bio_ClinicalBERT" (recommandé)
                       - "dmis-lab/biobert-base-cased-v1.1"
                       - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = False

        if BERT_AVAILABLE:
            self._initialize_model()

    def _initialize_model(self):
        """Initialise le modèle BERT médical"""
        try:
            add_log(f"  loading BERT medicale model : {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True
            add_log(f"  BERT  model medicale loaded succesfully {self.device}")
        except Exception as e:
            print(f"  Erreur while loading the bert model : {str(e)}")
            print("  Suggestion: Vérify your cennection")
            self.initialized = False

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode un texte en utilisant BERT médical

        Args:
            text: Texte à encoder

        Returns:
            Vecteur d'embedding numpy
        """
        if not self.initialized or not text.strip():
            return np.zeros(768)  # Dimension par défaut de BERT

        try:
            # Préparation du texte
            text = text.strip()[:512]  # Limiter à 512 tokens

            # Tokenisation
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)

            # Génération des embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Utiliser la moyenne des tokens (pooling moyen)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings.cpu().numpy().squeeze()

        except Exception as e:
            print(f"Erreur while encoding: {str(e)}")
            return np.zeros(768)

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité sémantique entre deux textes

        Args:
            text1: Premier texte
            text2: Deuxième texte

        Returns:
            Score de similarité entre 0 et 1
        """
        if not self.initialized:
            # Fallback vers la méthode textuelle classique
            return self._fallback_similarity(text1, text2)

        # Encoder les deux textes
        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)

        # Calculer la similarité cosinus
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]

        # Normaliser entre 0 et 1
        return max(0.0, min(1.0, (similarity + 1) / 2))

    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Méthode de fallback si BERT n'est pas disponible"""
        return normalize_text_similarity(text1, text2)


# Instance globale de l'évaluateur BERT
bert_evaluator = MedicalBERTEvaluator() if BERT_AVAILABLE else None


def normalize_text(text: str) -> str:
    """Normalise le texte pour une meilleure comparaison"""
    if not text:
        return ""

    # Convertir en minuscules
    text = text.lower()

    # Supprimer les caractères de saut de ligne et espaces multiples
    text = re.sub(r'\s+', ' ', text)

    # Supprimer les caractères de ponctuation en début/fin
    text = text.strip(' \t\n\r.,;:!?-')

    return text


def normalize_text_similarity(prediction: str, ground_truth: str) -> float:
    """
    Méthode de similarité textuelle classique (fallback)
    """
    pred_normalized = normalize_text(prediction)
    truth_normalized = normalize_text(ground_truth)

    # Correspondance exacte normalisée
    if truth_normalized in pred_normalized:
        return 1.0

    # Correspondance par mots-clés
    truth_words = set(truth_normalized.split())
    pred_words = set(pred_normalized.split())

    # Filtrer les mots trop courts
    truth_words = {word for word in truth_words if len(word) > 2}
    pred_words = {word for word in pred_words if len(word) > 2}

    if not truth_words:
        return 0.0

    # Score de correspondance
    intersection = truth_words.intersection(pred_words)
    similarity_score = len(intersection) / len(truth_words)

    # Vérifier les segments importants
    truth_segments = [segment.strip() for segment in truth_normalized.split('.') if len(segment.strip()) > 10]

    segment_matches = 0
    for segment in truth_segments:
        if segment in pred_normalized:
            segment_matches += 1

    if truth_segments:
        segment_score = segment_matches / len(truth_segments)
        similarity_score = max(similarity_score, segment_score)

    return similarity_score


def check_text_similarity(prediction: str, ground_truth: str, threshold: float = 0.7) -> float:
    """
    Vérifie la similarité entre deux textes avec encodage sémantique BERT médical

    Args:
        prediction: Texte de prédiction
        ground_truth: Texte de vérité terrain
        threshold: Seuil de similarité (non utilisé pour le calcul, mais pour référence)

    Returns:
        Score de similarité entre 0 et 1
    """
    if not prediction or not ground_truth:
        return 0.0

    # Utiliser BERT médical si disponible
    if bert_evaluator and bert_evaluator.initialized:
        try:
            # Similarité sémantique via BERT médical
            semantic_similarity = bert_evaluator.calculate_semantic_similarity(prediction, ground_truth)

            # Similarité textuelle classique pour comparaison
            text_similarity = normalize_text_similarity(prediction, ground_truth)

            # Combiner les deux scores (pondération: 70% sémantique, 30% textuelle)
            combined_similarity = 0.7 * semantic_similarity + 0.3 * text_similarity

            # Afficher les détails pour debug
            add_log(f" ✔ semantic similarity (BERT): {semantic_similarity:.4f}")
            add_log(f"✔ textual similarity: {text_similarity:.4f}")
            add_log(f"✔  combined score: {combined_similarity:.4f}")

            return combined_similarity

        except Exception as e:
            print(f"  Error BERT, using fallback: {str(e)}")
            return normalize_text_similarity(prediction, ground_truth)
    else:
        # Fallback vers la méthode textuelle classique
        return normalize_text_similarity(prediction, ground_truth)


def enhanced_similarity_analysis(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Analyse de similarité détaillée avec différentes métriques
    """
    analysis = {
        'semantic_similarity': 0.0,
        'textual_similarity': 0.0,
        'combined_similarity': 0.0,
        'keyword_overlap': 0.0,
        'medical_concepts_overlap': 0.0
    }

    # Similarité sémantique BERT
    if bert_evaluator and bert_evaluator.initialized:
        try:
            analysis['semantic_similarity'] = bert_evaluator.calculate_semantic_similarity(prediction, ground_truth)
        except Exception as e:
            print(f"Erreur similarité sémantique: {e}")

    # Similarité textuelle
    analysis['textual_similarity'] = normalize_text_similarity(prediction, ground_truth)

    # Analyse des mots-clés médicaux
    medical_keywords = extract_medical_keywords(ground_truth)
    if medical_keywords:
        pred_keywords = extract_medical_keywords(prediction)
        overlap = len(medical_keywords.intersection(pred_keywords))
        analysis['keyword_overlap'] = overlap / len(medical_keywords) if medical_keywords else 0.0

    # Score combiné
    if analysis['semantic_similarity'] > 0:
        analysis['combined_similarity'] = (
                0.5 * analysis['semantic_similarity'] +
                0.3 * analysis['textual_similarity'] +
                0.2 * analysis['keyword_overlap']
        )
    else:
        analysis['combined_similarity'] = (
                0.7 * analysis['textual_similarity'] +
                0.3 * analysis['keyword_overlap']
        )

    return analysis


def extract_medical_keywords(text: str) -> set:
    """
    Extrait les mots-clés médicaux potentiels d'un texte
    """
    if not text:
        return set()

    # Patterns pour identifier les termes médicaux
    medical_patterns = [
        r'\b[A-Z][a-z]+itis\b',  # Inflammations
        r'\b[A-Z][a-z]*oma\b',  # Tumeurs
        r'\b[A-Z][a-z]*osis\b',  # Conditions
        r'\b[A-Z][a-z]*pathy\b',  # Maladies
        r'\bDB\d+\b',  # DrugBank IDs
        r'\b\d{3}\.\d\b',  # Codes ICD
        r'\b[A-Z][a-z]{4,}\b'  # Mots longs commençant par majuscule
    ]

    keywords = set()
    for pattern in medical_patterns:
        matches = re.findall(pattern, text)
        keywords.update([match.lower() for match in matches])

    # Ajouter les mots significatifs (>4 caractères)
    words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
    keywords.update(words)

    return keywords


def calculate_relevance(prediction: str, ground_truth: str, question: str = "", threshold: float = 0.7) -> Dict[
    str, float]:
    """
    Calcule la pertinence : La réponse répond-elle à la question ?

    Args:
        prediction: Réponse prédite
        ground_truth: Réponse attendue
        question: Question originale (optionnelle)
        threshold: Seuil de pertinence

    Returns:
        Dict contenant les scores de pertinence
    """
    relevance_scores = {
        'pertinence_score': 0.0,
        'pertinence_binaire': 0,  # 1 si pertinent, 0 sinon
        'pertinence_qualitative': 'Non pertinent'
    }

    if not prediction or not ground_truth:
        return relevance_scores

    # Calculer la similarité entre la prédiction et la vérité terrain
    similarity_score = check_text_similarity(prediction, ground_truth)

    # Si une question est fournie, calculer aussi la pertinence par rapport à la question
    if question:
        question_relevance = check_text_similarity(prediction, question)
        # Combiner les deux scores (70% vérité terrain, 30% question)
        combined_score = 0.7 * similarity_score + 0.3 * question_relevance
    else:
        combined_score = similarity_score

    relevance_scores['pertinence_score'] = combined_score

    # Classification binaire basée sur le seuil
    if combined_score >= threshold:
        relevance_scores['pertinence_binaire'] = 1

        # Classification qualitative plus fine
        if combined_score >= 0.9:
            relevance_scores['pertinence_qualitative'] = 'Très pertinent'
        elif combined_score >= 0.8:
            relevance_scores['pertinence_qualitative'] = 'Pertinent'
        else:
            relevance_scores['pertinence_qualitative'] = 'Modérément pertinent'
    else:
        relevance_scores['pertinence_binaire'] = 0

        if combined_score >= 0.5:
            relevance_scores['pertinence_qualitative'] = 'Peu pertinent'
        else:
            relevance_scores['pertinence_qualitative'] = 'Non pertinent'

    return relevance_scores


def evaluate_individual_query(predictions: List[str], ground_truth: str, query_idx: int, query_text: str,
                              similarity_threshold: float = 0.7, verbose=True) -> Dict[str, float]:
    """Évalue une requête individuelle avec analyse de pertinence"""

    metrics = {}

    # Analyse de similarité détaillée
    if predictions:
        detailed_analysis = enhanced_similarity_analysis(predictions[0], ground_truth)
        metrics.update(detailed_analysis)

        # Calculer la pertinence
        relevance_metrics = calculate_relevance(
            prediction=predictions[0],
            ground_truth=ground_truth,
            question=query_text,
            threshold=similarity_threshold
        )
        metrics.update(relevance_metrics)

    if verbose:
        # Afficher les résultats individuels avec détails sémantiques
        add_log(f"\n************ ÉVALUATION of query {query_idx + 1}: *********")
        add_log(f"-- Query: '{query_text}'")
        add_log(f"-- ground_truth: '{ground_truth}'")
        add_log(f"-- Expected answer: '{predictions[0] if predictions else 'Aucune réponse'}'")
        add_log(f"-- Relevance threshold used: {similarity_threshold}")

        # MÉTRIQUES DE SIMILARITÉ
        if predictions and 'semantic_similarity' in metrics:
            add_log(f"-- SIMILARITY:")
            add_log(f"   Semantic Similarity (BERT): {metrics['semantic_similarity']:.4f}")
            add_log(f"   Textual Similarity: {metrics['textual_similarity']:.4f}")
            add_log(f"   Keyword Overlap: {metrics['keyword_overlap']:.4f}")
            add_log(f"   Combined Score: {metrics['combined_similarity']:.4f}")

        # MÉTRIQUES DE PERTINENCE
        if 'pertinence_score' in metrics:
            add_log(f"-- Relevance:")
            add_log(f"   Relevance Score: {metrics['pertinence_score']:.4f}")
            add_log(f"   Binary Relevance: {' Oui' if metrics['pertinence_binaire'] == 1 else ' Non'}")
            add_log(f"   Qualitative Evaluation: {metrics['pertinence_qualitative']}")

            # Évaluation du seuil
            pertinence_score = metrics['pertinence_score']
            if pertinence_score >= similarity_threshold:
                add_log(f"   Relevant Answer (score ≥ {similarity_threshold})")
            else:
                add_log(f"   Réponse non pertinente (score < {similarity_threshold})")

        # Évaluation qualitative générale
        combined_score = metrics.get('combined_similarity', 0)
        add_log(f"-- GLOBAL EVALUATION :")
        if combined_score >= 0.9:
            add_log("    Excellent semantic match!")
        elif combined_score >= 0.8:
            add_log("    Very good semantic match")
        elif combined_score >= 0.7:
            add_log("    Good semantic match")
        elif combined_score >= 0.5:
            add_log("    Moderate match")
        elif combined_score > 0.2:
            add_log("   Partial match")
        else:
            add_log("    No significant match")

        add_log("-" * 50)

    return metrics


def calculate_average_relevance(all_predictions: List[List[str]],
                                all_ground_truths: List[str],
                                all_questions: List[str] = None,
                                threshold: float = 0.7) -> Dict[str, float]:
    """
    Calcule les métriques de pertinence moyennes pour toutes les requêtes

    Args:
        all_predictions: Liste des listes de prédictions
        all_ground_truths: Liste des vérités terrain
        all_questions: Liste des questions (optionnel)
        threshold: Seuil de pertinence

    Returns:
        Dict avec les métriques moyennes
    """
    if not all_predictions or not all_ground_truths:
        return {
            'pertinence_moyenne': 0.0,
            'taux_pertinence': 0.0,
            'nombre_pertinentes': 0,
            'nombre_total': 0
        }

    pertinence_scores = []
    pertinence_binaires = []

    for i, (predictions, ground_truth) in enumerate(zip(all_predictions, all_ground_truths)):
        if predictions:  # Si il y a au moins une prédiction
            question = all_questions[i] if all_questions and i < len(all_questions) else ""
            relevance_metrics = calculate_relevance(
                prediction=predictions[0],  # Prendre la première prédiction
                ground_truth=ground_truth,
                question=question,
                threshold=threshold
            )

            pertinence_scores.append(relevance_metrics['pertinence_score'])
            pertinence_binaires.append(relevance_metrics['pertinence_binaire'])
        else:
            pertinence_scores.append(0.0)
            pertinence_binaires.append(0)

    # Calculer les métriques moyennes
    avg_relevance = sum(pertinence_scores) / len(pertinence_scores) if pertinence_scores else 0.0
    relevance_rate = sum(pertinence_binaires) / len(pertinence_binaires) if pertinence_binaires else 0.0

    return {
        'pertinence_moyenne': avg_relevance,
        'taux_pertinence': relevance_rate,
        'nombre_pertinentes': sum(pertinence_binaires),
        'nombre_total': len(pertinence_scores)
    }


def initialize_bert_evaluator():
    """Initialise l'évaluateur BERT médical avec gestion d'erreurs"""
    global bert_evaluator

    if not BERT_AVAILABLE:
        print("  BERT not available. Using standard text-based metrics instead.")
        print("  To use medical BERT, install: pip install transformers torch")
        return False

    try:
        bert_evaluator = MedicalBERTEvaluator()
        return bert_evaluator.initialized
    except Exception as e:
        print(f"  Error while initialisation of  BERT: {str(e)}")
        return False


def process_single_query(query_text):
    """
    Fonction qui traite une seule requête en utilisant les modèles pré-chargés
    Support des interactions vaccinales ajouté
    """
    global query_parser, embeddings_dict, combined_graph
    start_time = time.time()

    if not MODELS_LOADED:
        if MODELS_LOADING:
            response_time = time.time() - start_time
            return {
                "error": "Modèles en cours de chargement...",
                "status": "loading",
                "progress": initialization_progress,
                "response_time": f"{response_time:.3f}s"
            }
        else:
            response_time = time.time() - start_time
            return {
                "error": "Modèles non initialisés",
                "status": "not_initialized",
                "response_time": f"{response_time:.3f}s"
            }

    try:
        start_time = time.time()
        add_log(f" ***  Processing the request: '{query_text}'***")

        # Parser la requête avec votre SemanticQueryParser
        query_analysis = query_parser.parse_query(query_text)

        # Vérifier plusieurs types d'entités pour déterminer si on peut traiter la requête
        can_process = (
            query_analysis.get("numeric_entities") or
            query_analysis.get("drugbank_ids") or
            query_analysis.get("drug_names") or
            any("drug_name:" in target for target in query_analysis.get("search_targets", []))
        )

        if can_process:
            # Log du type d'entité trouvée
            if query_analysis.get("numeric_entities"):
                code = query_analysis["numeric_entities"][0]
                add_log(f"\n -> Search in the Knowledge Graph for the ICD code{code}...")
            elif query_analysis.get("drugbank_ids"):
                drug_id = query_analysis["drugbank_ids"][0]
                add_log(f"\n -> Search in the Knowledge Graph for the DrugBank ID {drug_id}...")
            elif query_analysis.get("drug_names"):
                drug_name = query_analysis["drug_names"][0]
                add_log(f"\n -> Search in the Knowledge Graph for the drug{drug_name}...")

            if embeddings_dict is not None and query_analysis is not None:
                # Rechercher les cibles dans les embeddings et extraire les triplets associés
                search_results = query_parser.search_in_embeddings(
                    query_analysis,
                    embeddings_dict,
                    combined_graph=combined_graph
                )

                # Si des triplets ont été extraits, extraire la réponse pertinente
                if search_results and "related_triplets" in search_results:
                    triplets = search_results["related_triplets"]
                    top_entity = search_results.get("top_entity", "")

                    # Utiliser la fonction pour extraire la réponse pertinente
                    query_result = query_parser.extract_query_response(query_analysis, triplets)
                    response_time = time.time() - start_time

                    return {
                        "success": True,
                        "query": query_text,
                        "response": query_result.get("natural_response", "Aucune réponse générée"),
                        "entities_found": top_entity,
                        "query_type": query_result.get("query_type", "unknown"),
                        "status": "ready",
                        "response_time": f"{response_time:.3f}s"
                    }
                else:
                    response_time = time.time() - start_time
                    return {
                        "success": False,
                        "query": query_text,
                        "response": "Aucun triplet trouvé pour cette requête.",
                        "entities_found": "",
                        "status": "ready",
                        "response_time": f"{response_time:.3f}s"
                    }
            else:
                response_time = time.time() - start_time
                return {
                    "success": False,
                    "query": query_text,
                    "response": "Embeddings ou analyse de requête non disponibles.",
                    "entities_found": "",
                    "status": "ready",
                    "response_time": f"{response_time:.3f}s"
                }
        else:
            response_time = time.time() - start_time
            return {
                "success": False,
                "query": query_text,
                "response": "Aucune entité reconnue (code ICD, ID DrugBank ou nom de médicament) détectée dans votre requête.",
                "entities_found": "",
                "status": "ready",
                "response_time": f"{response_time:.3f}s"
            }

    except Exception as e:
        add_log(f"  Erreur lors du traitement: {str(e)}")
        add_log(f"  Trace: {traceback.format_exc()}")
        response_time = time.time() - start_time
        return {
            "error": f"Erreur lors du traitement: {str(e)}",
            "status": "error",
            "response_time": f"{response_time:.3f}s"
        }





import os
import pickle
import time
import traceback
import torch
import json
from datetime import datetime
from flask import Flask
from flask_cors import CORS

# Variables globales pour stocker le modèle pré-entraîné
MODELS_LOADED = False
MODELS_LOADING = False
rdf2vec_model = None
rgcn_model = None
entity_to_idx = None
embeddings_dict = None
combined_graph = None
query_parser = None
initialization_progress = "En attente..."

# Chemins pour la sauvegarde des modèles
MODEL_SAVE_DIR = "saved_models"
RDF2VEC_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "rdf2vec_model.pkl")
RGCN_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "rgcn_model.pth")
EMBEDDINGS_PATH = os.path.join(MODEL_SAVE_DIR, "embeddings_dict.pkl")
ENTITY_IDX_PATH = os.path.join(MODEL_SAVE_DIR, "entity_to_idx.pkl")
METADATA_PATH = os.path.join(MODEL_SAVE_DIR, "model_metadata.json")

# Créer l'application Flask avec le bon chemin pour les templates
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)


def create_model_directory():
    """Créer le répertoire de sauvegarde des modèles s'il n'existe pas"""
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        add_log(f"  Directory created: {MODEL_SAVE_DIR}")


def get_files_hash(kg_files):
    """Calculer un hash des fichiers pour détecter les changements"""
    import hashlib
    hasher = hashlib.md5()

    for filename in kg_files:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                hasher.update(f.read())
        else:
            # Si un fichier n'existe pas, inclure son nom dans le hash
            hasher.update(filename.encode())

    return hasher.hexdigest()


def save_model_metadata(kg_files, training_time):
    """Sauvegarder les métadonnées du modèle"""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "files_hash": get_files_hash(kg_files),
        "training_time": training_time,
        "kg_files": kg_files
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    add_log(f"  Métadonnées sauvegardées: {METADATA_PATH}")


def load_model_metadata():
    """Charger les métadonnées du modèle"""
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Metadata saved: {e}")
    return None


def should_retrain_models(kg_files):
    """Vérifier si les modèles doivent être réentraînés"""
    # Vérifier si tous les fichiers de modèles existent
    model_files = [RDF2VEC_MODEL_PATH, RGCN_MODEL_PATH, EMBEDDINGS_PATH, ENTITY_IDX_PATH]
    if not all(os.path.exists(f) for f in model_files):
        print("  Some model files are missing, retraining required")
        return True

    # Charger les métadonnées
    metadata = load_model_metadata()
    if not metadata:
        print("  No metadata found, retraining required")
        return True

    # Vérifier si les fichiers KG ont changé
    current_hash = get_files_hash(kg_files)
    if metadata.get("files_hash") != current_hash:
        print("  KG files have changed, retraining required")
        return True

    print("  Existing and up-to-date models found")
    return False


def save_models(rdf2vec_model, rgcn_model, embeddings_dict, entity_to_idx):
    """Sauvegarder tous les modèles"""
    try:
        create_model_directory()

        # Sauvegarder RDF2Vec
        with open(RDF2VEC_MODEL_PATH, 'wb') as f:
            pickle.dump(rdf2vec_model, f)
        print(f"  RDF2Vec saved: {RDF2VEC_MODEL_PATH}")

        # Sauvegarder RGCN
        torch.save(rgcn_model.state_dict(), RGCN_MODEL_PATH)
        print(f" RGCN saved: {RGCN_MODEL_PATH}")

        # Sauvegarder embeddings
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f" Embeddings saved: {EMBEDDINGS_PATH}")

        # Sauvegarder entity_to_idx
        with open(ENTITY_IDX_PATH, 'wb') as f:
            pickle.dump(entity_to_idx, f)
        print(f" Entity mapping saved: {ENTITY_IDX_PATH}")

        return True

    except Exception as e:
        print(f"  eroor while saving: {e}")
        return False


def load_models():
    """Charger tous les modèles sauvegardés"""
    try:
        # Charger RDF2Vec
        with open(RDF2VEC_MODEL_PATH, 'rb') as f:
            rdf2vec_model = pickle.load(f)
        print(f" -> RDF2Vec loaded: {RDF2VEC_MODEL_PATH}")

        # Charger embeddings
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f" -> Embeddings loaded: {EMBEDDINGS_PATH}")

        # Charger entity_to_idx
        with open(ENTITY_IDX_PATH, 'rb') as f:
            entity_to_idx = pickle.load(f)
        print(f" -> Entity mapping loaded: {ENTITY_IDX_PATH}")

        # Pour RGCN, nous devons d'abord créer le modèle puis charger les poids
        # Ceci nécessitera les paramètres du modèle (sera fait dans initialize_models_background)

        return rdf2vec_model, embeddings_dict, entity_to_idx

    except Exception as e:
        print(f"  Error while loading: {e}")
        return None, None, None


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os


def normalize_embeddings_dimensions(embeddings_dict: Dict[str, np.ndarray],
                                    target_dim: Optional[int] = None,
                                    method: str = 'truncate') -> Dict[str, np.ndarray]:
    """
    Normalise les dimensions des embeddings pour assurer la compatibilité

    Args:
        embeddings_dict: Dictionnaire des embeddings
        target_dim: Dimension cible (None = utilise la dimension minimale)
        method: 'truncate' (tronquer) ou 'pad' (remplir avec des zéros)

    Returns:
        Dict[str, np.ndarray]: Embeddings avec dimensions normalisées
    """
    if not embeddings_dict:
        return {}

    # Déterminer les dimensions - CORRECTION ICI
    dimensions = []
    for entity, emb in embeddings_dict.items():
        # S'assurer que l'embedding est un array numpy
        if not isinstance(emb, np.ndarray):
            emb = np.array(emb)

        # Aplatir si nécessaire et obtenir la vraie dimension
        if len(emb.shape) > 1:
            emb = emb.flatten()

        dimensions.append(emb.shape[0])
        print(f"  Debug - {entity}: shape original = {embeddings_dict[entity].shape}, après flatten = {emb.shape[0]}")

    if target_dim is None:
        if method == 'truncate':
            target_dim = min(dimensions)
        else:  # pad
            target_dim = max(dimensions)

    print(f"  Normalisation of dimensions to {target_dim} (method: {method})")
    print(f"  Dimensions detected: {set(dimensions)}")

    normalized_embeddings = {}

    for entity, embedding in embeddings_dict.items():
        # S'assurer que c'est un array numpy
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # S'assurer que l'embedding est en 1D
        if len(embedding.shape) > 1:
            embedding = embedding.flatten()

        current_dim = embedding.shape[0]

        if current_dim == target_dim:
            normalized_embeddings[entity] = embedding
        elif current_dim > target_dim and method == 'truncate':
            # Tronquer
            normalized_embeddings[entity] = embedding[:target_dim]
        elif current_dim < target_dim and method == 'pad':
            # Remplir avec des zéros
            padded = np.zeros(target_dim)
            padded[:current_dim] = embedding
            normalized_embeddings[entity] = padded
        else:
            print(f"  Impossible to normalize {entity}: dim={current_dim}, target={target_dim}")
            continue

    print(f"  Normalisation finished: {len(normalized_embeddings)} embeddings")
    return normalized_embeddings


def load_embeddings_from_csv(filepath: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Charge les embeddings depuis un fichier CSV

    Args:
        filepath: Chemin vers le fichier CSV

    Returns:
        Tuple[Dict[str, np.ndarray], List[str]]: (dictionnaire embeddings, liste entités)
    """
    try:
        print(f"  loading of {filepath}...")
        df = pd.read_csv(filepath, index_col=0)
        embeddings_dict = {}
        entities = df.index.tolist()

        print(f"  Form of DataFrame: {df.shape}")
        print(f"  Columns: {len(df.columns)} columns")
        print(f"  Index: {len(df.index)} entities")

        for entity in entities:
            # Convertir la ligne en array numpy
            embedding = df.loc[entity].values.astype(float)
            embeddings_dict[entity] = embedding

            # Debug pour les premières entités
            if len(embeddings_dict) <= 3:
                print(f"  Debug - {entity}: shape = {embedding.shape}, first value = {embedding[0]:.4f}")

        # Vérifier que toutes les dimensions sont cohérentes
        dimensions = [emb.shape[0] for emb in embeddings_dict.values()]
        unique_dims = set(dimensions)

        print(f"  Embeddings loaded: {len(embeddings_dict)} entities")
        print(f"   Unique dimensions found: {unique_dims}")

        if len(unique_dims) == 1:
            print(f" all dimensions are coherente: {list(unique_dims)[0]}")
        else:
            print(f"  Dimensions incoherente detected: {unique_dims}")

        return embeddings_dict, entities

    except Exception as e:
        print(f"  error while loading of  {filepath}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, []




def initialize_models_background():
    """
    Fonction d'initialisation qui charge et entraîne tous les modèles en arrière-plan
    Version corrigée avec ordre d'exécution séquentiel
    """
    global MODELS_LOADED, MODELS_LOADING, initialization_progress
    global rdf2vec_model, rgcn_model, entity_to_idx, embeddings_dict, combined_graph, query_parser

    if MODELS_LOADED:
        print("  Models already loaded*** ")
        return True

    if MODELS_LOADING:
        print("  Models loading...")
        return False

    MODELS_LOADING = True

    try:
        # Démarrer le chronomètre global
        global_start_time = time.time()
        print("  Model initialization running in the background...")

        # Liste des fichiers .nt à traiter
        kg_files = [
            "patients_basic.nt",
            "prescriptions.nt",
            "drug_patients.nt",
            "diagnose_icd_information.nt",
            "ddi.nt",
            "BMI_information.nt",
            "age_gender.nt"
        ]

        # ÉTAPE 1: Vérification de l'existence des fichiers
        initialization_progress = "Vérification of  KG files..."

        print(Fore.BLUE + "Step 1: Vérification of the existance of kh files..." + Fore.RESET)

        missing_files = []
        for filename in kg_files:
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                print(f"  ->  {filename} ({size_mb:.2f} MB)")
            else:
                add_log(f"   {filename} - missing file")
                missing_files.append(filename)

        if missing_files:
            raise FileNotFoundError(f"files missing: {', '.join(missing_files)}")

        # ÉTAPE 2: Vérifier si nous devons réentraîner les modèles
        initialization_progress = "Vérification of existance of models..."

        print(Fore.BLUE + "Step 2: Vérification of models existance..." + Fore.RESET)
        need_retrain = should_retrain_models(kg_files)

        if need_retrain:
            print("  Retraining required – Starting full pipeline...")

            # ÉTAPE 3: Chargement des données KG
            initialization_progress = "loading of all kg files..."

            print(Fore.BLUE + "Step3: loading all  kg files one time..." + Fore.RESET)


            print("=" * 80)
            print("  start of processing kg files")
            print("=" * 80)

            triples, combined_graph = load_all_kg_files_once(kg_files, visualize=True, max_nodes=50, output_dir="kg_visualizations", preprocess=True)
            if triples is None or len(triples) == 0:
                raise ValueError("no triplet loaded from kg files.")

            print(f"  Raw triplets loaded: {len(triples)}")
            add_log(f"  Raw triplets loaded: {len(triples)}")

            # ÉTAPE 4: Entraînement RDF2Vec avec sauvegarde automatique des embeddings
            initialization_progress = "entrainement RDF2Vec... (5-15 min)"

            print(Fore.BLUE + "Step 4:  entrainement of  RDF2Vec model ..." + Fore.RESET)

            rdf2vec_model, entity_to_id, entities = train_rdf2vec(
                triples,
                vector_size=128,
                window=8,
                min_count=1,
                sg=1,
                workers=8,
                save_embeddings=True  # Activé pour sauvegarder en CSV
            )

            add_log( " 🔶 Step 4 finished: RDF2Vec processed succesfully" )

            # ÉTAPE 5: Construction des données RGCN
            initialization_progress = "Construction of  RGCN data ..."

            print(Fore.BLUE + "Step 5: Construction of  RGCN data .." + Fore.RESET)

            data, entity_to_idx, num_relations = build_rgcn_data(
                triples, rdf2vec_model, entity_to_id, embedding_dim=128, batch_process=True
            )


            add_log(" 🔶Step 5 finished:  RGCN  data build ")

            # ÉTAPE 6: Initialisation du modèle RGCN
            initialization_progress = "Initialisation of  RGCN model ..."

            print(Fore.BLUE + " Step 6: Initialisation of  RGCN model.." + Fore.RESET)

            num_bases = min(40, num_relations)
            rgcn_model = RGCNAdvanced(
                in_channels=128,
                hidden_channels=256,
                out_channels=192,
                num_relations=num_relations,
                num_bases=num_bases,
                num_blocks=3,
                attention_heads=8,
                dropout=0.25,
                use_layer_norm=True,
                residual_connections=True
            )

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"  Utilisation du dispositif: {device}")
            rgcn_model = rgcn_model.to(device)
            data = data.to(device)
            add_log(" 🔶 Step 6 FINISHED: Modèle RGCN initialized")

            # ÉTAPE 7: Entraînement RGCN
            initialization_progress = " RGCN  Entrainement... (10-30 min)"

            print(Fore.BLUE + " Step 7: Entraînement of RGCN model (100 époques)..." + Fore.RESET)

            losses, rgcn_model = train_rgcn_improved(
                rgcn_model, data, num_epochs=100, patience=15, lr=0.001, device=device
            )
            add_log(" 🔶 Step 7 finished:  RGCN  model entrained")

            # ÉTAPE 8: Génération des embeddings
            initialization_progress = "Génération of final embeddings..."

            print(Fore.BLUE + " Step 8: Génération of final embeddings..." + Fore.RESET)

            rgcn_model.eval()
            with torch.no_grad():
                updated_embeddings = rgcn_model(data).cpu().numpy()

            embeddings_dict = export_embeddings(
                updated_embeddings,
                entity_to_idx,
                filename="embeddings_improved_kg.csv",
                return_dict=True
            )

            add_log(" 🔶 Step 8 finished: Embeddings generated")


            # ÉTAPE 9: Initialisation du query parser
            initialization_progress = "Initialisation of query parser..."

            print(Fore.BLUE + " Step 9: Initialisation of query parser..." + Fore.RESET)

            try:
                query_parser = SemanticQueryParser()
                add_log("🔶 Step 9 finished: Query parser initialized")
            except Exception as parser_error:
                print(f"  Error query parser: {parser_error}")
                query_parser = None
                add_log("  Step 9: Query parser basic utilized")

            # ÉTAPE 10: Sauvegarde des modèles
            initialization_progress = "Saving models..."

            print(Fore.BLUE + "  Step 10: saving des modèles..." + Fore.RESET)

            training_time = time.time() - global_start_time
            if save_models(rdf2vec_model, rgcn_model, embeddings_dict, entity_to_idx):
                save_model_metadata(kg_files, training_time)
                add_log(" 🔶Step 10 finished: saved models")
            else:
                add_log("  Step 10: Error while saving")

        else:

            print(Fore.BLUE + " loading  models pre-entrained..." + Fore.RESET)

            # ÉTAPE 3 (alternative): Chargement des modèles existants
            initialization_progress = "loading  models pre-entrained..."

            print(Fore.BLUE + " Step 3: loading of models pre-entrained..." + Fore.RESET)

            rdf2vec_model, embeddings_dict, entity_to_idx = load_models()

            if rdf2vec_model is None or embeddings_dict is None or entity_to_idx is None:
                print("  error hile loading - Retraining forced...")
                # Relancer avec réentraînement forcé
                MODELS_LOADING = False
                return initialize_models_background()


            add_log( " 🔶 Step 3 finished:  Base Models  saved (rdf2vec + rgcn )" )

            # ÉTAPE 4: Reconstruction des données pour RGCN
            initialization_progress = "Reconstruction of  RGCN data ..."

            print(Fore.BLUE + " Step 4: Reconstruction of  RGCN data..." + Fore.RESET)

            triples, combined_graph = load_all_kg_files_once(kg_files)
            data, _, num_relations = build_rgcn_data(
                triples, rdf2vec_model, None, embedding_dim=128, batch_process=True
            )
            add_log("🔶 Step 4 finished: RGCN data  rebuild")



            # ÉTAPE 5: Chargement du modèle RGCN
            initialization_progress = "loading of RGCN model..."

            print(Fore.BLUE + " Step 5: loading of RGCN model..." + Fore.RESET)

            num_bases = min(40, num_relations)
            rgcn_model = RGCNAdvanced(
                in_channels=128,
                hidden_channels=256,
                out_channels=192,
                num_relations=num_relations,
                num_bases=num_bases,
                num_blocks=3,
                attention_heads=8,
                dropout=0.25,
                use_layer_norm=True,
                residual_connections=True
            )

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rgcn_model = rgcn_model.to(device)
            rgcn_model.load_state_dict(torch.load(RGCN_MODEL_PATH, map_location=device))
            rgcn_model.eval()
            data = data.to(device)
            add_log(" 🔶 Step 5 finished:  RGCN  model loaded")

            # ÉTAPE 6: Initialisation du query parser
            initialization_progress = "Initialisation of query parser..."

            print(Fore.BLUE + "  Step 6: Initialisation of query parser..." + Fore.RESET)

            try:
                query_parser = SemanticQueryParser()
                print("  🔶 Step 6 finished: Query parser initialized")
            except Exception as parser_error:
                print(f"  Error query parser: {parser_error}")
                query_parser = None
                add_log("  🔶 Step 6: Query parser basic utilized")


        # ÉTAPE FINALE: Évaluation (commune aux deux branches)
        initialization_progress = "Evaluation of modèle..."

        print(Fore.BLUE + " Final Step: Evaluation of  RGCN model ..." + Fore.RESET)

        try:
            rgcn_model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data = data.to(device)

            map_score = evaluate_model(rgcn_model, data, entity_to_idx)
            print(f"  Score MAP: {map_score:.4f}")
            add_log(f" 🔶 Final Step finished: Score MAP = {map_score:.4f}")
        except Exception as eval_error:
            print(f"  Error while  evaluation: {str(eval_error)}")
            add_log(f"  Final Step:  Evaluation Error  = {str(eval_error)}")


        # Marquer les modèles comme chargés
        MODELS_LOADED = True
        MODELS_LOADING = False

        # Temps total d'initialisation
        total_initialization_time = time.time() - global_start_time
        initialization_progress = f"  Initialisation terminée en {total_initialization_time:.2f} seconds"
        add_log(f" 🔶 PIPELINE finished in {total_initialization_time:.2f} seconds")

        # Tests complets

        print(Fore.BLUE + " Initiating complete testing..." + Fore.RESET)
        run_comprehensive_tests(global_start_time)

        return True

    except Exception as e:
        MODELS_LOADING = False
        initialization_progress = f"  initialisation error : {str(e)}"
        print(f" error while initialisation: {str(e)}")
        print(f"  Trace complet: {traceback.format_exc()}")
        add_log(f" PIPELINE ERROR : {str(e)}")
        return False



def run_comprehensive_tests(pipeline_start_time):
    """
    Exécuter les tests complets avec métriques individuelles et temps
    MODIFICATION: Tests d'interactions vaccinales ajoutés + métriques de pertinence
    """
    # MODIFICATION: Tests étendus pour inclure les interactions vaccinales
    test_queries = [
        "what is the categorie of disease of the icd code 432.1 ?",
        "what is the name of the drug with ID DB00530 ?",
        "tell me  about this drug bank id DB00530 ?",
        "does methimazole interact with rotavirus vaccine ?",
        "tell me  about this drug bank id DB01053 ?",
        "Could you tell me more about the drug identified as DB09153 in DrugBank?",
        "What is the drug corresponding to DrugBank ID DB09153?",
        "Could you tell me more about the drug identified as DB00415 in DrugBank?",
        "does sirolimus interact with rotavirus vaccine ?",
        "What is the drug corresponding to DrugBank ID DB00331?"
    ]

    # MODIFICATION: Vérité terrain étendue pour les nouveaux tests
    ground_truths = [
        "CEREBROVASCULAR DISEASE",
        "Erlotinib",
        "A quinazoline derivative and ANTINEOPLASTIC AGENT that functions as a PROTEIN KINASE INHIBITOR for EGFR associated tyrosine kinase. It is used in the treatment of NON-SMALL CELL LUNG CANCER.",
        "The therapeutic efficacy of Rotavirus vaccine can be decreased when used in combination with Methimazole.",
        "Benzylpenicillin (Penicillin G) is narrow spectrum antibiotic used to treat infections caused by susceptible bacteria. It is a natural penicillin antibiotic that is administered intravenously or intramuscularly due to poor oral absorption. Penicillin G may also be used in some cases as prophylaxis against susceptible organisms.",
        "Sodium chloride or table salt is a mineral substance belonging to the larger class of compounds called ionic salts. Salt in its natural form is known as rock salt or halite. Salt is present in vast quantities in the ocean, which has about 35 grams of sodium chloride per litre, corresponding to a salinity of 3.5%. ",
        "Sodium chloride",
        "Ampicillin is a broad-spectrum, semi-synthetic, beta-lactam penicillin antibiotic with bactericidal activity. Ampicillin binds to and inactivates penicillin-binding proteins (PBP) located on the inner membrane of the bacterial cell wall. ",
        "The therapeutic efficacy of Rotavirus vaccine can be decreased when used in combination with Sirolimus.",
        "Metformin"
    ]

    add_log("\n  Query Testing with Relevance Evaluation (Including Vaccine Interactions)")
    add_log("=" * 80)

    test_results = []
    query_times = []
    individual_metrics = []

    # Listes pour stocker les données d'évaluation globale
    all_predictions = []
    all_ground_truths = []
    all_questions = []

    # Temps de début des tests (séparé du pipeline global)
    tests_start_time = time.time()

    # Exécuter les requêtes avec mesure du temps individuel
    for i, test_query in enumerate(test_queries):
        add_log(f"\n *********** EXECUTION OF  QUERY {i + 1}/{len(test_queries)}***********")
        add_log(f" Query: '{test_query}'")
        add_log(f" Expected answer: '{ground_truths[i]}'")

        # NOUVEAU: Identifier le type de test
        if "vaccine" in test_query.lower() and "interaction" in test_query.lower():
            add_log(" Type: Test d'interaction vaccinale")
        elif "interaction" in test_query.lower():
            add_log(" Type: Test d'interaction générale")
        elif "DB" in test_query:
            add_log(" Type: Test DrugBank ID")
        elif any(char.isdigit() for char in test_query):
            add_log("  Type: Test code ICD")

        # Mesurer le temps pour cette requête
        query_start_time = time.time()

        try:
            result = process_single_query(test_query)
            query_end_time = time.time()
            query_duration = query_end_time - query_start_time
            query_times.append(query_duration)

            test_results.append(result)

            if result.get("success"):
                response = result['response']
                query_type = result.get('query_type', 'unknown')

                # Stocker les données pour l'évaluation
                all_predictions.append([response])  # Liste de prédictions (ici une seule)
                all_ground_truths.append(ground_truths[i])
                all_questions.append(test_query)

                # Calculer les métriques individuelles pour cette requête
                try:
                    # MODIFICATION: Utiliser la nouvelle fonction d'évaluation (verbose=True pour affichage détaillé)
                    individual_metric = evaluate_individual_query(
                        predictions=[response],
                        ground_truth=ground_truths[i],
                        query_idx=i,
                        query_text=test_query,
                        similarity_threshold=0.7,  # Seuil de pertinence ajustable
                        verbose=True  # Affichage détaillé dans la fonction
                    )
                    individual_metrics.append(individual_metric)

                except Exception as metric_error:
                    add_log(f"  Error calculating individuell metrics : {str(metric_error)}")
                    individual_metrics.append({})
                    # Ajouter des valeurs par défaut pour l'évaluation globale
                    all_predictions.append([""])
                    all_ground_truths.append(ground_truths[i])
                    all_questions.append(test_query)

            else:
                add_log(f"  Error after {query_duration:.3f}s: {result.get('response', result.get('error'))}")
                individual_metrics.append({})
                # Ajouter des valeurs par défaut pour les erreurs
                all_predictions.append([""])
                all_ground_truths.append(ground_truths[i])
                all_questions.append(test_query)

        except Exception as e:
            query_end_time = time.time()
            query_duration = query_end_time - query_start_time
            query_times.append(query_duration)

            add_log(f"  Exception after {query_duration:.3f}s: {str(e)}")
            test_results.append({"success": False, "error": str(e), "response": ""})
            individual_metrics.append({})
            # Ajouter des valeurs par défaut pour les exceptions
            all_predictions.append([""])
            all_ground_truths.append(ground_truths[i])
            all_questions.append(test_query)

    # NOUVEAU: Calculer les métriques globales de pertinence
    add_log(f"\n***********GLOBAL METRICS  OF RELEVANCE:***********")
    add_log("=" * 50)

    try:
        global_relevance_metrics = calculate_average_relevance(
            all_predictions=all_predictions,
            all_ground_truths=all_ground_truths,
            all_questions=all_questions,
            threshold=0.7
        )

        add_log(f"  Average relevance: {global_relevance_metrics['pertinence_moyenne']:.4f}")
        add_log(f"  Relevance rate: {global_relevance_metrics['taux_pertinence']:.2%}")
        add_log(f"  Relevant answers: {global_relevance_metrics['nombre_pertinentes']}/{global_relevance_metrics['nombre_total']}")

        # Classification globale
        avg_relevance = global_relevance_metrics['pertinence_moyenne']
        if avg_relevance >= 0.8:
            add_log(" global Performance : EXCELLENT")
        elif avg_relevance >= 0.7:
            add_log(" global  Performance : GOOD")
        elif avg_relevance >= 0.5:
            add_log(" global Performance : MODERATE")
        else:
            add_log("global  Performance: low")

    except Exception as global_metric_error:
        add_log(f"  Error calculating global metrics : {str(global_metric_error)}")

    # Temps total des tests
    tests_end_time = time.time()
    total_tests_time = tests_end_time - tests_start_time

    # Statistiques de temps détaillées
    add_log(f"\n *********** DETAILED TIME STATISTICS:************")
    add_log("=" * 40)

    total_query_time = sum(query_times)
    average_query_time = total_query_time / len(query_times) if query_times else 0

    for i, query_time in enumerate(query_times):
        query_type_info = "💉" if "vaccine" in test_queries[i].lower() else "🔍"
        # Ajouter indicateur de pertinence si disponible
        pertinence_info = ""
        if i < len(individual_metrics) and 'pertinence_binaire' in individual_metrics[i]:
            pertinence_info = " ✅" if individual_metrics[i]['pertinence_binaire'] == 1 else " ❌"
        add_log(f"⏰ Requête {i + 1} {query_type_info}: {query_time:.3f}s{pertinence_info}")

    add_log(f"\n  total time of queries: {total_query_time:.3f}s")
    add_log(f"⚡ average time per query: {average_query_time:.3f}s")
    if query_times:
        add_log(f"  Fastest query: {min(query_times):.3f}s")
        add_log(f"  slowest query: {max(query_times):.3f}s")

    # Temps total du pipeline complet
    total_pipeline_time = time.time() - pipeline_start_time
    initialization_time = tests_start_time - pipeline_start_time

    add_log(f"\n*************️ total time of pipeline:***********************")
    add_log("=" * 40)
    add_log(f" Initialisation: {initialization_time:.2f}s")
    add_log(f" Tests: {total_tests_time:.3f}s")
    add_log(f" TOTAL PIPELINE: {total_pipeline_time:.2f}s")

    # Formatage du temps en heures/minutes si nécessaire
    hours, remainder = divmod(total_pipeline_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        add_log(f" Soit: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    elif minutes > 0:
        add_log(f" Soit: {int(minutes)}m {seconds:.2f}s")

    # NOUVEAU: Résumé final avec métriques de pertinence
    add_log(f"\n ***************** FINAL Summary:***************************")
    add_log("=" * 40)
    successful_queries = sum(1 for result in test_results if result.get("success", False))
    add_log(f"   Successful queries: {successful_queries}/{len(test_queries)}")

    if 'global_relevance_metrics' in locals() and global_relevance_metrics:
        add_log(f" pertinent response: {global_relevance_metrics['nombre_pertinentes']}/{global_relevance_metrics['nombre_total']}")
        add_log(f" Score of average  pertinence : {global_relevance_metrics['pertinence_moyenne']:.4f}")

    add_log("=" * 80)

    # Retourner les résultats pour utilisation ultérieure si nécessaire
    return {
        'test_results': test_results,
        'individual_metrics': individual_metrics,
        'global_metrics': global_relevance_metrics if 'global_relevance_metrics' in locals() else {},
        'timing': {
            'total_pipeline_time': total_pipeline_time,
            'total_tests_time': total_tests_time,
            'average_query_time': average_query_time,
            'query_times': query_times
        }
    }
# Fonction utilitaire pour forcer la recréation des modèles
def force_retrain():
    """Forcer le réentraînement en supprimant les modèles sauvegardés"""
    global MODELS_LOADED, MODELS_LOADING

    try:
        import shutil
        if os.path.exists(MODEL_SAVE_DIR):
            shutil.rmtree(MODEL_SAVE_DIR)
            print(f"  Model directory deleted: {MODEL_SAVE_DIR}")

        MODELS_LOADED = False
        MODELS_LOADING = False
        print(" Force retrain enabled – models will be retrained")
        return True
    except Exception as e:
        print(f" Error during deletion: {e}")
        return False



@app.route('/')
def home():
    """Page d'accueil"""
    return render_template('accueill.html')

@app.route('/page1')
def page1():
    """Page 1"""
    return render_template('datapage.html')

@app.route('/page2')
def page2():
    """Page 2"""
    return render_template('pipeline.html')
@app.route('/api/search', methods=['POST'])
def search_api():
    """API endpoint pour traiter les requêtes de recherche"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({"error": "empty query"}), 400

        # Traiter la requête
        result = process_single_query(query)
        return jsonify(result)

    except Exception as e:
        print(f" Error API: {str(e)}")
        print(f"  Trace: {traceback.format_exc()}")
        return jsonify({"error": f"error while processing: {str(e)}"}), 500


@app.route('/api/status')
def status():
    """Vérifier le statut des modèles avec les logs récents"""
    try:
        # Récupérer les 5 derniers logs
        recent_logs = log_history[-5:] if log_history else []

        return jsonify({
            "models_loaded": MODELS_LOADED,
            "models_loading": MODELS_LOADING,
            "status": "ready" if MODELS_LOADED else ("loading" if MODELS_LOADING else "not_started"),
            "progress": initialization_progress,
            "recent_logs": recent_logs,
            "total_logs": len(log_history)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/logs')
def get_logs():
    """Récupérer tous les logs ou les nouveaux logs"""
    try:
        since = request.args.get('since', type=int, default=0)

        # Retourner tous les logs depuis l'index 'since'
        logs_to_return = log_history[since:] if since < len(log_history) else []

        return jsonify({
            "logs": logs_to_return,
            "total_logs": len(log_history),
            "new_count": len(logs_to_return)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Servir les fichiers statiques"""
    return send_from_directory('static', filename)



@app.route('/<path:filename>')
def serve_html_pages(filename):
    """Servir les pages HTML"""
    try:
        if filename.endswith('.html'):
            return render_template(filename)
        else:
            return "Page non trouvée", 404
    except:
        return "Page non trouvée", 404


def main_improved():
    """
    Version qui démarre le serveur immédiatement et charge les modèles en parallèle
    """
    print(" Starting web server...")
    print(" Server accessible at: http://localhost:5000")
    print(" Models are loading in the background...")

    # Démarrer l'initialisation des modèles en arrière-plan
    initialization_thread = threading.Thread(target=initialize_models_background)
    initialization_thread.daemon = True  # Se termine avec le programme principal
    initialization_thread.start()

    # Démarrer le serveur Flask immédiatement
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)


if __name__ == "__main__":
    import sys

    sys.setrecursionlimit(10000)
    main_improved()