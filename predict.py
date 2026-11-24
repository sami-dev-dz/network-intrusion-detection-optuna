#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT D'INFÉRENCE & PRÉDICTION TEMPS-RÉEL — NIDS (UNSW-NB15)
Modèle Champion : XGBoost Optimisé (Hyperparamètres : Optuna / Random Search)
================================================================================
Auteurs : GHOUL Sami & BELGHACHE Walid (Université de Béjaïa)
"""

import sys
import os
import time
import argparse
import joblib
import pandas as pd
import numpy as np


def load_champion_pipeline(model_path="models/ids_xgboost_champion.joblib"):
    """Charge le modèle et l'ensemble des transformateurs ajustés (sans fuite de données)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable à l'emplacement : {model_path}")
    
    artifact = joblib.load(model_path)
    return artifact["model"], artifact["encoder"], artifact["scaler"], artifact["selected_cols"]


def preprocess_and_predict(df_input, model, encoder, scaler, selected_cols):
    """
    Applique le prétraitement rigoureux sans fuite d'information et génère
    les prédictions avec probabilités de menace et mesure de latence.
    """
    df = df_input.copy()
    
    # 1. Nettoyage des colonnes parasites ou sources de fuite
    cols_to_drop = [c for c in ['id', 'attack_cat', 'label'] if c in df.columns]
    X_raw = df.drop(columns=cols_to_drop)
    
    # Remplacement des valeurs infinies éventuelles
    num_cols = X_raw.select_dtypes(include=[np.number]).columns
    X_raw[num_cols] = X_raw[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    t0 = time.perf_counter()
    
    # 2. Encodage One-Hot (catégories inconnues ignorées sans lever d'erreur)
    X_enc = pd.DataFrame(
        encoder.transform(X_raw),
        columns=encoder.get_feature_names_out(),
        index=df.index
    )
    
    # 3. Standardisation des variables numériques
    X_scaled = X_enc.copy()
    num_scale_cols = scaler.feature_names_in_
    X_scaled[num_scale_cols] = scaler.transform(X_enc[num_scale_cols])
    
    # 4. Sélection des 46 variables discriminantes
    X_final = X_scaled[selected_cols]
    
    # 5. Inférence XGBoost
    predictions = model.predict(X_final)
    probabilities = model.predict_proba(X_final)[:, 1]
    
    elapsed_total = time.perf_counter() - t0
    latency_per_sample_us = (elapsed_total / max(len(df), 1)) * 1e6
    
    results = pd.DataFrame({
        "Statut_Pred": [" ATTAQUE" if p == 1 else " NORMAL" for p in predictions],
        "Score_Menace": probabilities.round(4),
        "Decision_Binaire": predictions
    }, index=df.index)
    
    return results, elapsed_total, latency_per_sample_us


def run_demo():
    print("=" * 78)
    print("   NIDS CYBERSECURITY INFERENCE ENGINE — MODÈLE CHAMPION XGBOOST")
    print("=" * 78)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "ids_xgboost_champion.joblib")
    data_test_path = os.path.join(base_dir, "data", "UNSW_NB15_testing-set.parquet")
    
    print(f"Chargement du pipeline depuis : {model_path} ...")
    model, encoder, scaler, selected_cols = load_champion_pipeline(model_path)
    print(" Pipeline chargé avec succès.")
    print(f"  • Modèle       : {type(model).__name__}")
    print(f"  • Variables    : {len(selected_cols)} caractéristiques filtrées")
    
    if os.path.exists(data_test_path):
        print(f"\nChargement d'un échantillon du jeu de test holdout : {data_test_path} ...")
        df_test = pd.read_parquet(data_test_path)
        sample = df_test.sample(n=10, random_state=42)
    else:
        print("\nCréation d'un échantillon synthétique pour démonstration...")
        sample = pd.DataFrame({
            "dur": [0.00001, 1.25, 0.05, 0.000002, 2.1],
            "proto": ["tcp", "udp", "tcp", "udp", "tcp"],
            "service": ["http", "dns", "-", "dns", "ftp"],
            "state": ["FIN", "CON", "FIN", "INT", "FIN"],
            "spkts": [2, 10, 4, 2, 50],
            "dpkts": [0, 8, 4, 0, 45],
            "sbytes": [120, 520, 240, 100, 4200],
            "dbytes": [0, 840, 260, 0, 3100],
            "rate": [50000.0, 12.5, 80.0, 500000.0, 40.0],
            "sttl": [254, 31, 62, 254, 31],
            "dttl": [0, 29, 252, 0, 29],
            "sload": [180000.0, 4200.0, 15000.0, 850000.0, 12000.0],
            "dload": [0.0, 6500.0, 18000.0, 0.0, 9500.0],
            "sloss": [0, 0, 0, 0, 1],
            "dloss": [0, 0, 0, 0, 1],
            "sinpkt": [0.01, 12.0, 5.0, 0.001, 20.0],
            "dinpkt": [0.0, 10.0, 4.5, 0.0, 18.0],
            "sjit": [0.0, 1.2, 0.5, 0.0, 2.4],
            "djit": [0.0, 0.8, 0.4, 0.0, 1.5],
            "swin": [0, 0, 255, 0, 255],
            "stcpb": [0, 0, 123456, 0, 654321],
            "dtcpb": [0, 0, 234567, 0, 765432],
            "dwin": [0, 0, 255, 0, 255],
            "tcprtt": [0.0, 0.0, 0.02, 0.0, 0.03],
            "synack": [0.0, 0.0, 0.01, 0.0, 0.015],
            "ackdat": [0.0, 0.0, 0.01, 0.0, 0.015],
            "smean": [60, 52, 60, 50, 84],
            "dmean": [0, 105, 65, 0, 68],
            "trans_depth": [0, 0, 0, 0, 0],
            "response_body_len": [0, 0, 0, 0, 0],
            "ct_srv_src": [1, 2, 1, 10, 2],
            "ct_state_ttl": [2, 0, 1, 2, 0],
            "ct_dst_ltm": [1, 2, 1, 5, 2],
            "ct_src_dport_ltm": [1, 1, 1, 5, 1],
            "ct_dst_sport_ltm": [1, 1, 1, 5, 1],
            "ct_dst_src_ltm": [1, 2, 1, 10, 2],
            "is_ftp_login": [0, 0, 0, 0, 1],
            "ct_ftp_cmd": [0, 0, 0, 0, 1],
            "ct_flw_http_mthd": [0, 0, 0, 0, 0],
            "ct_src_ltm": [1, 2, 1, 5, 2],
            "ct_srv_dst": [1, 2, 1, 10, 2],
            "is_sm_ips_ports": [0, 0, 0, 0, 0]
        })
    
    print(f"\nLancement de la prédiction sur {len(sample)} flux réseau ...\n")
    results, elapsed, latency = preprocess_and_predict(sample, model, encoder, scaler, selected_cols)
    
    if "label" in sample.columns:
        results["Vérité_Terrain"] = ["Attaque (1)" if y == 1 else "Normal (0)" for y in sample["label"]]
        acc = (results["Decision_Binaire"] == sample["label"]).mean() * 100
        print(f"Précision sur cet échantillon : {acc:.1f}%\n")
    
    print("-" * 78)
    print(results.to_string())
    print("-" * 78)
    print(f"  Temps total d'exécution : {elapsed*1000:.2f} ms")
    print(f" Latence par flux       : {latency:.2f} µs/flux (~{int(1e6/latency):,} flux/seconde)")
    print("=" * 78)
    print(" Démonstration terminée avec succès.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inférence NIDS avec le modèle XGBoost Champion")
    parser.add_argument("--file", type=str, default=None, help="Chemin vers un fichier CSV ou Parquet à analyser")
    args = parser.parse_args()
    
    if args.file and os.path.exists(args.file):
        df_in = pd.read_parquet(args.file) if args.file.endswith(".parquet") else pd.read_csv(args.file)
        model, encoder, scaler, selected_cols = load_champion_pipeline()
        res, el, lat = preprocess_and_predict(df_in, model, encoder, scaler, selected_cols)
        print(res.head(20).to_string())
        print(f"\nAnalysé {len(df_in):,} flux en {el:.2f}s ({lat:.2f} µs/flux)")
    else:
        run_demo()
