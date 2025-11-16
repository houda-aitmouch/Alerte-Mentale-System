# Alerte Mentale – Système d’Alerte Précoce en Santé Mentale

**Un message. Une seconde. Une vie sauvée.**

---

## Description

Système intelligent de **détection précoce de détresse mentale** à partir d’un message texte.  
- **SVM** → détecte l’état psychologique (70,66 % accuracy)  
- **Enrichissement** → ajoute `[STATUS: Suicidal]`  
- **Régression Logistique** → prédit l’action (82,9 % accuracy)  
- **Streamlit** → interface locale avec **bouton 112**  
- **FastAPI** → API REST (prêt pour mobile)

---

## Démo (Capture)

![Interface Streamlit](figures/chargement_dataset.png)

---

## Installation (Locale – Maroc)

```bash
git clone https://github.com/ton-pseudo/Alerte-Mentale-System.git
cd Alerte-Mentale-System
pip install -r requirements.txt
streamlit run app.py
