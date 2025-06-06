from flask import Flask, request, render_template
import torch
import numpy as np
from model import DummyChemModel

app = Flask(__name__)

# Charger le modèle
model = DummyChemModel()
model.load_state_dict(torch.load("best.pt", map_location=torch.device('cpu')))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    reaction_name = ""
    reactif1 = ""
    reactif2 = ""
    produit = ""
    
    if request.method == 'POST':
        # Récupérer les données du formulaire
        reaction_name = request.form['reactionName']
        reaction_type = request.form.get('reactionType', 'Non spécifié')
        reactif1 = request.form['reactif1']
        reactif2 = request.form.get('reactif2', '')
        produit = request.form['produit']
        
        # Générer des données aléatoires pour la démonstration
        input_data = np.random.rand(1, 10).astype(np.float32)
        tensor = torch.tensor(input_data)
        
        with torch.no_grad():
            # Obtenir la prédiction
            pred = model(tensor).numpy()[0]
            
            # Formater les résultats
            prediction = {
                "Rendement": f"{pred[0]*100:.1f}%",
                "Température optimale": f"{150 + pred[1]*50:.0f} °C",
                "Durée réaction": f"{60 + pred[2]*30:.0f} min"
            }
    
    return render_template(
        'index.html', 
        prediction=prediction,
        reaction_name=reaction_name,
        reactif1=reactif1,
        reactif2=reactif2,
        produit=produit
    )

if __name__ == '__main__':
    app.run(debug=True)