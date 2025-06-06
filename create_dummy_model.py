import torch
import torch.nn as nn

# Définition d'un modèle minimaliste
class DummyChemModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 3)  # 10 entrées, 3 sorties (ex: rendement, température, durée)
    
    def forward(self, x):
        return self.layer(x)

# Création et sauvegarde du modèle
model = DummyChemModel()
torch.save(model.state_dict(), "best.pt")
print("Fichier best.pt créé avec succès!")