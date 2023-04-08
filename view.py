from evtool.dvs import DvsFile
from evtool.utils import Player

# load event data
data = DvsFile.load(
    "/media/kuga/瓜果山/results/final/evflow/D-END/Ride/Ride-ND16-1.pkl")

print(data['events'].shape)

# load data into player and choose core
player = Player(data, core='matplotlib')

# view data
player.view("25ms", use_aps=False)
