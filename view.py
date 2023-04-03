from evtool.dvs import DvsFile
from evtool.utils import Player

# load event data
data = DvsFile.load("./results/D-END/Architecture/Architecture-ND00-1.pkl")

# load data into player and choose core
player = Player(data, core='matplotlib')

# view data
player.view("25ms", use_aps=True)
