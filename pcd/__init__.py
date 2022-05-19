from flask import Flask
from pcd.models.cardmodel import CardModel
from pcd.models.cornermodel import CornerModel

app = Flask(__name__)

cardmodel = CardModel()
cardmodel.load_wights('models_weights/resnet50_1.hdf5')

cornermodel = CornerModel(alpha=0.7, init_alpha=0.2, stop=32)
cornermodel.load_wights('models_weights/cornermodel.hdf5')

import pcd.views
import pcd.utils
