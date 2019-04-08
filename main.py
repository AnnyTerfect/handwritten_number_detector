#-*- coding: utf8 -*-

from capture import capture
from hook import hook
from keras.models import load_model

model = load_model('model.h5')

capture(model, hook)