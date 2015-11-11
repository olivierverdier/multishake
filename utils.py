#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import wavemap

def get_wavemap(dt=.1, dx=.1, border=wavemap.periodic):
	wm = wavemap.WaveMap(dt=dt, dx=dx, border=border)
	return wm

