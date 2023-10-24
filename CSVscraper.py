#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:33:21 2023

@author: kmark7
"""

#!pip install -q jmd_imagescraper

csv_file = "KutyaFajtak.csv"

last_elements = []

with open(csv_file, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(";")
        if len(parts) >= 2:
            last_element = parts[-1]
            last_elements.append(last_element)

"""
Ez itten arra jó, hogyha nincs időnk egyszerre leszedni a képeket, akkor
megtudjuk állítani, majd az itt látható elemtől folytatni.
"""
#del last_elements[:329]

for element in last_elements:
    print(element)
