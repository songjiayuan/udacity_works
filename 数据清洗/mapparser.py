#!/usr/bin/env python
# -*- coding:utf-8 -*-
import xml.etree.cElementTree as ET 
import pprint

def count_tags(filename):
	ui=[]
	tags={}
	for event,elem in ET.iterparse(filename):
		if elem.tag not in ui:
			ui.append(elem.tag)
			tags[elem.tag]=1
		else:
			tags[elem.tag]+=1
	print tags
count_tags('honolulu.osm')