#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import codecs
import pprint
import re
import xml.etree.cElementTree as ET
from collections import defaultdict

#引入一些需要的库
#import cerberus

#import schema

OSM_PATH = "honolulu.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"
#将xml解析后写入5️⃣个csv文件
LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
street_type_re = re.compile(r'\S+\.?$', re.IGNORECASE)  #匹配最后一个词
street_types = defaultdict(int)
#正则表达式
#SCHEMA = schema.schema
mapping = { "St": "Street",
            "St.": "Street",
            "Ave":"Avenue",
            "Blvd":"Boulevard",
            "Blvd.":"Boulevard",
            "Dr":"Drive",
            "Dr.":"Drive",
            "Rd.":'Road',
            "Rd":'Road'
            }
# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

#主要编写的函数
def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    node_tags={}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements
    if element.tag == 'node':
        for i in NODE_FIELDS:
            node_attribs[i] = element.attrib[i]
        node_attribs['user']=element.attrib['user'].title() #统一姓名格式
        if int(element.attrib['timestamp'][11:13]) >= 12:     #统一时间戳
            i='pm'
        else:
            i='am'
        node_attribs['timestamp']=element.attrib['timestamp'][:10]+' '+i+' '+element.attrib['timestamp'][11:-1]
        if len(element):
            for child in element:
                tags_dict={}
                if child.tag == 'tag':
                    
                    problems = PROBLEMCHARS.search(child.attrib['k'])

                    if problems:
                        continue
                    else:
                        colon = LOWER_COLON.search(child.attrib['k'])
                        if colon:
                            tags_dict['type'] = child.attrib['k'].split(':',1)[0]
                            tags_dict['key'] = child.attrib['k'].split(':',1)[-1]
                        else:
                            tags_dict['type'] = 'regular'
                            tags_dict['key'] = child.attrib['k']
                        tags_dict['id'] = element.attrib['id']
                        tags_dict['value'] = child.attrib['v']
                        if tags_dict['key']=='postcode':     #修复邮政编码,统一格式
                            tags_dict['value']=re.findall(r"\d+",tags_dict['value'])
                            tags_dict['value']=str(tags_dict['value'])[2:7]
                        if tags_dict['key']=='street':
                            for i,j in mapping.items():#      完善街道名
                                if i in re.findall(street_type_re,tags_dict['value']):
                                    tags_dict['value']=tags_dict['value'].replace(i,j)


                else:
                    continue
                tags.append(tags_dict)
        else:
            tags = []
    elif element.tag=='way':
        for i in WAY_FIELDS:
            way_attribs[i]=element.attrib[i]
        way_attribs['user']=element.attrib['user'].title()
        if int(element.attrib['timestamp'][11:13]) >= 12:
            i='pm'
        else:
            i='am'
        way_attribs['timestamp']=element.attrib['timestamp'][:10]+' '+i+' '+element.attrib['timestamp'][11:-1]

        if len(element):
            for i, nd in enumerate(element.iter('nd')):
                way_node={}
                way_node['node_id']=nd.attrib['ref']
                way_node['id']=element.attrib['id']
                way_node['position']=i
                way_nodes.append(way_node)
            for child in element:
                tags_dict={}
                if child.tag == 'tag':
                    
                    problems = PROBLEMCHARS.search(child.attrib['k'])
                    if problems:
                        print problems
                    else:
                        colon = LOWER_COLON.search(child.attrib['k'])
                        if colon:
                            tags_dict['type'] = child.attrib['k'].split(':',1)[0]
                            tags_dict['key'] = child.attrib['k'].split(':',1)[-1]
                        else:
                            tags_dict['type'] = 'regular'
                            tags_dict['key'] = child.attrib['k']
                        tags_dict['id'] = element.attrib['id']
                        tags_dict['value'] = child.attrib['v']
                        if tags_dict['key']=='postcode':     #修复邮政编码,统一格式
                            tags_dict['value']=re.findall(r"\d+",tags_dict['value'])
                            tags_dict['value']=str(tags_dict['value'])[2:7]
                        if tags_dict['key']=='street':
                            for i,j in mapping.items():#      完善街道名
                                if i in re.findall(street_type_re,tags_dict['value']):
                                    tags_dict['value']=tags_dict['value'].replace(i,j)

                else:
                    continue
                tags.append(tags_dict)
            

            
    # YOUR CODE HERE
    if element.tag == 'node':
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


#def validate_element(element, validator, schema=SCHEMA):
#    """Raise ValidationError if element does not match schema"""
#    if validator.validate(element, schema) is not True:
#        field, errors = next(validator.errors.iteritems())
#        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
#        error_string = pprint.pformat(errors)
        
#        raise Exception(message_string.format(field, error_string))


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        #validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                #if validate is True:
                 #   validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
    # Note: Validation is ~ 10X slower. For the project consider using a small
    # sample of the map when validating.
    process_map(OSM_PATH, validate=False)
