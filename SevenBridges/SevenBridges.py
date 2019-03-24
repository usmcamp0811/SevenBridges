import neo4j
import pandas as pd
from string import Template
import ast
import pytz
import datetime
import warnings
import copy


class SevenBridges(object):
    def __init__(self, data_model=None, import_data=None, import_node=None):
        self.data_model = data_model
        self.import_data = import_data
        self.import_node = import_node

def build_properties(properties):
    """
    function takes in a dictionary of properties and converts them to a string representation used in a Template
    :param properties: key value pairs representing the properties of a node or relationship
    :type properties: dict()
    :return: string containing ${} placehoders that Template can inject the value of the property value
    :rtype: string
    """
    c_prop = "{}"
    for k in properties.keys():
        if properties[k] is None:
            continue
        else:
            if str(properties[k]) == properties[k]:
                c_prop = c_prop.format(k) + ' : "${}"'.format(k) + ", {}"
            elif hasattr(properties[k], 'is_point'):
                if properties[k].to_string() is None:
                    continue
                properties[k] = properties[k].to_string()
                c_prop = c_prop.format(k) + ' : "${}"'.format(k) + ", {}"
            elif hasattr(properties[k], 'is_datetime'):
                properties[k] = properties[k].to_string()
                c_prop = c_prop.format(k) + ' : "${}"'.format(k) + ", {}"
            elif isinstance(properties[k], Point):
                c_prop = c_prop.format(k) + ' : "${}"'.format(k) + ", {}"
            else:
                c_prop = c_prop.format(k) + ' : "${}"'.format(k) + ", {}"
    return "{"+c_prop[:-4]+"}"

def build_labels(labels):
    """
    simple helper function to join a list of labels into a format the is compliant with CQL
    :param labels: one or more labels
    :type labels: list of strings
    :return: string representation of the list passed in.. with ":" seperating the multiple labels
    :rtype: string
    """

    if type(labels) == list:
        return ":".join(labels)
    else:
        return labels

def get_node_def(label, data_model):
    for l in data_model:
        if build_labels(label) == build_labels(l["label"]):
            return l
    warnings.warn(f"{label} is not in the Data Model. `None` was returned.")
    return None

class Node(object):
    """
    A class that holds all the properties of a node including the Cypher representations of it
    """
    def __init__(self, labels=None, properties=None, relationships=[], key=[], required_properties=[], unique_properties=[]):
        """
        Initilize this object with labels and properties, similarly to Py2Neo or other neo4j packages
        :param labels: this is either one or more labels as used in Neo4j
        :type labels: either a list of strings or just a single string
        :param properties: key value store of properties that the node will have
        :type properties: dict
        """
        self.labels = labels
        self.properties = properties


        self.relationships = relationships

        if type(labels) == str:
            self.labels_string = self.labels
            self.labels = self.labels.split(":") # make it be uniform for if we ever reference this attribute
        else: # its gotta be a liist in which case we can use the build labels funciton
            self.labels_string = build_labels(self.labels)
        if properties is not None:
            property_strings = Template(build_properties(self.properties))
            self.property_strings = property_strings.substitute(**self.properties)
        else:
            self.property_strings = None
        self.required_properties = required_properties
        self.unique_properties = unique_properties
        self.__primarykey__ = key
        if labels is not None:
            self.__primarylabel__ = self.labels[0]

    def load_properties_from_series(self, series=pd.Series()):
        series = series.reindex(list(self.properties.values()), axis=1)
        properties = dict()
        for k,v in zip(self.properties.keys(), series.items()):
            properties[k] = v[1]
        self.properties = properties
        property_strings = Template(build_properties(self.properties))
        self.property_strings = property_strings.substitute(**self.properties)

    def __repr__(self):
        id = self.ENTITY()
        return id

    def update_properties(self, properties):
        self.properties = properties
        property_strings = Template(build_properties(self.properties))
        self.property_strings = property_strings.substitute(**self.properties)

    def ENTITY(self, n="n"):
        """
        This is the information that would be used to find or create this node in CQL without MERGE, MATCH or CREATE
        prefixed to it.
        :param n: the alias to be used to refer to this node in CQL
        :type n: string
        :return: the string representation of the node less the CQL action
        :rtype: string
        """

        return f"({n}:{self.labels_string} {self.property_strings})"
        
    def CREATE(self, n="n"):
        """
        generates the Cypher query to create this node in Neo4j
        :param n: the alias to be used to refer to this node in CQL
        :type n: string
        :return: the CQL to make this node in neo4j
        :rtype: string
        """
        return f"CREATE ({n}:{self.labels_string} {self.property_strings}) RETURN {n}"
        
    def MATCH(self, n="n"):
        """
        generates the Cypher query to find this node
        :param n: the alias to be used to refer to this node in CQL
        :type n: string
        :return: the CQL to query this node in neo4j
        :rtype: string
        """
        return f"MATCH ({n}:{self.labels_string} {self.property_strings}) RETURN {n}"
        
    def MERGE(self, n="n"):
        """
        generates the Cypher query to merge this node
        :param n: the alias to be used to refer to this node in CQL
        :type n: string
        :return: the CQL to make this node in neo4j
        :rtype: string
        """
        return f"MERGE ({n}:{self.labels_string} {self.property_strings})"

    def UNIQUE(self, n="n"):
        return f"CREATE CONSTRAINT ON ({n}:{self.labels_string}) ASSERT {n}.isbn IS UNIQUE"

class Relationship():
    """
    A object containing all the information to create or find a relationship
    """
    def __init__(self, node_a=None, relates_to=None, node_b=None, properties=None, rel_tuple=None):
        """

        :param NodeA: The node that is the origin for the relationship
        :type NodeA: Node object
        :param relates_to: the relationship name/type
        :type relates_to: string
        :param NodeB: The node that the edge is directed toward
        :type NodeB: Node object
        :param properties: key values for the properties that the relationship may have.. this is optional
        :type properties: dict
        """
        self.NodeA = node_a # Node Object
        self.NodeB = node_b # Node Object
        self.label = relates_to # string?
        if rel_tuple is not None:
            assert len(rel_tuple) <= 4 # make sure its avalid rel tuple. len 3 == no properties to pass len 4 means properties
            if len(rel_tuple) == 4:
                if properties is None:
                    self.properties = rel_tuple[3]
                    properties = self.properties
                rel_tuple = rel_tuple[:3]
                self.label = rel_tuple[1]
        if properties is None:
            self.properties = {}



        self.rel_tuple = rel_tuple

        if properties is not None:
            property_string = Template(build_properties(self.properties))
            self.property_strings = property_string.substitute(**self.properties)
        else:
            self.property_strings = None

    def load_properties_from_series(self, series=pd.Series()):
        # print("YUT YUT", list(self.properties.values()))
        series = series.copy()
        series = series.reindex(list(self.properties.values()), axis=1)
        # print(series)
        properties = dict()
        for k,v in zip(self.properties.keys(), series.items()):
            # print(v)
            properties[k] = v[1]

        self.properties = properties
        # print("THESE ARE =>", self.properties)
        property_strings = Template(build_properties(self.properties))
        self.property_strings = property_strings.substitute(**self.properties)

    def update_properties(self, properties):
        self.properties = properties
        property_strings = Template(build_properties(self.properties))
        self.property_strings = property_strings.substitute(**self.properties)

    def __repr__(self):
        if self.NodeA is None:
            return str(self.rel_tuple)
        else:
            id = self.MERGE()
            return id

    # def ENTITY(self, n="r"):
    #     """
    #     This is the information that would be used to find or create this node in CQL without MERGE, MATCH or CREATE
    #     prefixed to it.
    #     :param n: the alias to be used to refer to this node in CQL
    #     :type n: string
    #     :return: the string representation of the node less the CQL action
    #     :rtype: string
    #     """
    #     return f"({n}:{self.label} {self.property_strings})"

    def MERGE(self):
        """
        creates the Cypher query to make this relationship
        :return: CQL query to merge the relationship in Neo4j
        :rtype: string
        """
        a = self.NodeA.ENTITY(n="a")
        b = self.NodeB.ENTITY(n="b")
        if self.property_strings is None:
            return f"MATCH {a}, {b} MERGE (a)-[r:{self.label}]->(b) RETURN r"
        else:
            return f"MATCH {a}, {b} MERGE (a)-[r:{self.label} {self.property_strings}]->(b) RETURN r"

"""
DateTime() and Point() are probably the two main reasons I am creating this package. Py2Neo doesn't support
Temporal or Spacial properties, and the Neo4j driver isn't as full featured as Py2Neo. 
"""
class DateTime():
    """
    Create a Neo4j Temporal property value. It returns a string representation of the datetime it is passed
    """
    def __init__(self, date_time=None, timezone="UTC"):
        self.timezone = timezone
        if date_time is None:
            self.datetime = datetime.datetime.now()
        else:
            self.datetime = date_time
            
    def is_datetime(self):
        return True

    def to_string(self):
        return pytz.timezone(self.timezone).localize(self.datetime).strftime("%Y-%m-%dT%H:%M:%S%Z")
    
    
class Point():
    """
    Creates a Neo4j Spatial property value. It returns a string representation of the spatial point in Neo4j speak
    """
    def __init__(self, x, y, z=None, crs=None):
        """

        :param x: your X coordinate
        :type x: has to be a number, float or int should work
        :param y: your Y coodinate
        :type y: has to be a number, float or int should work
        :param z: your Z coordinate (optional)
        :type z: has to be a number, float or int should work
        :param crs: the coordinate grid system you want to use. Defaults to cartesian but you could use WGSI or any other
        supported grid system.
        :type crs: string
        """
        self.X = x
        self.Y = y
        self.Z = z
        self.crs = crs
        if self.Z is None:
            self._3D = False
        else:
            self._3D = True
        
    def to_string(self):
        try:
            float(self.X)
            float(self.Y)
            if self.Z is not None:
                float(self.Z)
        except:
            warn_text = f"""One of the X,Y,Z coords is not a number. `None` was returned.\nX: {self.X}\tY: {self.Y}\tZ: {self.Z}"""
            warnings.warn(warn_text)
            return None
        if self.crs == None:
            if self._3D is False:
                p = f"x: toFloat({self.X}), y: toFloat({self.Y})"
            else:
                p = f"x: toFloat({self.X}), y: toFloat({self.Y}), z: toFloat({self.Z})"
        else:
            if self._3D is False:
                p = f"x: toFloat({self.X}), y: toFloat({self.Y}), crs: '{self.crs}'"
            else:
                p = f"x: toFloat({self.X}), y: toFloat({self.Y}), z: toFloat({self.Z}), crs: '{self.crs}'"
                
        point_temp = Template("point({ $points })")
        return point_temp.substitute(points=p)
    
    def is_point(self):
        return True
    
def parse_str(s):
    try:
        return ast.literal_eval(str(s))
    except:
        return s

def dm_relationships(data_model):
    rel_fields = []
    rels = []
    for node in data_model:
        rs = node.relationships
        for r in rs:
            rels.append(r)
            rel_dm = dict(label=[str(r)], relationships=[str(r)], data_fields=list(r.properties.values()),
                          property_fields=list(r.properties.keys()))
            rel_fields.append(rel_dm)

    return rel_fields, rels

    
def apply_data_model(df, data_model, import_by=None):
    """
    This function takes in a dataframe of data (could be empty..alsong as it has all the columns)
    and a aray of dictionaries representing the data model. This DataMode
    :param df: dataframe of data that will be imported into neo4j
    :param data_model: a liist of dictionaries, could be json like
    :param import_bu: String: name of the column that is the main thing to be imported... so like if you are inporting
    student data, then the student field name wouldbe the import_by.. there shouldnt' be any null values in this field
    :return: multi-level columns based on node, properties nad original field names for the imput dataframe
    """
    # TODO: reexamine the import_by parameter and determine if thats the best thing to do
    rel_fields = []
    rels = []
    for node in data_model:
        rs = node.relationships
        for r in rs:
            rels.append(r)
            data_fields = list(r.properties.values())
            property_fields = list(r.properties.keys())
            rel_fields.append([(r.__repr__(), pf, df) for pf, df in zip(property_fields, data_fields)])
        data_fields = list(node.properties.values())
        property_fields = list(node.properties.keys())
        rel_fields.append([(node.labels_string, pf, df) for pf, df in zip(property_fields, data_fields)])

    mlti_lvl_labels = pd.DataFrame(sum(rel_fields, []), columns=["node", "property", "data_field"])
    unused_fields = list(set(df.columns) - set(mlti_lvl_labels["data_field"]))
    rel_fields.append([("UnusedFields", "ignored", field) for field in unused_fields])
    mlti_lvl_labels = pd.DataFrame(sum(rel_fields, []), columns=["node", "property", "data_field"])

    df = df.reindex(sorted(df.columns), axis=1)
    # making a multilevel index that is the primary key for the import and the actual row index number
    if import_by is not None:
        df.index = pd.MultiIndex.from_tuples([(pk, i) for i,pk in zip(range(len(df[import_by].values.tolist())),df[import_by].values.tolist())], names=["pk", "index"])
    # makes sure if we have extra columns defined in teh nodes that we don't shit the bed when applying our data model to the
    # the dataframe we have in front of us right now
    mlti_lvl_labels = mlti_lvl_labels[mlti_lvl_labels["data_field"].isin(df.columns)]
    mlti_lvl_labels.sort_values("data_field", inplace=True, axis=0)

    df.columns = pd.MultiIndex.from_tuples([tuple(x) for x in mlti_lvl_labels.values], names=["node", "property", "data_field"])

    return df

def create_nodes_and_relationships(df, data_model):
    """
    This generates all the Cypher needed to turn your dataframe into a graph. It does not actually run cypher queries
    :param dm: a dataframe that has been preprocessed by apply_data_model
    :type dm: Pandas DataFrame with multilevel indexes applied
    :param data_model: the definition for how the graph shouldbe created given your data
    :type data_model: list of dictionaries representing nodes
    :return: 2 lists of strings contianing Cypher statements that if run will generate a graph in Neo4j
    :rtype: 2 lists of strings
    """
    # drop the unused fields when creating the nodes.. we don't need a bunch of odd nodes floating around
    # # create
    nodes_to_make = dict()
    rels_to_make = dict()
    # for each record we make node obejects then we make relationships objects
    for ix, data_content in df.iterrows():
        rec_nodes_to_make = dict()
        rec_rels_to_make = dict()
        for model_node in data_model:
            # make a copy of the node model
            NewNode = copy.deepcopy(model_node)
            NewNode.load_properties_from_series(data_content)
            if pd.DataFrame(NewNode.properties, index=[0]).dropna(how='all').shape[0] > 0:
                rec_nodes_to_make[NewNode.labels_string] = NewNode

        for label, new_node in rec_nodes_to_make.items():
            for rel in new_node.relationships:
                rel.NodeA = new_node
                if rel.rel_tuple[2] in rec_nodes_to_make.keys():
                    rel.NodeB = rec_nodes_to_make[rel.rel_tuple[2]]
                    rel.label = rel.rel_tuple[1]
                    rel.load_properties_from_series(data_content)
                    rec_rels_to_make[rel.__repr__()] = rel

        nodes_to_make[ix] = rec_nodes_to_make
        rels_to_make[ix] = rec_rels_to_make
    return nodes_to_make, rels_to_make
