import neo4j
import pandas as pd
from string import Template
import ast
import pytz
import datetime
import warnings


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

def create_nodes_and_relationships(dm, data_model):
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
    dm = dm.drop(columns=["UnusedFields"], level="node")
    node_defs = dict()
    rel_defs = dict()
    # Makes nodes
    for pk in list(set(dm.index)):
        # iterate through the index.. aka the records we are importing.. ie student
        nodes = dm.loc[[pk]]
        node_defs[pk] = dict()
        rel_defs[pk] = dict()
        for labels in list(set(nodes.T.index.get_level_values("node"))):
            # iterate over all the labels.. aka nodes in the data model
            try:
                is_rel = type(eval(labels)) == tuple
            except:
                is_rel = False
            if is_rel is True:
                rel_data = nodes[labels]
                rel_data.columns = rel_data.columns.get_level_values("property")
                rel_defs[pk][labels] = []
                for ix in range(rel_data.shape[0]):
                    # iterate over every node of the nodes.. may be just one.. may be many if we have something like two dads

                    properties = rel_data.iloc[ix].dropna().to_dict() # test that I dont need to do dropna conditions like all or any or i dunno
                    if len(properties.keys()) == 0:
                        continue
                    if len(properties.keys()) > 0:
                        # make sure we don't attempt to make an empty node for no reason

                        rel_defs[pk][labels].append(properties)   # save node objects in a dictionary indexed by the levels of this loop

            else:
                node_data = nodes[labels]
                node_data.columns = node_data.columns.get_level_values("property")
                node_defs[pk][labels] = []
                for ix in range(node_data.shape[0]):
                    # iterate over every node of the nodes.. may be just one.. may be many if we have something like two dads

                    properties = node_data.iloc[ix].dropna().to_dict() # test that I dont need to do dropna conditions like all or any or i dunno

                    if len(properties.keys()) > 0:
                        nd = get_node_def(labels, data_model)
                        # make sure we don't attempt to make an empty node for no reason
                        node = Node(labels, properties, key=nd["key"], required_properties=nd["required_properties"], unique_properties=nd["unique_properties"])
                        node_defs[pk][labels].append(node)   # save node objects in a dictionary indexed by the levels of this loop

    _, relationships = dm_relationships(data_model)

    for pk in list(set(dm.index)):
        for rel in relationships:
            for node in node_defs[pk].keys():
                try:
                    a = node_defs[pk][rel[0]]
                    relationship = rel[1]
                    b = node_defs[pk][rel[2]]

                except:
                    continue


    node_dfs = pd.DataFrame(node_defs)
    node_dfs.columns = pd.MultiIndex.from_tuples(list(node_defs.keys()), names=["pk", "index"])
    rel_dfs = pd.DataFrame(rel_defs)
    rel_dfs.columns = pd.MultiIndex.from_tuples(list(node_defs.keys()), names=["pk", "index"])
    _relationships_to_make_maybe_dupes = []
    for pk in node_dfs.columns.get_level_values("pk"):
        for _relationship in relationships:
            a_node = node_dfs.xs(_relationship[0])[pk]
            b_node = node_dfs.xs(_relationship[2])[pk]
            a_ixs = a_node.index.get_level_values("index")
            b_ixs = b_node.index.get_level_values("index")
            rel = _relationship[1]
            for a_ix in a_ixs:
                rel_ix = (pk, a_ix)

                a = a_node.xs(a_ix)

                if len(a) == 0:
                    continue
                else:
                    a = a[0]
                for b_ix in b_ixs:
                    b = b_node.xs(b_ix)
                    if len(b) == 0:
                        continue
                    else:
                        b = b[0]
                    rel_col_name = str((a.labels_string, rel, b.labels_string))

                    if rel_col_name in dm.columns.get_level_values("node"):
                        _props = rel_dfs.xs(rel_col_name).loc[rel_ix][0]
                        # make rel with props.
                        relationship = Relationship(a, rel, b, properties=_props)

                    else:
                        relationship = Relationship(a, rel, b)
                    _relationships_to_make_maybe_dupes.append(relationship)


    # complicated set like function basically cause the objects don't equal but we rely on the repr method to compare thigns
    # TODO: Figure out why objects with the same __repr__ are not equal
    relationships_to_make = [_relationships_to_make_maybe_dupes[0]]
    for z in _relationships_to_make_maybe_dupes:
        _skip = z.__repr__() == relationships_to_make[-1].__repr__()
        if _skip is False:
            relationships_to_make.append(z)

    # convert the df of nodes to a list to make
    nodes_to_make = []
    for node_type in node_dfs.index:
        node_dfs.loc[node_type]
        nodes_to_make.append(node_dfs.loc[node_type].tolist())
    nodes_to_make = sum(sum(nodes_to_make, []),[])
    return nodes_to_make, relationships_to_make


if __name__ == "__main__":
    import datetime
    import pandas as pd

    student = Node(labels=["Person", "Student"],
                   properties=dict(id="student_id",
                                   name="student_name",
                                   dob="student_dob",
                                   grade="student_grade",
                                   gpa="student_gpa",
                                   test="notthere"
                                   ),
                   relationships=[
                       Relationship(rel_tuple=("Person:Student", "_IS_MEMBER_OF_", "Class")),
                       Relationship(rel_tuple=("Person:Student", "_ATTENDS_", "School"))
                   ],
                   key=["id"],
                   required_properties=["id", "name", "dob", "grade"],
                   unique_properties=["id"]
                   )

    teacher = Node(labels=["Person", "Teacher"],
                   properties=dict(id="teacher_id",
                                   name="teacher_name",
                                   degree="teacher_degree",
                                   employment_date="teacher_employment_date"
                                   ),
                   relationships=[
                       Relationship(rel_tuple=("Person:Teacher", "_TEACHES_", "Class",
                                               dict(years_teaching="years_teaching")
                                               )
                                    ),
                       Relationship(rel_tuple=("Person:Teacher", "_TEACHES_", "Person:Student"))
                   ],
                   key=["id"],
                   required_properties=["id", "name"],
                   unique_properties=["id"]
                   )

    class_node = Node(labels=["Class"],
                      properties=dict(number="class_number",
                                      subject="subject",
                                      grade="grade"
                                      ),
                      relationships=[
                          Relationship(rel_tuple=("Class", "_IS_AT_", "School",
                                                  dict(room_number="class_room_number")
                                                  )
                                       )
                      ],
                      key=[],
                      required_properties=["number", "subject", "grade"],
                      unique_properties=[]
                      )

    father = Node(labels=["Person", "Parent", "Father"],
                  properties=dict(name="father_name",
                                  age="father_age"
                                  ),
                  relationships=[
                      Relationship(rel_tuple=("Person:Parent:Father", "_PARENT_OF_", "Person:Student",
                                              dict(type="father_type",
                                                   pickup="f_pickup")
                                              )
                                   )
                  ],
                  key=[],
                  required_properties=["name"],
                  unique_properties=[]
                  )

    mother = Node(labels=["Person", "Parent", "Mother"],
                  properties=dict(name="mother_name",
                                  age="mother_age"
                                  ),
                  relationships=[
                      Relationship(rel_tuple=("Person:Parent:Mother", "_PARENT_OF_", "Person:Student",
                                              dict(type="mother_type",
                                                   pickup="m_pickup")
                                              )
                                   )
                  ],
                  key=[],
                  required_properties=["name"],
                  unique_properties=[]
                  )

    school = Node(labels=["School"],
                  properties=dict(name="school_name",
                                  address="school_address"
                                  ),
                  relationships=[],
                  key=["name", "address"],
                  required_properties=["name", "address"],
                  unique_properties=["name", "address"]
                  )

    data_model = [student, teacher, class_node, father, mother, school]

    school_data = pd.DataFrame([
        {
            "student_id": "1234", "student_name": "Matt Camp", "studen_dob": "12/01/2001", "student_grade": 12,
            "student_gpa": 3.8,
            "teacher_id": "t234f", "teacher_name": "Kathy Fisher", "teacher_degree": "Math",
            "teacher_employment_date": "08/01/1998",
            "class_number": "MA301", "class_name": "Algebra II", "class_grade": 12,
            "father_name": "Paul Camp", "father_age": 72,
            "mother_name": "Helen Camp", "mother_age": 49,
            "school_name": "Notre Dame High School", "school_address": "1234 Harvard Way, Chattanooga TN 35761",
            "class_room_number": 303,
            "years_teaching": 15,
            "father_type": "Birth", "f_pickup": False,
            "mother_type": "Birth", "m_pickup": True,
            "NotUsedData": "Test"
        },
        {
            "student_id": "1233434", "student_name": "Matt Billings", "studen_dob": "2/01/2001", "student_grade": 12,
            "student_gpa": 3.7,
            "teacher_id": "t234f", "teacher_name": "Kathy Fisher", "teacher_degree": "Math",
            "teacher_employment_date": "08/01/1998",
            "class_number": "MA301", "class_name": "Algebra II", "class_grade": 12,
            "father_name": "Paul Billings", "father_age": 52,
            "mother_name": "Helen Billings", "mother_age": 43,
            "school_name": "Notre Dame High School", "school_address": "1234 Harvard Way, Chattanooga TN 35761",
            "class_room_number": 303,
            "years_teaching": 15,
            "father_type": "Step", "f_pickup": True,
            "mother_type": "Birth", "m_pickup": True
        },
        {
            "student_id": "12534", "student_name": "Jesse Jones", "studen_dob": "12/01/2004", "student_grade": 9,
            "student_gpa": 3.2,
            "teacher_id": "t232323234f", "teacher_name": "Marie Daily", "teacher_degree": "English",
            "teacher_employment_date": "08/01/1999",
            "class_number": "EN230", "class_name": "English", "class_grade": 12,
            "father_name": "Bob Jones", "father_age": 32,
            "mother_name": "Mary Jones", "mother_age": 39,
            "school_name": "Sudden Valley High School", "school_address": "1234 Jones Road, Nashville TN 35761",
            "class_room_number": 43,
            "years_teaching": 3,
            "father_type": "Birth", "f_pickup": True,
            "mother_type": "Birth", "m_pickup": True
        },
        {
            "student_id": "12324", "student_name": "Billy Jones", "studen_dob": "12/01/2003", "student_grade": 9,
            "student_gpa": 2.9,
            "teacher_id": "twsv234f", "teacher_name": "Kathy Freeley", "teacher_degree": "History",
            "teacher_employment_date": "08/01/1988",
            "class_number": "HIS432", "class_name": "American History", "class_grade": 11,
            "father_name": "Gary Carell", "father_age": 49,
            "school_name": "Sudden Valley High School", "school_address": "1234 Jones Road, Nashville TN 35761",
            "class_room_number": 423,
            "years_teaching": 1,
            "father_type": "Birth", "f_pickup": True
        },
        {
            "student_id": "12324", "student_name": "Billy Jones", "studen_dob": "12/01/2003", "student_grade": 9,
            "student_gpa": 2.9,
            "father_name": "Ace Colbert", "father_age": 32,
            "father_type": "Step", "f_pickup": True
        }

    ])
    school_data
    dm = apply_data_model(school_data, data_model)
    dm.T.sort_values([('node')], ascending=False)

    # # create
    nodes_to_make = dict()
    # for model_node in data_model:
    #     # TODO: figure a way to not have to do extra loop here
    #     nodes_to_make[model_node.labels_string] = []
    test = []
    df = school_data
    for ix, data_content in df.iterrows():
        rec_nodes_to_make = dict()
        for model_node in data_model:
            properties = dict()
            # make a copy of the node model
            NewNode = Node()
            NewNode.__dict__ = model_node.__dict__.copy()
            NewNode.load_properties_from_series(data_content)
            rec_nodes_to_make[NewNode.labels_string] = NewNode

        for label, new_node in rec_nodes_to_make.items():
            rels = []
            for rel in new_node.relationships:
                #             print(rel.properties)
                rel.load_properties_from_series(data_content)
                rels.append(rel)
                test.append(rels.copy())
            new_node.relationships = rels


        #         nodes_to_make[model_node.labels_string].append(NewNode)
        nodes_to_make[ix] = rec_nodes_to_make


    print(nodes_to_make[0]["Person:Teacher"].relationships[0].properties)
    print(test[2][0].properties)
    # nodes_to_make