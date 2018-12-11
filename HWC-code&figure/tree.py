from __future__ import print_function
from collections import OrderedDict
# build the tree
outlook = OrderedDict()
outlook["overcast"] = {"yes":None}
outlook["rain"]={"false":{"yes":None}, "true":{"no":None}}
outlook["sunny"]={"high":{"no":None}, "low":{"yes":None}}

# tranverse the tree
cur = outlook
for con1 in outlook:
    if con1=="rain":
        second_node = "Wind"
    elif con1=="sunny":
        second_node = "Humidity"
    for con2 in outlook[con1]: 
        if outlook[con1][con2] is None:
            print("If Outlook="+con1+", Golf="+con2)
        else:
            for con3 in outlook[con1][con2]:
                if outlook[con1][con2][con3] is None:
                    print("If Outlook="+con1+" and "+second_node+"="+con2+", Golf="+con3)